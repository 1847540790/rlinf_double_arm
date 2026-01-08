#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IsaacLab 双机械臂仿真端服务器

用法：
python step5_encapsulate_sim_double_arm.py \
  --host 0.0.0.0 --port 8445 \
  --train-host 10.8.0.1 --train-port 8446 \
  --handshake-token 123 --send-handshake \
  --num-envs 8 \
  --image-height 224 --image-width 224 \
  --isaaclab-url http://127.0.0.1:8000 \
  --arm-id 0 \
  --save-images

观测值格式：
{
    "policy": {
        "Camera_left": torch.Tensor,  # shape: (num_envs, H, W, 3), dtype=uint8
        "Camera_right": torch.Tensor,  # shape: (num_envs, H, W, 3), dtype=uint8
        "Robot_state": torch.Tensor,   # shape: (num_envs, 14), dtype=float64
    }
}

14维Robot_state组成：
- robot0_eef_pos (3维): 左手位置 [x, y, z]
- robot0_eef_rot_axis_angle (3维): 左手姿态 [ax, ay, az]
- robot0_gripper_width (1维): 左手夹爪宽度
- robot1_eef_pos (3维): 右手位置 [x, y, z]
- robot1_eef_rot_axis_angle (3维): 右手姿态 [ax, ay, az]
- robot1_gripper_width (1维): 右手夹爪宽度
"""

import argparse
import base64
import io
import pickle
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from flask import Flask, jsonify, request as flask_request
import logging

# 为了在直接运行脚本时找到项目内的模块，动态加入项目根目录到 sys.path
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 IsaacLab Client
from isaaclab_lib.isaaclab_client import IsaaclabApp
from isaaclab_lib.pose import quat_to_axis_angle
from configs.test import config as isaaclab_config

app = Flask(__name__)

# ------------------------- 全局状态 -------------------------
CFG: Dict[str, Any] = {}
isaaclab_app: Optional[IsaaclabApp] = None
isaaclab_lock = threading.Lock()  # 保护 IsaacLab 操作的锁

# 请求队列和事件
reset_queue = queue.Queue()
step_queue = queue.Queue()
reset_event = threading.Event()
step_event = threading.Event()

reset_request_count = 0
step_request_count = 0

# 统计信息
total_step_requests = 0
lightweight_step_requests = 0
full_step_requests = 0

# 打印控制
step_print_counter = 0  # 用于控制每N步打印一次
step_print_interval = 5  # 每5步打印一次

# 当前状态缓存
current_tcp_pose: Optional[np.ndarray] = None
current_joint_pos: Optional[np.ndarray] = None
current_gripper_pos: Optional[float] = None
state_lock = threading.Lock()

# 图像保存目录
save_dir: Optional[Path] = None

# ------------------------- 工具函数 -------------------------

def move_to_cpu(data):
    """将数据移动到CPU"""
    if isinstance(data, torch.Tensor):
        return data.cpu()
    if isinstance(data, dict):
        return {k: move_to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_cpu(v) for v in data)
    return data


def serialize_data(data) -> str:
    """序列化数据为base64字符串"""
    buffer = io.BytesIO()
    data = move_to_cpu(data)
    pickle.dump(data, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def send_handshake(train_host: str, train_port: int, token: str) -> bool:
    """向训练端发送握手"""
    url = f"http://{train_host}:{train_port}/handshake"
    try:
        resp = requests.post(url, json={"data": token}, timeout=5)
        print(f"[仿真端] [handshake] status={resp.status_code}, body={resp.text}")
        return resp.ok
    except Exception as e:
        print(f"[仿真端] [handshake] Failed: {e}")
        return False


def save_image(image: np.ndarray, filename: str, subdir: str = ""):
    """
    保存图像到 save 目录
    
    Args:
        image: 图像数组
        filename: 文件名
        subdir: 子目录（可选）
    """
    global save_dir
    
    if save_dir is None:
        return
    
    # 创建子目录
    if subdir:
        target_dir = save_dir / subdir
    else:
        target_dir = save_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保图像是 uint8 类型
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_save = (image * 255).astype(np.uint8)
        else:
            image_save = image.astype(np.uint8)
    else:
        image_save = image.copy()
    
    filepath = target_dir / filename
    cv2.imwrite(str(filepath), image_save)


def parse_tcp_pose(tcp_pose) -> tuple:
    """解析TCP位姿，返回位置和四元数"""
    if isinstance(tcp_pose, (list, tuple)):
        tcp_pose = np.array(tcp_pose)
    
    if len(tcp_pose) >= 7:
        pos = tcp_pose[:3]
        quat = tcp_pose[3:7]
    else:
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    return pos, quat


def get_gripper_width(isaaclab_app: IsaaclabApp, arm_id: int) -> float:
    """获取夹爪宽度"""
    gripper_width = isaaclab_app.get_gripper(arm_id=arm_id)
    if gripper_width is not None:
        gripper_width_arr = np.asarray(gripper_width).flatten()
        return float(gripper_width_arr[0] if len(gripper_width_arr) > 0 else 0.0)
    return 0.0


def get_camera_image_from_isaaclab(camera_id: int, height: int, width: int, save: bool = False, request_id: int = 0, request_type: str = "step") -> np.ndarray:
    """
    从 IsaacLab 获取相机图像
    
    Args:
        camera_id: 相机ID
        height: 目标图像高度
        width: 目标图像宽度
        save: 是否保存图像
        request_id: 请求ID（用于文件名）
        request_type: 请求类型（"step" 或 "reset"）
    
    Returns:
        图像数组 (height, width, 3) uint8
    """
    global isaaclab_app
    
    if isaaclab_app is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        with isaaclab_lock:
            rgb, depth = isaaclab_app.get_rgbd(camera_id)
            
            # 调整图像大小
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # 确保是 uint8 类型
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            
            # 保存图像
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                camera_name = "camera_left" if camera_id == 0 else f"camera_right" if camera_id == 1 else f"camera_{camera_id}"
                filename = f"{request_type}_{request_id:06d}_{camera_name}_{timestamp}.png"
                save_image(rgb, filename, subdir=request_type)
            
            return rgb
            
    except Exception as e:
        print(f"[仿真端] 获取相机图像失败 (camera_id={camera_id}): {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((height, width, 3), dtype=np.uint8)


def create_real_obs_from_isaaclab(
    num_envs: int,
    image_height: int,
    image_width: int,
    save_images: bool = False,
    request_id: int = 0,
    request_type: str = "step",
) -> Dict[str, Any]:
    """
    从 IsaacLab 仿真环境创建双机械臂观测数据
    
    期望输出格式：
    {
        "policy": {
            "Camera_left": np.ndarray,  # shape: (num_envs, H, W, 3), dtype=uint8
            "Camera_right": np.ndarray,  # shape: (num_envs, H, W, 3), dtype=uint8
            "Robot_state": np.ndarray,   # shape: (num_envs, 14), dtype=float64
        }
    }
    
    14维Robot_state组成：
    - robot0_eef_pos (3维): 左手位置 [x, y, z]
    - robot0_eef_rot_axis_angle (3维): 左手姿态 [ax, ay, az]
    - robot0_gripper_width (1维): 左手夹爪宽度
    - robot1_eef_pos (3维): 右手位置 [x, y, z]
    - robot1_eef_rot_axis_angle (3维): 右手姿态 [ax, ay, az]
    - robot1_gripper_width (1维): 右手夹爪宽度
    
    Args:
        num_envs: 环境数量
        image_height: 图像高度
        image_width: 图像宽度
        save_images: 是否保存图像
        request_id: 请求ID
        request_type: 请求类型
    
    Returns:
        观测数据字典
    """
    global isaaclab_app
    
    if isaaclab_app is None:
        # 如果 IsaacLab 未初始化，返回零数据
        camera_left = torch.zeros(num_envs, image_height, image_width, 3, dtype=torch.uint8)
        camera_right = torch.zeros(num_envs, image_height, image_width, 3, dtype=torch.uint8)
        robot_state = torch.zeros(num_envs, 14, dtype=torch.float64)
        
        return {
            "policy": {
                "Camera_left": camera_left,
                "Camera_right": camera_right,
                "Robot_state": robot_state,
            }
        }
    
    with isaaclab_lock:
        try:
            # 1. 获取两个机械臂的TCP位姿
            tcp_pose_0 = isaaclab_app.get_tcp_pose(arm_id=0, frame="base")
            tcp_pose_1 = isaaclab_app.get_tcp_pose(arm_id=1, frame="base")
            
            robot0_eef_pos, robot0_eef_quat = parse_tcp_pose(tcp_pose_0)
            robot1_eef_pos, robot1_eef_quat = parse_tcp_pose(tcp_pose_1)
            
            # 2. 转换为轴角表示
            robot0_eef_rot_axis_angle = quat_to_axis_angle(robot0_eef_quat)
            robot1_eef_rot_axis_angle = quat_to_axis_angle(robot1_eef_quat)
            
            # 3. 获取夹爪宽度
            robot0_gripper_width = get_gripper_width(isaaclab_app, arm_id=0)
            robot1_gripper_width = get_gripper_width(isaaclab_app, arm_id=1)
            
        except Exception as e:
            print(f"[仿真端] 获取观测数据失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用默认值
            robot0_eef_pos = np.zeros(3)
            robot0_eef_rot_axis_angle = np.zeros(3)
            robot0_gripper_width = 0.0
            robot1_eef_pos = np.zeros(3)
            robot1_eef_rot_axis_angle = np.zeros(3)
            robot1_gripper_width = 0.0
    
    # 4. 获取相机图像
    camera_left = get_camera_image_from_isaaclab(0, image_height, image_width, save=save_images, request_id=request_id, request_type=request_type)
    camera_right = get_camera_image_from_isaaclab(1, image_height, image_width, save=save_images, request_id=request_id, request_type=request_type)
    
    # 5. 构建14维state
    states_14d = np.concatenate([
        robot0_eef_pos,
        robot0_eef_rot_axis_angle,
        np.array([robot0_gripper_width]),
        robot1_eef_pos,
        robot1_eef_rot_axis_angle,
        np.array([robot1_gripper_width]),
    ]).astype(np.float64)
    
    # 6. 扩展到num_envs维度并转换为torch tensor
    camera_left_batch = torch.from_numpy(camera_left).unsqueeze(0).repeat(num_envs, 1, 1, 1).to(torch.uint8)
    camera_right_batch = torch.from_numpy(camera_right).unsqueeze(0).repeat(num_envs, 1, 1, 1).to(torch.uint8)
    robot_state_batch = torch.from_numpy(states_14d).unsqueeze(0).repeat(num_envs, 1).to(torch.float64)
    
    obs = {
        "policy": {
            "Camera_left": camera_left_batch,
            "Camera_right": camera_right_batch,
            "Robot_state": robot_state_batch,
        }
    }
    
    return obs


def create_step_result_from_isaaclab(
    num_envs: int,
    image_height: int,
    image_width: int,
    return_obs: bool = True,
    save_images: bool = False,
    request_id: int = 0,
) -> Tuple[Optional[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    创建step结果（从 IsaacLab 仿真环境获取）
    
    Args:
        num_envs: 环境数量
        image_height: 图像高度
        image_width: 图像宽度
        return_obs: 是否返回观测数据
        save_images: 是否保存图像
        request_id: 请求ID
    
    Returns:
        (obs, reward, terminated, truncated, info)
    """
    obs = None
    if return_obs:
        obs = create_real_obs_from_isaaclab(
            num_envs, image_height, image_width,
            save_images=save_images, request_id=request_id, request_type="step"
        )
    
    # TODO: 从环境获取真实的奖励和终止信号
    # 目前返回占位数据，需要根据实际环境接口实现
    reward = torch.zeros(num_envs, dtype=torch.float32)
    terminated = torch.zeros(num_envs, dtype=torch.bool)
    truncated = torch.zeros(num_envs, dtype=torch.bool)
    info = [{} for _ in range(num_envs)]
    
    return obs, reward, terminated, truncated, info


def clamp_joint_limits(action: Any) -> Any:
    """
    检查并裁剪关节限位
    
    Args:
        action: 动作数据，可能是：
            - 关节位置控制: [j1, j2, ..., j7, gripper]
            - TCP位姿控制: [x, y, z, qw, qx, qy, qz, gripper]
            - 或其他格式
    
    Returns:
        裁剪后的action（如果超出限位）
    """
    # 关节限位定义（7个关节的最小值和最大值）
    joint_min_limits = np.array([-2.88, -2.35, -3.05, -1.95, -3.05, -1.48, -3.05])
    joint_max_limits = np.array([2.88, 2.35, 3.05, 2.74, 3.05, 4.63, 3.05])
    
    # 将action转换为numpy数组（如果还不是）
    if isinstance(action, (list, tuple)):
        action = np.array(action)
    elif isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    # 如果不是numpy数组，直接返回
    if not isinstance(action, np.ndarray):
        return action
    
    # 处理多维数组
    original_shape = action.shape
    if action.ndim > 1:
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]  # 取第一行
        else:
            action = action.flatten()
    
    # 检查是否是关节位置控制（应该有7个关节 + 1个gripper = 8个元素）
    # 或者只有7个关节（没有gripper）
    if len(action) == 8:
        # 8个元素：7个关节 + 1个gripper
        joints = action[:7]
        gripper = action[7]
        
        # 检查关节是否超出限位
        below_min = joints < joint_min_limits
        above_max = joints > joint_max_limits
        
        if np.any(below_min) or np.any(above_max):
            # 记录超出限位的关节
            violated_joints = []
            for i in range(7):
                if below_min[i]:
                    violated_joints.append(f"关节{i+1}({joints[i]:.3f} < {joint_min_limits[i]:.3f})")
                elif above_max[i]:
                    violated_joints.append(f"关节{i+1}({joints[i]:.3f} > {joint_max_limits[i]:.3f})")
            
            print(f"[关节限位] 超出限位，已自动裁剪: {', '.join(violated_joints)}")
            
            # 裁剪到限位范围内
            joints_clamped = np.clip(joints, joint_min_limits, joint_max_limits)
            
            # 重构action
            action_clamped = np.concatenate([joints_clamped, [gripper]])
            
            # 恢复原始形状
            if len(original_shape) > 1:
                action_clamped = action_clamped.reshape(original_shape)
            
            return action_clamped
        else:
            # 恢复原始形状（如果需要）
            if len(original_shape) > 1 and action.ndim == 1:
                return action.reshape(original_shape)
            return action
    
    elif len(action) == 7:
        # 7个元素：只有关节，没有gripper
        joints = action[:7]
        
        # 检查关节是否超出限位
        below_min = joints < joint_min_limits
        above_max = joints > joint_max_limits
        
        if np.any(below_min) or np.any(above_max):
            # 记录超出限位的关节
            violated_joints = []
            for i in range(7):
                if below_min[i]:
                    violated_joints.append(f"关节{i+1}({joints[i]:.3f} < {joint_min_limits[i]:.3f})")
                elif above_max[i]:
                    violated_joints.append(f"关节{i+1}({joints[i]:.3f} > {joint_max_limits[i]:.3f})")
            
            print(f"[关节限位] 超出限位，已自动裁剪: {', '.join(violated_joints)}")
            
            # 裁剪到限位范围内
            joints_clamped = np.clip(joints, joint_min_limits, joint_max_limits)
            
            # 恢复原始形状
            if len(original_shape) > 1:
                joints_clamped = joints_clamped.reshape(original_shape)
            
            return joints_clamped
        else:
            # 恢复原始形状（如果需要）
            if len(original_shape) > 1 and action.ndim == 1:
                return action.reshape(original_shape)
            return action
    
    # 对于其他格式（如TCP位姿控制或其他），不进行限位检查
    # 恢复原始形状（如果需要）
    if len(original_shape) > 1 and action.ndim == 1:
        return action.reshape(original_shape)
    return action


def apply_action_to_isaaclab(action: Any, arm_id: int = 0):
    """
    将action应用到 IsaacLab 仿真环境（仅处理关节位置控制）
    
    Args:
        action: 关节位置控制动作数据，格式: [j1, j2, ..., jn, gripper]
            - 7个值: [j1, j2, j3, j4, j5, j6, j7] (7个关节，无gripper)
            - 8个值: [j1, j2, j3, j4, j5, j6, j7, gripper] (7个关节 + 1个gripper)
            - 其他长度: [j1, j2, ..., jn, gripper] (n个关节 + 1个gripper)
        arm_id: 机械臂ID
    
    注意: TCP位姿控制会通过其他接口单独发送，不经过此函数
    """
    global isaaclab_app
    
    if isaaclab_app is None:
        print("[仿真端] 警告: IsaacLab 未初始化")
        return
    
    try:
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        if isinstance(action, np.ndarray) and action.ndim > 1:
            if action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            else:
                action = action.flatten()
        
        with isaaclab_lock:
            if len(action) >= 2:
                if len(action) == 7:
                    joint_pos = action
                    gripper_pos = 0.0
                elif len(action) == 8:
                    joint_pos = action[:-1]
                    gripper_pos_raw = action[-1]
                    gripper_pos_arr = np.asarray(gripper_pos_raw).flatten()
                    gripper_pos = float(gripper_pos_arr[0] if len(gripper_pos_arr) > 0 else 0.0)
                else:
                    joint_pos = action[:-1]
                    gripper_pos_raw = action[-1]
                    gripper_pos_arr = np.asarray(gripper_pos_raw).flatten()
                    gripper_pos = float(gripper_pos_arr[0] if len(gripper_pos_arr) > 0 else 0.0)
                
                isaaclab_app.move_arm_joint(joints=joint_pos.tolist(), arm_id=arm_id, blocking=False)
                
                if gripper_pos > 0.5:
                    isaaclab_app.open_gripper(arm_id=arm_id, width=gripper_pos)
                elif gripper_pos > 0.0:
                    isaaclab_app.close_gripper(arm_id=arm_id)
            elif len(action) == 1:
                joint_pos = action
                isaaclab_app.move_arm_joint(joints=joint_pos.tolist(), arm_id=arm_id, blocking=False)
            else:
                print(f"[仿真端] 警告: action长度无效，长度={len(action)}")
                return
    
    except Exception as e:
        print(f"[仿真端] 应用action失败: {e}")
        import traceback
        traceback.print_exc()


def send_reset_result_to_train(seed, env_ids):
    """发送reset结果到训练端"""
    train_host = CFG["train_host"]
    train_port = CFG["train_port"]
    
    if env_ids is None:
        num_envs = CFG["num_envs"]
    else:
        num_envs = len(env_ids) if isinstance(env_ids, list) else env_ids.shape[0] if hasattr(env_ids, 'shape') else CFG["num_envs"]
    
    image_h = CFG["image_height"]
    image_w = CFG["image_width"]
    save_images = CFG.get("save_images", False)
    
    # 从 IsaacLab 获取真实观测
    obs = create_real_obs_from_isaaclab(
        num_envs, image_h, image_w,
        save_images=save_images, request_id=reset_request_count, request_type="reset"
    )
    info = {}
    result_data = serialize_data((obs, info))
    
    url = f"http://{train_host}:{train_port}/reset"
    try:
        t0 = time.time()
        resp = requests.post(
            url,
            json={"seed": seed, "env_ids": env_ids, "data": result_data},
            timeout=30,
        )
        print(f"[仿真端] → 发送 reset 结果: status={resp.status_code}")
    except Exception as e:
        print(f"[仿真端] 发送 reset 结果失败: {e}")


def send_step_result_to_train(action, return_obs: bool = True):
    """
    发送step结果到训练端（使用 IsaacLab 真实数据）
    
    Args:
        action: 动作数据
        return_obs: 是否返回观测数据
    """
    global lightweight_step_requests, full_step_requests, step_print_counter
    
    train_host = CFG["train_host"]
    train_port = CFG["train_port"]
    num_envs = CFG["num_envs"]
    image_h = CFG["image_height"]
    image_w = CFG["image_width"]
    arm_id = CFG.get("arm_id", 0)
    save_images = CFG.get("save_images", False)
    
    if return_obs:
        full_step_requests += 1
    else:
        lightweight_step_requests += 1
    
    # 检查和裁剪关节限位
    action = clamp_joint_limits(action)
    
    # 应用action到仿真环境
    apply_action_to_isaaclab(action, arm_id)
    
    # 等待一小段时间让仿真环境更新（根据实际情况调整）
    time.sleep(0.01)  # 10ms，仿真环境通常很快
    
    # 获取step结果
    step_result = create_step_result_from_isaaclab(
        num_envs, image_h, image_w, return_obs=return_obs,
        save_images=save_images, request_id=step_request_count
    )
    result_data = serialize_data(step_result)
    
    # 序列化 action
    action_data = serialize_data(action)
    
    url = f"http://{train_host}:{train_port}/step"
    try:
        t0 = time.time()
        resp = requests.post(
            url, 
            json={
                "action": action_data, 
                "data": result_data,
                "return_obs": return_obs
            }, 
            timeout=30
        )
        cost = (time.time() - t0) * 1000
        
        # 更新计数器
        step_print_counter += 1
        
        # 每5步打印一次，打印到同一行（使用stderr避免被Flask日志覆盖）
        if step_print_counter >= step_print_interval:
            mode_str = "完整观测" if return_obs else "轻量模式"
            print(
                f"[仿真端] → step #{step_request_count}: status={resp.status_code}, "
                f"耗时={cost:.1f}ms, 模式={mode_str}",
                end='\r',
                flush=True,
                file=sys.stderr
            )
            step_print_counter = 0
    except Exception as e:
        print(f"[仿真端] 发送 step 结果失败: {e}")
        step_print_counter += 1


# ------------------------- HTTP 路由 -------------------------

@app.route("/reset_from_train", methods=["POST"])
def reset_from_train():
    """接收训练端的reset请求"""
    global reset_request_count, isaaclab_app
    
    reset_request_count += 1
    
    data = flask_request.json or {}
    seed = data.get("seed")
    env_ids = data.get("env_ids")
    
    # 执行reset操作（重置仿真环境）
    if isaaclab_app is not None:
        try:
            with isaaclab_lock:
                isaaclab_app.reset()
            print(f"[仿真端] 仿真环境已重置")
        except Exception as e:
            print(f"[仿真端] Reset操作失败: {e}")
    
    reset_queue.put({"seed": seed, "env_ids": env_ids, "raw": data})
    reset_event.set()
    
    if CFG["auto_reply_reset"]:
        threading.Thread(target=send_reset_result_to_train, args=(seed, env_ids), daemon=True).start()
    
    return jsonify({"status": "success", "message": f"收到 reset #{reset_request_count}"}), 200


@app.route("/step_from_train", methods=["POST"])
def step_from_train():
    """接收训练端的step请求（支持return_obs标志）"""
    global step_request_count, total_step_requests
    
    step_request_count += 1
    total_step_requests += 1
    
    data = flask_request.json or {}
    action = data.get("action")
    return_obs = data.get("return_obs", True)
    
    step_queue.put({"action": action, "return_obs": return_obs, "raw": data})
    step_event.set()
    
    if CFG["auto_reply_step"]:
        threading.Thread(
            target=send_step_result_to_train, 
            args=(action, return_obs), 
            daemon=True
        ).start()
    
    return jsonify({
        "status": "success", 
        "message": f"收到 step #{step_request_count}",
        "return_obs": return_obs
    }), 200


@app.route("/ping", methods=["GET"])
def ping():
    """健康检查端点"""
    global lightweight_step_requests, full_step_requests, total_step_requests, isaaclab_app
    
    lightweight_ratio = (lightweight_step_requests / total_step_requests * 100) if total_step_requests > 0 else 0
    full_ratio = (full_step_requests / total_step_requests * 100) if total_step_requests > 0 else 0
    
    isaaclab_status = "connected" if isaaclab_app is not None else "disconnected"
    
    return jsonify({
        "status": "ok",
        "isaaclab_status": isaaclab_status,
        "reset_request_count": reset_request_count,
        "step_request_count": step_request_count,
        "total_step_requests": total_step_requests,
        "lightweight_step_requests": lightweight_step_requests,
        "full_step_requests": full_step_requests,
        "lightweight_ratio": f"{lightweight_ratio:.1f}%",
        "full_ratio": f"{full_ratio:.1f}%",
    })


@app.route("/stats", methods=["GET"])
def stats():
    """获取统计信息"""
    return ping()


# ------------------------- 运行入口 -------------------------

def run_server(host: str, port: int, debug: bool):
    """运行Flask服务器"""
    # 禁用Flask的访问日志，避免覆盖我们的打印输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("=" * 60)
    print("IsaacLab 仿真端服务器启动")
    print(f"监听: {host}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


def main():
    global isaaclab_app, CFG, save_dir
    
    parser = argparse.ArgumentParser(description="IsaacLab 仿真端服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器监听地址")
    parser.add_argument("--port", type=int, default=8445, help="服务器监听端口")
    parser.add_argument("--train-host", default="10.8.0.1", help="训练端地址")
    parser.add_argument("--train-port", type=int, default=8446, help="训练端端口")
    parser.add_argument("--handshake-token", default="123", help="握手token")
    parser.add_argument("--send-handshake", action="store_true", default=True, help="启动时发送握手")
    parser.add_argument("--num-envs", type=int, default=1, 
                        help="环境总数（应与训练配置一致）")
    parser.add_argument("--image-height", type=int, default=224, help="图像高度")
    parser.add_argument("--image-width", type=int, default=224, help="图像宽度")
    parser.add_argument("--auto-reply-reset", action="store_true", default=True, 
                        help="收到 reset 后自动回传结果")
    parser.add_argument("--auto-reply-step", action="store_true", default=True, 
                        help="收到 step 后自动回传结果")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    # IsaacLab 相关参数
    parser.add_argument("--isaaclab-url", type=str, default="http://127.0.0.1:8000",
                        help="IsaacLab 服务器地址")
    parser.add_argument("--arm-id", type=int, default=0, help="机械臂ID（用于action应用，观测值使用双臂）")
    
    # 图像保存相关参数
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="保存图像到 save 目录")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="图像保存目录（默认: test_rlinf/test_env/save）")
    
    args = parser.parse_args()
    
    # 设置保存目录
    if args.save_images:
        if args.save_dir:
            save_dir = Path(args.save_dir)
        else:
            save_dir = current_file.parent / "save"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[仿真端] 图像保存目录: {save_dir}")
    
    # 初始化 IsaacLab App
    print(f"[仿真端] 正在连接 IsaacLab 服务器: {args.isaaclab_url}")
    try:
        isaaclab_app = IsaaclabApp(base_url=args.isaaclab_url, config=isaaclab_config)
        if isaaclab_app.is_operational():
            print(f"[仿真端] IsaacLab 连接成功")
        else:
            print(f"[仿真端] 警告: IsaacLab 连接状态未知")
    
    except Exception as e:
        print(f"[仿真端] 错误: 无法连接 IsaacLab: {e}")
        import traceback
        traceback.print_exc()
        print(f"[仿真端] 将继续运行，但将使用占位数据")
        isaaclab_app = None
    
    CFG.update({
        "train_host": args.train_host,
        "train_port": args.train_port,
        "num_envs": args.num_envs,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "auto_reply_reset": args.auto_reply_reset,
        "auto_reply_step": args.auto_reply_step,
        "arm_id": args.arm_id,
        "save_images": args.save_images,
    })
    
    server_thread = threading.Thread(
        target=run_server, 
        args=(args.host, args.port, args.debug), 
        daemon=True
    )
    server_thread.start()
    
    print("[仿真端] 等待服务器启动...")
    time.sleep(2)
    
    if args.send_handshake:
        print("[仿真端] 向训练端发送握手...")
        ok = send_handshake(args.train_host, args.train_port, args.handshake_token)
        print("[仿真端] 握手成功" if ok else "[仿真端] 握手失败")
    
    print("=" * 60)
    print("服务器已启动，开始等待请求...")
    if args.save_images:
        print(f"图像保存目录: {save_dir}")
    print("Ctrl+C 退出")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
            # 定期打印统计信息
            if total_step_requests > 0 and total_step_requests % 50 == 0:
                lightweight_ratio = (lightweight_step_requests / total_step_requests * 100)
                print(
                    f"[仿真端] 统计: 总请求={total_step_requests}, "
                    f"轻量模式={lightweight_step_requests} ({lightweight_ratio:.1f}%), "
                    f"完整模式={full_step_requests}"
                )
    except KeyboardInterrupt:
        print("\n[仿真端] 停止")
        print(f"[仿真端] 最终统计: 总请求={total_step_requests}, "
              f"轻量模式={lightweight_step_requests}, 完整模式={full_step_requests}")
        if args.save_images:
            print(f"[仿真端] 图像已保存到: {save_dir}")


if __name__ == "__main__":
    main()

