#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练端收发测试脚本

模拟训练端的完整收发流程：
1. 启动 RemoteIsaacLabServer（接收仿真端发送的数据）
2. 使用 RemoteSimSender（向仿真端发送 reset/step 指令）
3. 测试完整的 reset 和 step 流程

用法：
python train_test.py \
  --server-host 0.0.0.0 --server-port 8446 \
  --sim-host 10.8.0.4 --sim-port 8446 \
  --handshake-token 123 \
  --num-envs 8 \
  --test-steps 1000 \
  --obs-interval 5

参数说明：
  --test-steps: 测试步数，0 表示无限循环（默认 1000）
  --obs-interval: 每隔几步返回一次完整观测（默认 5）
"""

import argparse
import threading
import time
import torch

from rlinf.envs.isaaclab_remote_double_arm.remote_server import RemoteIsaacLabServer
from rlinf.envs.isaaclab_remote_double_arm.remote_sender import RemoteSimSender


def test_reset(server: RemoteIsaacLabServer, sender: RemoteSimSender, seed: int = None, env_ids: list = None):
    """测试 reset 流程"""
    print("\n" + "=" * 60)
    print("[测试] 开始测试 reset 流程")
    print("=" * 60)
    
    # 训练端发送 reset 指令到仿真端
    print(f"[训练端] 发送 reset 请求到仿真端...")
    payload = {}
    if seed is not None:
        payload["seed"] = seed
    if env_ids is not None:
        payload["env_ids"] = env_ids
    
    send_start = time.perf_counter()
    sender.send_reset(payload)
    send_elapsed = time.perf_counter() - send_start
    print(f"[训练端] reset 请求已发送，耗时 {send_elapsed:.3f}s")
    
    # 等待仿真端发送 reset 结果数据
    print(f"[训练端] 等待仿真端发送 reset 结果数据...")
    wait_start = time.perf_counter()
    result = server.wait_for_reset_result(timeout=30)
    wait_elapsed = time.perf_counter() - wait_start
    
    if result is None:
        print(f"[训练端] ✗ 未收到 reset 结果数据（超时）")
        return False
    
    if isinstance(result, Exception):
        print(f"[训练端] ✗ reset 失败: {result}")
        return False
    
    obs, info = result
    total_elapsed = time.perf_counter() - send_start
    
    print(f"[训练端] ✓ 收到 reset 结果数据")
    print(f"  - 发送耗时: {send_elapsed:.3f}s")
    print(f"  - 等待耗时: {wait_elapsed:.3f}s")
    print(f"  - 总耗时: {total_elapsed:.3f}s")
    print(f"  - obs 类型: {type(obs)}")
    if isinstance(obs, dict):
        print(f"  - obs 键: {list(obs.keys())}")
    print(f"  - info 类型: {type(info)}")
    
    # 确认收到
    server.put_reset_result_ack()
    
    return True


def test_step(server: RemoteIsaacLabServer, sender: RemoteSimSender, actions, return_obs: bool = True):
    """测试 step 流程"""
    print("\n" + "=" * 60)
    print(f"[测试] 开始测试 step 流程 (return_obs={return_obs})")
    print("=" * 60)
    
    # 打印 action 信息
    print(f"[训练端] action 类型: {type(actions)}")
    if isinstance(actions, torch.Tensor):
        print(f"[训练端] action shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"[训练端] action min: {actions.min()}, max: {actions.max()}")
    elif hasattr(actions, 'shape'):
        print(f"[训练端] action shape: {actions.shape}")
    else:
        print(f"[训练端] action: {actions}")
    
    # 训练端发送 step 指令到仿真端
    print(f"[训练端] 发送 step 请求到仿真端...")
    send_start = time.perf_counter()
    sender.send_step(actions, return_obs=return_obs)
    send_elapsed = time.perf_counter() - send_start
    print(f"[训练端] step 请求已发送，耗时 {send_elapsed:.3f}s")
    
    # 等待仿真端发送 step 结果数据
    print(f"[训练端] 等待仿真端发送 step 结果数据...")
    wait_start = time.perf_counter()
    result = server.wait_for_step_result(timeout=30)
    wait_elapsed = time.perf_counter() - wait_start
    
    if result is None:
        print(f"[训练端] ✗ 未收到 step 结果数据（超时）")
        return False
    
    if isinstance(result, Exception):
        print(f"[训练端] ✗ step 失败: {result}")
        return False
    
    obs, step_reward, terminations, truncations, infos = result
    total_elapsed = time.perf_counter() - send_start
    
    print(f"[训练端] ✓ 收到 step 结果数据")
    print(f"  - 发送耗时: {send_elapsed:.3f}s")
    print(f"  - 等待耗时: {wait_elapsed:.3f}s")
    print(f"  - 总耗时: {total_elapsed:.3f}s")
    
    if return_obs:
        print(f"  - obs 类型: {type(obs)}")
        if isinstance(obs, dict):
            print(f"  - obs 键: {list(obs.keys())}")
    else:
        print(f"  - obs: {obs} (轻量模式，无观测数据)")
    
    print(f"  - reward 类型: {type(step_reward)}")
    if isinstance(step_reward, torch.Tensor):
        print(f"  - reward shape: {step_reward.shape}")
        print(f"  - reward 范围: [{step_reward.min():.4f}, {step_reward.max():.4f}]")
    
    print(f"  - terminations 类型: {type(terminations)}")
    if isinstance(terminations, torch.Tensor):
        print(f"  - terminations shape: {terminations.shape}")
        print(f"  - terminations 数量: {terminations.sum().item()}")
    
    print(f"  - truncations 类型: {type(truncations)}")
    if isinstance(truncations, torch.Tensor):
        print(f"  - truncations shape: {truncations.shape}")
        print(f"  - truncations 数量: {truncations.sum().item()}")
    
    # 确认收到
    server.put_step_result_ack()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="训练端收发测试脚本")
    parser.add_argument("--server-host", default="0.0.0.0", help="训练服务器监听地址（默认 0.0.0.0）")
    parser.add_argument("--server-port", type=int, default=8446, help="训练服务器监听端口（默认 8446）")
    parser.add_argument("--sim-host", default="10.8.0.4", help="仿真端地址（默认 10.8.0.4）")
    parser.add_argument("--sim-port", type=int, default=8445, help="仿真端端口（默认 8446）")
    parser.add_argument("--handshake-token", default="123", help="握手数据（需与仿真端一致）")
    parser.add_argument("--num-envs", type=int, default=1, help="环境数量（默认 8）")
    parser.add_argument("--action-dim", type=int, default=7, help="动作维度（默认 7）")
    parser.add_argument("--test-steps", type=int, default=1000, help="测试步数（默认 1000，0 表示无限循环）")
    parser.add_argument("--obs-interval", type=int, default=5, help="每隔几步返回一次完整观测（默认 5）")
    parser.add_argument("--device", default="cuda", help="设备（默认 cuda）")
    parser.add_argument("--skip-reset", action="store_true", help="跳过 reset 测试")
    parser.add_argument("--skip-step", action="store_true", help="跳过 step 测试")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("训练端收发测试")
    print("=" * 60)
    print(f"训练服务器: {args.server_host}:{args.server_port}")
    print(f"仿真端: {args.sim_host}:{args.sim_port}")
    print(f"握手数据: {args.handshake_token}")
    print(f"环境数量: {args.num_envs}")
    print(f"动作维度: {args.action_dim}")
    print(f"测试步数: {args.test_steps if args.test_steps > 0 else '无限循环'}")
    print(f"观测间隔: 每 {args.obs_interval} 步返回一次完整观测")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 创建并启动服务器
    print("\n[初始化] 创建训练服务器...")
    server = RemoteIsaacLabServer(
        host=args.server_host,
        port=args.server_port,
        local_device=args.device,
    )
    
    # 在后台线程中启动服务器
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    print(f"[初始化] 等待服务器启动...")
    time.sleep(2)
    
    # 创建发送器
    print(f"[初始化] 创建仿真端发送器...")
    sender = RemoteSimSender(
        sim_host=args.sim_host,
        sim_port=args.sim_port,
    )
    
    # 等待仿真端握手
    print(f"\n[初始化] 等待仿真端握手数据: {args.handshake_token}")
    success = server.wait_for_handshake(expected_data=args.handshake_token, timeout=300)
    if not success:
        print("[初始化] ✗ 等待仿真端握手超时或失败")
        return
    
    print("[初始化] ✓ 仿真端握手成功，开始测试\n")
    
    # 测试 reset
    if not args.skip_reset:
        reset_success = test_reset(server, sender, seed=42, env_ids=None)
        if not reset_success:
            print("\n[测试] ✗ reset 测试失败")
            return
        print("\n[测试] ✓ reset 测试成功")
        time.sleep(1)
    
    # 测试 step
    if not args.skip_step:
        print("\n" + "=" * 60)
        if args.test_steps > 0:
            print(f"[测试] 开始测试 {args.test_steps} 步 step 流程")
        else:
            print(f"[测试] 开始无限循环测试（按 Ctrl+C 退出）")
        print(f"[测试] 每 {args.obs_interval} 步返回一次完整观测")
        print("=" * 60)
        
        success_count = 0
        total_time = 0
        step_idx = 0
        
        # 如果 test_steps > 0，执行指定步数；否则无限循环
        while True:
            if args.test_steps > 0:
                if step_idx >= args.test_steps:
                    break
                step_label = f"Step {step_idx + 1}/{args.test_steps}"
            else:
                step_label = f"Step {step_idx + 1}"
            
            print(f"\n--- {step_label} ---")
            
            # 生成随机动作
            actions = torch.randn(args.num_envs, args.action_dim).to(args.device)
            # fixed_action = [0, 0, 0, 1.5, 1.5, 1, 0]
            actions = torch.tensor([actions] * args.num_envs, dtype=torch.float32).to(args.device)
            print(f"[训练端] actions: {actions}")
            
            # 每隔 obs_interval 步返回一次完整观测
            # 例如 obs_interval=5: 第5, 10, 15...步返回完整观测
            return_obs = ((step_idx + 1) % args.obs_interval == 0)
            
            step_start = time.perf_counter()
            step_success = test_step(server, sender, actions, return_obs=return_obs)
            step_elapsed = time.perf_counter() - step_start
            
            if step_success:
                success_count += 1
                total_time += step_elapsed
                mode_str = "完整模式" if return_obs else "轻量模式"
                print(f"[测试] {step_label} 完成（{mode_str}），耗时 {step_elapsed:.3f}s")
            else:
                print(f"[测试] {step_label} 失败")
                break
            
            step_idx += 1
            
            # 短暂延迟，避免过快发送
            time.sleep(0.1)
        
        print("\n" + "=" * 60)
        print("[测试] Step 测试统计")
        print("=" * 60)
        print(f"成功步数: {success_count}/{step_idx}")
        if success_count > 0:
            avg_time = total_time / success_count
            print(f"平均耗时: {avg_time:.3f}s/步")
            print(f"总耗时: {total_time:.3f}s")
        print("=" * 60)
    
    print("\n[测试] 所有测试完成")
    print("\n提示：按 Ctrl+C 退出")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[测试] 用户中断，退出")
    except Exception as e:
        print(f"\n[测试] 发生错误: {e}")
        import traceback
        traceback.print_exc()

