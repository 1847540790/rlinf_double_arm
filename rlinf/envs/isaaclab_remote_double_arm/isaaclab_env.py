# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
远程 IsaacLab 环境
通过网络连接远程仿真服务器，适用于分布式训练场景
"""

import copy
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import imageio
import torch

from rlinf.envs.isaaclab_remote_double_arm.remote_server import RemoteIsaacLabServer

# 强制无缓冲输出
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


class RemoteIsaaclabBaseEnv(gym.Env):
    """
    远程 IsaacLab 环境基类
    
    与本地 IsaaclabBaseEnv 接口一致，但通过网络从远程服务器获取仿真数据。
    适用于以下场景：
    1. 训练服务器没有 GPU 或 IsaacLab 环境
    2. 需要集中管理仿真资源
    3. 多个训练任务共享仿真集群
    
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
    ):
        self.cfg = cfg
        self.isaaclab_env_id = self.cfg.init_params.id
        # Use the num_envs parameter passed from env_manager, not from config
        # This ensures consistency with the actual number of environments expected by the system
        self.num_envs = num_envs
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0

        self.server = None  # HTTP 服务器实例
        # 初始化远程服务器（不运行本地仿真）
        self._init_remote_client()
        # 设备信息从配置获取，因为不运行本地仿真
        self.device = getattr(self.cfg, 'device', 'cuda')

        self.task_description = cfg.init_params.task_description
        self._is_start = True  # if this is first time for simulator
        self.auto_reset = cfg.auto_reset
        self.prev_step_reward = torch.zeros(self.num_envs).to(self.device)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32).to(
            self.device
        )
        self.ignore_terminations = cfg.ignore_terminations

        self.images = []
        self._remote_data_cache = {}  # 缓存从仿真端接收的数据

    def _init_remote_client(self):
        """初始化远程服务器（训练服务器作为服务端，等待仿真端连接，不运行本地仿真）"""
        # 获取服务器配置
        server_host = getattr(self.cfg, 'server_host', '0.0.0.0')
        server_port = getattr(self.cfg, 'erver_port', 8446)
        local_device = getattr(self.cfg, 'device', 'cuda')
        handshake_token = getattr(self.cfg, 'remote_handshake_token', '123')
        
        print(f"[RemoteIsaaclabEnv] 启动训练服务器（无本地仿真），监听地址: {server_host}:{server_port}")
        print(f"[RemoteIsaaclabEnv] 等待仿真端连接并发送数据...")
        
        # 创建并启动 HTTP 服务器（不创建本地环境）
        self.server = RemoteIsaacLabServer(
            host=server_host,
            port=server_port,
            local_device=local_device,
        )
        # 用于主动向仿真端发送 reset/step 指令
        from rlinf.envs.isaaclab_remote_double_arm.remote_sender import RemoteSimSender

        sim_host = getattr(self.cfg, "remote_sim_host", "10.8.0.4")
        sim_port = getattr(self.cfg, "remote_sim_port", 8445)
        # 默认指向 receiver 配置的 host，如果未配置则复用 server_host
        sim_host = sim_host or server_host
        self.sim_sender = RemoteSimSender(sim_host=sim_host, sim_port=sim_port)
        
        # 在后台线程中启动服务器
        import threading
        server_thread = threading.Thread(target=self.server.run, daemon=True)
        server_thread.start()
        
        print(f"[RemoteIsaaclabEnv] HTTP 服务器已启动，等待仿真端连接...")

        # 阻塞等待仿真端握手（指定数据，例如 123）
        print(f"[RemoteIsaaclabEnv] 等待仿真端握手数据: {handshake_token}")
        success = self.server.wait_for_handshake(expected_data=handshake_token, timeout=None)
        if not success:
            raise RuntimeError("等待仿真端握手超时或失败")
        print(f"[RemoteIsaaclabEnv] 仿真端握手成功，继续后续流程")
        
        # 注意：不创建本地环境，只接收远程数据
        # self.env 将是一个占位符，实际数据来自仿真端
        self.env = None

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        # batch level
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[torch.Tensor] = None,
    ):
        """重置环境：训练端主动通知仿真端，并等待结果数据返回"""
        print(f"[RemoteIsaaclabEnv] [reset] 训练端发送 reset 请求到仿真端...")

        if self.server is None:
            raise RuntimeError("服务器未初始化")

        # 训练端主动发送 reset 指令到仿真端（不等待回执）
        payload = {}
        if seed is not None:
            payload["seed"] = int(seed)
        if env_ids is not None:
            payload["env_ids"] = (
                env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
            )
        self.sim_sender.send_reset(payload)

        # 等待仿真端发送 reset 结果数据
        print(f"[RemoteIsaaclabEnv] [reset] 等待仿真端发送 reset 结果数据...")
        result = self.server.wait_for_reset_result()

        if result is None:
            raise RuntimeError("未收到仿真端的 reset 结果数据")

        if isinstance(result, Exception):
            raise result

        # 从仿真端接收的数据
        obs, info = result

        # 将结果返回给仿真端（确认收到）
        self.server.put_reset_result_ack()
       
        infos = {}
        obs = self._wrap_obs(obs)
        self._reset_metrics(env_ids)
        print(f"[RemoteIsaaclabEnv] [reset] 收到仿真端数据，reset 完成")
        return obs, infos

    def step(self, actions=None, auto_reset=True, return_obs=True):
        """执行一步：训练端发送动作给仿真端，随后等待结果数据
        
        Args:
            actions: 动作张量
            auto_reset: 是否自动重置
            return_obs: 是否返回观测（False时跳过观测处理，节省资源）
        """
        # There will be an empty step when running env worker.
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False

            terminations = torch.zeros(self.num_envs, dtype=torch.bool).to(self.device)
            truncations = torch.zeros(self.num_envs, dtype=torch.bool).to(self.device)

            return obs, None, terminations, truncations, infos

        if self.server is None:
            raise RuntimeError("服务器未初始化")

        # 打印 action 信息用于调试（仅在需要观测时打印，减少日志）
        # if return_obs:
        #     print(f"[RemoteIsaaclabEnv] [step] action type: {type(actions)}")
        #     if isinstance(actions, torch.Tensor):
        #         print(f"[RemoteIsaaclabEnv] [step] action shape: {actions.shape}, dtype: {actions.dtype}")
        #         print(f"[RemoteIsaaclabEnv] [step] action min: {actions.min()}, max: {actions.max()}")
        #     elif hasattr(actions, 'shape'):
        #         print(f"[RemoteIsaaclabEnv] [step] action shape: {actions.shape}")
        #     else:
        #         print(f"[RemoteIsaaclabEnv] [step] action: {actions}")

        # 训练端主动发送动作到仿真端，并记录时间
        send_ts = time.perf_counter()
        self.sim_sender.send_step(actions, return_obs=return_obs)
        
        # 等待仿真端发送 step 结果数据
        wait_ts = time.perf_counter()
        
        obs, step_reward, terminations, truncations, infos = self.server.wait_for_step_result()
        
        total_elapsed = time.perf_counter() - send_ts
        wait_elapsed = time.perf_counter() - wait_ts
        
        # 将结果返回给仿真端（确认收到）
        self.server.put_step_result_ack()
        
        if return_obs:
            print(
                f"[RemoteIsaaclabEnv] [step] 收到仿真端数据，step 完成，"
                f"总耗时 {total_elapsed:.3f}s，等待结果 {wait_elapsed:.3f}s"
            )

        # 优化：如果不需要观测，跳过观测处理（虽然数据已传输，但至少减少处理开销）
        if return_obs:
            if self.video_cfg.save_video:
                self.images.append(self.add_image(obs))
            obs = self._wrap_obs(obs)
        else:
            # 不处理观测，返回 None 或占位符
            obs = None

        self._elapsed_steps += 1

        # 确保 truncations 是 torch tensor 并在正确的设备上
        if not isinstance(truncations, torch.Tensor):
            truncations = torch.tensor(truncations, dtype=torch.bool, device=self.device)
        else:
            truncations = truncations.to(self.device)
        
        # 确保 terminations 也是 torch tensor 并在正确的设备上
        if not isinstance(terminations, torch.Tensor):
            terminations = torch.tensor(terminations, dtype=torch.bool, device=self.device)
        else:
            terminations = terminations.to(self.device)

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        dones = terminations | truncations

        # 只在需要观测时记录指标（避免不必要的计算）
        if return_obs:
            infos = self._record_metrics(
                step_reward, terminations, {}
            )  # return infos is useless
            if self.ignore_terminations:
                infos["episode"]["success_at_end"] = terminations
                terminations[:] = False
        else:
            # 中间步骤：只保留基本的 infos 结构
            infos = {}

        _auto_reset = auto_reset and self.auto_reset  # always False
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        extracted_obs = None
        infos = None
        
        # 优化：中间步骤不处理观测，只获取奖励和终止信号
        # 只有最后一步需要完整观测用于下一轮推理
        for i in range(chunk_size): 
            actions = chunk_actions[:, i]
            is_last_step = (i == chunk_size - 1)
            
            # 中间步骤：只获取奖励和终止信号，跳过观测处理
            if not is_last_step:
                # 发送动作但不等待完整观测处理
                _, step_reward, terminations, truncations, _ = self.step(
                    actions, auto_reset=False, return_obs=False
                )
            else:
                # 最后一步：获取完整观测用于下一轮推理
                extracted_obs, step_reward, terminations, truncations, infos = self.step(
                    actions, auto_reset=False, return_obs=True
                )

            # 确保 step_reward 是张量，如果是 None 则使用零张量
            if step_reward is None:
                step_reward = torch.zeros(self.num_envs).to(self.device)
            elif not isinstance(step_reward, torch.Tensor):
                step_reward = torch.tensor(step_reward, dtype=torch.float32, device=self.device)
            else:
                step_reward = step_reward.to(self.device)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(
                self.device
            )
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(self.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = torch.arange(0, self.num_envs).to(dones.device)
        env_idx = env_idx[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_ids=env_idx,
        )

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _wrap_obs(self, obs):
        """
        子类需要实现此方法，将观测转换为需要的格式
        """
        raise NotImplementedError

    def add_image(self, obs):
        """
        子类需要实现此方法，从观测中提取图像用于视频记录
        """
        raise NotImplementedError

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in self.images:
            video_writer.append_data(img)
        video_writer.close()
        self.video_cnt += 1

    def close(self):
        """关闭环境（不执行本地仿真关闭）"""
        if self.server is not None:
            print(f"[RemoteIsaaclabEnv] 关闭远程连接")
            # 通知仿真端关闭
            self.server.close()
        # 不执行本地环境关闭，因为没有本地环境

    def update_reset_state_ids(self):
        """
        No multi task.
        """
        pass

    """
    Below codes are all copied from libero, thanks to the author of libero!
    """

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

