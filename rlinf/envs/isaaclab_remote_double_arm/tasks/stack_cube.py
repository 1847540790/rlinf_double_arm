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
远程版本的 Stack Cube 环境
通过网络从远程服务器获取观测数据
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import copy

from rlinf.envs.isaaclab_remote_double_arm.isaaclab_env import RemoteIsaaclabBaseEnv

class RemoteIsaaclabStackCubeEnv(RemoteIsaaclabBaseEnv):
    """
    远程 Stack Cube 环境
    
    与本地版本接口相同，但数据来自远程仿真服务器
    """
    
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
        )

    def _wrap_obs(self, obs):
        """
        将远程服务器返回的观测数据转换为标准格式
        使用 openpi_policy 的处理函数来对齐观测值和模型输入
        
        期望输入格式：
        {
            "policy": {
                "Camera_left": np.ndarray,  # shape: (num_envs, H, W, 3), dtype=uint8
                "Camera_right": np.ndarray,  # shape: (num_envs, H, W, 3), dtype=uint8
                "Robot_state": np.ndarray,   # shape: (num_envs, 14), dtype=float64
            }
        }
        """
        
        # 检查必需的数据字段
        if "Camera_left" not in obs.get("policy", {}) or "Camera_right" not in obs.get("policy", {}):
            raise KeyError("观测数据中缺少必需的字段: Camera_left 或 Camera_right")
        if "Robot_state" not in obs.get("policy", {}):
            raise KeyError("观测数据中缺少必需的字段: Robot_state")
        
        state = obs["policy"]["Robot_state"]
        left_wrist_image = obs["policy"]["Camera_left"].permute(0, 3, 1, 2)
        right_wrist_image = obs["policy"]["Camera_right"].permute(0, 3, 1, 2)
        instruction = [self.task_description] * self.num_envs
        
        env_obs = {
            "states": state,  # [num_envs, 14]
            "left_wrist_images": left_wrist_image,  # [num_envs, 3, 224, 224] - 使用复数形式以匹配 obs_processor 检测
            "right_wrist_images": right_wrist_image,  # [num_envs, 3, 224, 224] - 使用复数形式以匹配 obs_processor 检测
            "task_descriptions": instruction,  # List[str] of length num_envs
        }
        # print(env_obs.keys())
        # print(env_obs["states"].shape)
        # print(env_obs["left_wrist_images"].shape)
        # print(env_obs["right_wrist_images"].shape)
        # print(f"task_descriptions: {len(env_obs['task_descriptions'])} items, first: {env_obs['task_descriptions'][0] if env_obs['task_descriptions'] else 'empty'}")
        return env_obs


