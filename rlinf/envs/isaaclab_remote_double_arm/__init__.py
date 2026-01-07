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
远程 IsaacLab 环境模块

通过网络连接远程仿真服务器，适用于以下场景：
1. 训练服务器没有 GPU 或 IsaacLab 环境
2. 需要集中管理仿真资源
3. 多个训练任务共享仿真集群

使用方法：
    在配置文件中指定远程服务器地址：
    remote_sim_url: "http://10.8.0.4:8446"
    
    然后像使用本地环境一样使用远程环境：
    env = RemoteIsaaclabStackCubeEnv(cfg, seed_offset, total_num_processes)
"""

from .tasks.stack_cube import RemoteIsaaclabStackCubeEnv

REGISTER_REMOTE_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0": RemoteIsaaclabStackCubeEnv,
}

__all__ = ["REGISTER_REMOTE_ISAACLAB_ENVS", "RemoteIsaaclabStackCubeEnv"]

