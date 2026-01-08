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
简单的仿真端 HTTP 发送器

作用：
- 训练端作为客户端，向仿真端（receiver）发送控制指令，例如 reset
- 只负责发送，不等待仿真端返回 reset 结果
"""

from __future__ import annotations

from typing import Any

import requests
import torch


class RemoteSimSender:
    """向仿真端（receiver）发送控制指令的简易 HTTP 客户端。"""

    def __init__(
        self,
        sim_host: str,
        sim_port: int = 8446,
        timeout: int = 30,
    ):
        # 自动补齐协议
        sim_host = sim_host.rstrip("/")
        if not sim_host.startswith(("http://", "https://")):
            sim_host = f"http://{sim_host}"
        self.base_url = f"{sim_host}:{sim_port}"
        self.timeout = timeout

    def send_reset(self, payload: dict | None = None) -> None:
        """
        发送 reset 请求到仿真端，不等待返回结果。

        Args:
            payload: 可选的 JSON 数据（如 seed、env_ids 等）。
        """
        url = f"{self.base_url}/reset_from_train"
        try:
            resp = requests.post(url, json=payload or {}, timeout=self.timeout)
            resp.raise_for_status()
            # 仅记录状态，不阻塞等待结果
            print(
                f"[RemoteSimSender] reset 请求已发送，status={resp.status_code}, url={url}"
            )
        except Exception as e:
            print(f"[RemoteSimSender] reset 请求发送失败: {e}, url={url}")

    def send_step(self, action: Any, payload: dict | None = None, return_obs: bool = True) -> None:
        """
        发送 step 请求到仿真端，不等待返回结果。

        Args:
            action: 动作（Tensor/list/np），会转换为可 JSON 化格式。
            payload: 附加字段。
            return_obs: 是否返回观测数据（False时只返回奖励和终止信号，减少传输量）。
        """
        url = f"{self.base_url}/step_from_train"
        data = payload.copy() if payload else {}
        data["action"] = self._to_jsonable(action)
        data["return_obs"] = return_obs  # 传递标志给仿真端
        try:
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            mode_str = "完整观测" if return_obs else "轻量模式"
            # print(
            #     f"[RemoteSimSender] step 请求已发送 ({mode_str}), status={resp.status_code}, url={url}"
            # )
        except Exception as e:
            print(f"[RemoteSimSender] step 请求发送失败: {e}, url={url}")

    def _to_jsonable(self, obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        try:
            # numpy or list-like
            return obj.tolist()  # type: ignore[attr-defined]
        except Exception:
            return obj


__all__ = ["RemoteSimSender"]

