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
远程 IsaacLab 环境服务器
在训练服务器上运行，接收仿真端的请求，使用本地 isaaclab 环境执行操作
"""

import base64
import io
import os
import pickle
import queue
import sys
import threading
from threading import Lock, Event
from typing import Optional

import flask
import torch

from rlinf.envs.isaaclab_remote_double_arm.venv import SubProcIsaacLabEnv

# 强制无缓冲输出
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


class RemoteIsaacLabServer:
    """
    远程 IsaacLab 环境服务器
    
    在训练服务器上运行，接收仿真端的 HTTP 请求，
    使用本地 isaaclab 环境执行操作，并将结果返回给仿真端。
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8446,
        local_device: str = "cuda",
    ):
        """
        Args:
            host: 服务器监听地址
            port: 服务器监听端口
            local_device: 本地设备
        """
        self.host = host
        self.port = port
        self.local_device = local_device
        self.app = flask.Flask(__name__)
        self.env_config = None
        self._lock = Lock()  # 保护操作的锁
        
        # 同步机制：用于训练代码和 HTTP 请求之间的同步
        self.reset_request_queue = queue.Queue()  # reset 请求参数队列
        self.reset_result_queue = queue.Queue()  # reset 结果队列（从仿真端接收）
        self.step_request_queue = queue.Queue()  # step 请求参数队列
        self.step_result_queue = queue.Queue()   # step 结果队列（从仿真端接收）
        self.reset_event = Event()                # reset 事件
        self.step_event = Event()                 # step 事件
        self.reset_result_event = Event()         # reset 结果事件
        self.step_result_event = Event()         # step 结果事件
        # 握手同步：训练端等待仿真端首次连通
        self.handshake_queue = queue.Queue()
        self.handshake_event = Event()
        
        # Step 计数器
        self.step_count = 0
        # Reset 计数器
        self.reset_count = 0
        # 注册路由
        self._register_routes()
        
        print(f"[RemoteIsaacLabServer] 服务器初始化完成，监听地址: {host}:{port}")
    
    def _register_routes(self):
        """注册 HTTP 路由"""
        
        @self.app.route('/ping', methods=['GET'])
        def ping():
            """健康检查"""
            print(f"[RemoteIsaacLabServer] [ping] 收到健康检查请求")
            return flask.jsonify({"status": "ok", "message": "server is running"})

        @self.app.route('/handshake', methods=['POST'])
        def handshake():
            """仿真端首次握手，告知训练端连接已建立"""
            print(f"[RemoteIsaacLabServer] [handshake] 收到握手请求")
            try:
                data = flask.request.json or {}
                payload = data.get("data")
                self.handshake_queue.put(payload)
                self.handshake_event.set()
                print(f"[RemoteIsaacLabServer] [handshake] 收到数据: {payload}")
                return flask.jsonify({"status": "success"})
            except Exception as e:
                print(f"[RemoteIsaacLabServer] [handshake] 处理失败: {e}")
                return flask.jsonify({"status": "error", "detail": str(e)}), 500
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            """重置环境（接收仿真端发送的数据，不执行本地仿真）"""
            print(f"[RemoteIsaacLabServer] [reset] ========== 收到重置请求（来自仿真端） ==========")
            try:
                data = flask.request.json or {}
                seed = data.get("seed")
                env_ids = data.get("env_ids")
                
                # 检查是否有数据字段（仿真端发送的完整结果）
                if "data" in data:
                    # 仿真端发送了完整的结果数据
                    result_data = data["data"]
                    obs, info = self._deserialize_data(result_data)
                    self.reset_count += 1
                    
                    print(f"[RemoteIsaacLabServer] [reset] 收到仿真端发送的 reset 数据（第 {self.reset_count} 个 reset）")
                    
                    # 将请求参数放入队列，通知训练代码
                    self.reset_request_queue.put((seed, env_ids))
                    self.reset_event.set()
                    
                    # 将结果数据放入队列，供训练代码获取
                    self.reset_result_queue.put((obs, info))
                    self.reset_result_event.set()
                    
                    return flask.jsonify({
                        "status": "success",
                        "message": "数据已接收"
                    })
                else:
                    # 只有请求参数，等待训练代码处理（旧逻辑，保留兼容性）
                    print(f"[RemoteIsaacLabServer] [reset] 参数: seed={seed}, env_ids={env_ids}")
                    
                    # 将请求放入队列，等待训练代码处理
                    self.reset_request_queue.put((seed, env_ids))
                    self.reset_event.set()  # 通知训练代码
                    
                    # 等待训练代码返回结果（但训练代码不执行仿真，只返回占位符）
                    print(f"[RemoteIsaacLabServer] [reset] 等待训练代码响应...")
                    result = self.reset_result_queue.get(timeout=10)  # 等待结果
                    
                    if isinstance(result, Exception):
                        print(f"[RemoteIsaacLabServer] [reset] 重置失败: {result}")
                        return flask.jsonify({
                            "status": "error",
                            "detail": str(result)
                        }), 500
                    
                    obs, info = result
                    # 序列化结果
                    result_data = self._serialize_data((obs, info))
                    
                    print(f"[RemoteIsaacLabServer] [reset] 返回结果给仿真端")
                    return flask.jsonify({
                        "status": "success",
                        "data": result_data
                    })
                    
            except Exception as e:
                print(f"[RemoteIsaacLabServer] [reset] 重置失败: {e}")
                return flask.jsonify({
                    "status": "error",
                    "detail": str(e)
                }), 500
        
        @self.app.route('/step', methods=['POST'])
        def step():
            """执行一步（接收仿真端发送的完整结果，不执行本地仿真）"""
            try:
                data = flask.request.json or {}

                if "data" not in data:
                    return flask.jsonify({
                        "status": "error",
                        "detail": "缺少 data 字段（应包含仿真结果）"
                    }), 400

                result_data = data["data"]
                encoded_size = len(result_data) if isinstance(result_data, str) else None
                step_result = self._deserialize_data(result_data)
                self.step_count += 1
                extra_info = f", tuple_len={len(step_result)}" if isinstance(step_result, tuple) else ""
                print(
                    f"[RemoteIsaacLabServer] [step] 收到仿真端发送的 step 数据（第 {self.step_count} 个 step），"
                    f"类型={type(step_result)}, 编码大小={encoded_size}{extra_info}"
                , end='\r', flush=True, file=sys.stderr)
                sys.stderr.flush()

                # 仅将结果数据放入队列，供训练代码获取；不再处理/记录 action
                self.step_result_queue.put(step_result)
                self.step_result_event.set()

                return flask.jsonify({
                    "status": "success",
                    "message": "数据已接收"
                })

            except Exception as e:
                print(f"[RemoteIsaacLabServer] [step] step 失败: {e}")
                return flask.jsonify({
                    "status": "error",
                    "detail": str(e)
                }), 500
        
        @self.app.route('/close', methods=['POST'])
        def close():
            """关闭环境（不执行本地仿真关闭）"""
            print(f"[RemoteIsaacLabServer] [close] ========== 收到关闭请求 ==========")
            try:
                with self._lock:
                    self.env_config = None
                    print(f"[RemoteIsaacLabServer] [close] 连接已关闭（无本地仿真）")
                    
                    return flask.jsonify({
                        "status": "success"
                    })
                    
            except Exception as e:
                print(f"[RemoteIsaacLabServer] [close] 关闭失败: {e}")
                return flask.jsonify({
                    "status": "error",
                    "detail": str(e)
                }), 500
        
        @self.app.route('/status', methods=['GET'])
        def status():
            """获取状态"""
            print(f"[RemoteIsaacLabServer] [status] 收到状态查询请求")
            return flask.jsonify({
                "status": "ok",
                "initialized": self.env_config is not None,
                "device": self.local_device,
                "mode": "remote_only"  # 标识不运行本地仿真
            })
    
    def _serialize_data(self, data) -> str:
        """序列化数据为 base64 字符串"""
        buffer = io.BytesIO()
        data = self._move_to_cpu(data)
        pickle.dump(data, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _deserialize_data(self, data_str: str):
        """反序列化 base64 字符串"""
        buffer = io.BytesIO(base64.b64decode(data_str))
        data = pickle.load(buffer)
        return self._move_to_device(data, self.local_device)
    
    def _move_to_cpu(self, data):
        """递归将 tensor 移到 CPU"""
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_cpu(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_cpu(item) for item in data)
        return data
    
    def _move_to_device(self, data, device: str):
        """递归将 tensor 移到指定设备"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(item, device) for item in data)
        return data
    
    def run(self, debug: bool = False):
        """启动服务器"""
        print(f"[RemoteIsaacLabServer] 启动服务器，监听 {self.host}:{self.port}")
        # 禁用 Flask 的日志输出缓冲
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR) 
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)
    
    def set_env(self, env):
        """设置环境实例（由外部传入）"""
        with self._lock:
            self.env = env
            print(f"[RemoteIsaacLabServer] 环境实例已设置")
    
    def wait_for_reset_request(self, timeout=None):
        """等待仿真端发送 reset 请求，返回 (seed, env_ids)"""
        import time
        print(f"[RemoteIsaacLabServer] 训练代码等待 reset 请求...")
        start_time = time.time()
        last_reminder_time = start_time
        reminder_interval = 1.0
        
        while True:
            # 每次等待1秒，检查是否有数据
            if self.reset_event.wait(timeout=None):
                if not self.reset_request_queue.empty():
                    # 取出请求参数
                    seed, env_ids = self.reset_request_queue.get()
                    self.reset_event.clear()
                    elapsed = int(time.time() - start_time)
                    print(f"[RemoteIsaacLabServer] ✓ 收到 reset 请求，已等待 {elapsed} 秒")
                    return seed, env_ids
                else:
                    # 事件被设置但队列为空，清空事件继续等待
                    self.reset_event.clear()
            
            # 定期输出提醒
            current_time = time.time()
            if current_time - last_reminder_time >= reminder_interval:
                elapsed = int(current_time - start_time)
                print(f"[RemoteIsaacLabServer] ⏳ 仍在等待 reset 请求，已等待 {elapsed} 秒...")
                last_reminder_time = current_time
            
            # 检查超时
            if timeout is not None and (current_time - start_time) >= timeout:
                print(f"[RemoteIsaacLabServer] ✗ 等待 reset 请求超时（{timeout} 秒）")
                return None, None

    def wait_for_handshake(self, expected_data=None, timeout=None):
        """等待仿真端握手数据（例如 '123'），收到匹配数据即返回成功"""
        import time
        print(f"[RemoteIsaacLabServer] 训练端等待仿真端握手数据...")
        start_time = time.time()
        last_reminder_time = start_time
        reminder_interval = 5.0

        while True:
            if self.handshake_event.wait(timeout=1.0):
                if not self.handshake_queue.empty():
                    payload = self.handshake_queue.get()
                    self.handshake_event.clear()

                    if expected_data is None or payload == expected_data:
                        elapsed = int(time.time() - start_time)
                        print(f"[RemoteIsaacLabServer] ✓ 收到握手数据 '{payload}'，握手成功，耗时 {elapsed} 秒")
                        return True  # 返回 True 表示握手成功
                    else:
                        print(f"[RemoteIsaacLabServer] 握手数据不匹配，收到: {payload}，期望: {expected_data}，继续等待...")
                        continue
                else:
                    self.handshake_event.clear()

            current_time = time.time()
            if current_time - last_reminder_time >= reminder_interval:
                elapsed = int(current_time - start_time)
                # 使用 stderr 确保在 Ray 分布式环境中实时输出
                print(f"[RemoteIsaacLabServer] ⏳ 正在等待握手数据，已等待 {elapsed} 秒...", end='\r', file=sys.stderr, flush=True)
                sys.stderr.flush()
                last_reminder_time = current_time

            if timeout is not None and (current_time - start_time) >= timeout:
                print(f"[RemoteIsaacLabServer] ✗ 等待握手数据超时（{timeout} 秒）")
                return None
    
    def wait_for_reset_result(self, timeout=None):
        """等待仿真端发送 reset 结果数据"""
        import time
        print(f"[RemoteIsaacLabServer] 训练代码等待 reset 结果数据...")
        start_time = time.time()
        last_reminder_time = start_time
        reminder_interval = 10.0
        
        while True:
            # 每次等待1秒，检查是否有数据
            if self.reset_result_event.wait(timeout=1.0):
                if not self.reset_result_queue.empty():
                    result = self.reset_result_queue.get()
                    self.reset_result_event.clear()
                    elapsed = int(time.time() - start_time)
                    print(f"[RemoteIsaacLabServer] ✓ 收到 reset 结果数据，已等待 {elapsed} 秒")
                    return result
                else:
                    # 事件被设置但队列为空，清空事件继续等待
                    self.reset_result_event.clear()
            
            # 定期输出提醒
            current_time = time.time()
            if current_time - last_reminder_time >= reminder_interval:
                elapsed = int(current_time - start_time)
                print(f"[RemoteIsaacLabServer] ⏳ 仍在等待 reset 结果数据，已等待 {elapsed} 秒...")
                sys.stdout.flush()
                last_reminder_time = current_time
            
            # 检查超时
            if timeout is not None and (current_time - start_time) >= timeout:
                print(f"[RemoteIsaacLabServer] ✗ 等待 reset 结果数据超时（{timeout} 秒）")
                return None
    
    def put_reset_result_ack(self):
        """训练代码确认收到 reset 结果（占位方法）"""
        pass
    
    def wait_for_step_result(self, timeout=None):
        """等待仿真端发送 step 结果数据"""
        import time
        print(f"[RemoteIsaacLabServer] 训练代码等待第 {self.step_count + 1} 个 step 结果数据...", end='\r', flush=True, file=sys.stderr)
        sys.stderr.flush()
        start_time = time.time()
        last_reminder_time = start_time
        reminder_interval = 10.0
        
        while True:
            # 每次等待1秒，检查是否有数据
            if self.step_result_event.wait(timeout=None):
                if not self.step_result_queue.empty():
                    result = self.step_result_queue.get()
                    self.step_result_event.clear()
                    elapsed = float(time.time() - start_time)
                    print(f"[RemoteIsaacLabServer] ✓ 收到 第 {self.step_count + 1} 个 step 结果数据，已等待 {elapsed} 秒", end='\r', flush=True, file=sys.stderr)
                    sys.stderr.flush()
                    return result
                else:
                    # 事件被设置但队列为空，清空事件继续等待
                    self.step_result_event.clear()
            
            # 定期输出提醒
            current_time = time.time()
            if current_time - last_reminder_time >= reminder_interval:
                elapsed = int(current_time - start_time)
                print(f"[RemoteIsaacLabServer] ⏳ 仍在等待 step 结果数据，已等待 {elapsed} 秒...")
                last_reminder_time = current_time
            
            # 检查超时
            if timeout is not None and (current_time - start_time) >= timeout:
                print(f"[RemoteIsaacLabServer] ✗ 等待 step 结果数据超时（{timeout} 秒）")
                return None
    
    def put_step_result_ack(self):
        """训练代码确认收到 step 结果（占位方法）"""
        pass
    
    def close(self):
        """关闭服务器连接"""
        print(f"[RemoteIsaacLabServer] 关闭服务器连接")

