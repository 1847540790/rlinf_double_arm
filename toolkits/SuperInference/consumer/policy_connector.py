#!/usr/bin/env python3
"""
Policy Connector - runs a policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs a policy to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Jun Lv, Han Xue, Zheng Wang
"""

import time
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Union
import yaml
import sys
import os
import signal
from multiprocessing import Process, Pipe
import rerun as rr
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from consumer.base import BaseConsumer
from policy import POLICY_CLASSES
from utils.shm_utils import (
    DEVICE_HEADER_SIZE, FRAME_HEADER_SIZE,
    pack_frame_header, get_dtype, POLICY_HEADER_SIZE,
    unpack_policy_header, get_device_info_offset, unpack_device_header,
    pack_policy_header, get_policy_data_offset, connect_to_policy_shm
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.rerun_visualization import (
    RerunVisualizer,
    _get_visualizer_app_id,
    _get_visualizer_world_path,
    _visualize_camera_frames,
    _visualize_observation_poses,
    visualize_trajectory_with_rotation,
    gripper_width_to_color,
    visualize_action_time_series,
    log_text_summary,
    set_time_context
)
from utils.config_parser import load_config
from utils.action_chunk_utils import (
    has_debug_metadata,
)

try:
    from scripts.evaluation_camera.camera_recorder import CameraRecorder
except ModuleNotFoundError:
    CameraRecorder = None  # type: ignore

ActionType = Union[np.ndarray, Dict[str, np.ndarray]]


class PolicyConnector(BaseConsumer):
    """
    Connector that:
      - Reads observations from summary SHM (via BaseConsumer)
      - Runs a policy to compute action chunking
      - Writes action chunking into policy SHM for ActionExecutor to consume
    
    Policy SHM is connect-only, created by ActionExecutor.
    """

    def __init__(self,
                 summary_shm_name: str = "device_summary_data",
                 policy_class: str = "TinyMLPPolicy",
                 policy_params: Optional[Dict[str, Any]] = None,
                 obs_devices: Optional[List[str]] = None,
                 controls: Optional[List[Dict[str, Any]]] = None,
                 fps: float = 50.0,
                 original_chunk_length: int = 16,
                 chunk_length: int = 8,
                 policy_shm_name: str = "policy_actions",
                 retry_connect_secs: float = 0.5,
                 master_device: Optional[str] = None,
                 enable_visualization: bool = False,
                 max_trajectory_points: int = 200,
                 task_config: Optional[Dict[str, Any]] = None,
                 task_config_path: Optional[str] = None,
                 robot_latency_steps: int = 0,
                 gripper_latency_steps: int = 0,
                 infer_recorder_config: Optional[Dict[str, Any]] = None,
                 manual_confirm: bool = False) -> None:
        super().__init__(summary_shm_name)
        self.policy_class = policy_class
        self.policy_params = dict(policy_params or {})
        self.obs_devices = obs_devices or []
        assert master_device is None or (
                    master_device in obs_devices), f"master_device is not in obs_devices {obs_devices}"
        self.master_device = master_device
        
        # Latency compensation
        self.robot_latency_steps = robot_latency_steps
        self.gripper_latency_steps = gripper_latency_steps
        assert robot_latency_steps >= 0 and gripper_latency_steps >= 0, f"latency_steps must be non-negative, got {robot_latency_steps}"
        assert robot_latency_steps < chunk_length and gripper_latency_steps < chunk_length, f"latency_steps ({robot_latency_steps}) must be less than chunk_length ({chunk_length})"
        
        # Store original chunk_length for policy prediction, adjusted chunk_length for output
        self.original_chunk_length = original_chunk_length
        self.chunk_length = chunk_length

        # Build controls config from YAML
        self.controls_config: List[Dict[str, Any]] = []
        if controls:
            for c in controls:
                item = {
                    'shm': c['shm'],
                    'shape': tuple(c['shape']) if 'shape' in c and c['shape'] is not None else None,
                    'dtype': c.get('dtype', 'float64'),
                }
                self.controls_config.append(item)

        self.fps = fps
        self.update_interval = 1.0 / max(self.fps, 1e-6)
        self.retry_connect_secs = retry_connect_secs

        # Runtime state per control
        self.controls_state: List[Dict[str, Any]] = []

        # Policy SHM for outputting action chunking
        self.policy_shm = None  # Will be Union[shm.SharedMemory, ReadOnlySharedMemory]
        self.policy_shm_name = policy_shm_name
        self.policy_frame_size = 0

        # Task configuration metadata
        self.task_config_path = task_config_path
        self.task_config: Dict[str, Any] = self._load_task_config(task_config_path, task_config)
        self.device_metadata: Dict[str, Dict[str, Any]] = self._parse_task_devices(self.task_config)
        observation_metadata = self._build_observation_metadata()
        self.camera_devices: List[str] = observation_metadata['camera_devices']
        self.robot_device_name: Optional[str] = observation_metadata['robot_device']

        if self.policy_class == "DiffusionPolicy" or self.policy_class =="DiffusionPolicyIPhoneUMI":
            if self.camera_devices and 'camera_devices' not in self.policy_params:
                self.policy_params['camera_devices'] = self.camera_devices
            if self.robot_device_name and 'robot_device' not in self.policy_params:
                self.policy_params['robot_device'] = self.robot_device_name

        # Build policy
        if self.policy_class not in POLICY_CLASSES:
            raise ValueError(f"Unknown policy_class: {self.policy_class}")
        self.policy = POLICY_CLASSES[self.policy_class](**self.policy_params)
        self.policy.load()
        self.action_step_interval_s: float = float(
            getattr(
                self.policy,
                "action_step_interval_s",
                getattr(self.policy, "action_step_interval", 0.05),
            )
        )
        self.action_step_interval_ns: int = int(self.action_step_interval_s * 1e9)

        manual_env = os.environ.get("SI_POLICY_MANUAL_STEP", "").strip().lower()
        self.manual_confirm = manual_confirm or manual_env in ("1", "true", "yes", "on")
        if self.manual_confirm:
            logger.info("PolicyConnector: manual confirmation enabled before publishing actions")

        # Visualization configuration
        self.enable_visualization: bool = enable_visualization
        self.max_trajectory_points: int = max_trajectory_points
        self.chunk_trajectories: Dict[str, Dict[str, Any]] = {}
        self.chunk_counter: int = 0
        self.visualizer: Optional[RerunVisualizer] = None
        # if self.enable_visualization:
        #     self._initialize_visualizer()

        # Cache control metadata for policy prediction
        self.action_configs: List[Dict[str, Any]] = self._build_action_configs()
        
        # Inference recorder configuration
        self.infer_recorder_config: Dict[str, Any] = dict(infer_recorder_config or {})
        self.infer_recorder_process: Optional[Process] = None
        self._infer_recorder_started: bool = False

        # Log latency compensation configuration
        if self.robot_latency_steps > 0 or self.gripper_latency_steps > 0:
            logger.info(f"PolicyConnector: Latency compensation enabled. "
                        f"Robot: discarding first {self.robot_latency_steps} actions, "
                        f"Gripper: discarding first {self.gripper_latency_steps} actions "
                        f"from {self.original_chunk_length}-length chunks. "
                        f"Output chunk length: {self.chunk_length}")
        else:
            logger.info(f"PolicyConnector: No latency compensation. Chunk length: {self.chunk_length}")
        

    def _connect_policy_shm(self) -> None:
        """Connect to policy SHM created by ActionExecutor."""
        try:
            start_time = time.time()
            timeout_sec = max(10.0, 60.0)
            while True:
                try:
                    # Connect to existing policy SHM using unified interface (read-write for policy connector)
                    self.policy_shm = connect_to_policy_shm(read_only=False)
                    break
                except Exception as exc:
                    if time.time() - start_time < timeout_sec:
                        logger.info(f"PolicyConnector: Waiting for policy SHM 'policy_data'... retry in {self.retry_connect_secs}s")
                        time.sleep(self.retry_connect_secs)
                        continue
                    raise exc

            # Read manager header to get device count and cache device info
            buf = self.policy_shm.buf
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header['device_count']

            # Cache device info for efficient writing
            self.policy_devices = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset+DEVICE_HEADER_SIZE])

                device_name = device_header['device_type']
                shape = device_header['shape']  # [chunk_length, action_dim]
                frame_size = device_header['frame_size']

                self.policy_devices.append({
                    'device_name': device_name,
                    'shape': shape,
                    'frame_size': frame_size,
                    'data_offset': None
                })

            logger.info(f"PolicyConnector: Connected to policy SHM: {self.policy_shm_name} | devices={len(self.policy_devices)}")
            for device_info in self.policy_devices:
                logger.debug(f"Device {device_info['device_name']}: expected_shape={device_info['shape']}")

        except Exception as e:
            logger.error(f"PolicyConnector: Failed to connect to policy SHM: {e}")
            raise




    def _initialize_visualizer(self) -> None:
        try:
            app_id = _get_visualizer_app_id()
            world_path = _get_visualizer_world_path()
            self.visualizer = RerunVisualizer(app_id, spawn=True)
            self.visualizer.setup_3d_world(world_path, coordinate_system="z_up")
        except Exception as exc:
            logger.error(f"PolicyConnector: Failed to initialize visualization: {exc}")
            self.visualizer = None
            self.enable_visualization = False


    def _parse_task_devices(self, task_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        if not isinstance(task_config, dict):
            return metadata
        for device_cfg in task_config.get('devices', []):
            device_class = device_cfg.get('class')
            device_id = device_cfg.get('device_id')
            if device_class is None or device_id is None:
                continue
            name = f"{device_class}_{device_id}"
            metadata[name] = device_cfg
        return metadata

    def _build_action_configs(self) -> List[Dict[str, Any]]:
        action_configs: List[Dict[str, Any]] = []
        for cfg in self.controls_config:
            shm_name = cfg['shm']
            device_name = shm_name.replace('_control', '') if shm_name.endswith('_control') else shm_name
            device_meta = self.device_metadata.get(device_name, {})
            shape = cfg.get('shape')
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                action_dim = int(shape[0])
            else:
                action_dim = int(device_meta.get('control_shape', [1])[0]) if device_meta.get('control_shape') else 1
            action_configs.append({
                'device_name': device_name,
                'action_dim': action_dim,
                'shm_name': shm_name,
                'metadata': device_meta
            })
        return action_configs

    def _augment_chunk_with_debug_metadata(self, chunk: np.ndarray, iteration: int) -> np.ndarray:
        """Insert [step_index, iteration] columns after the left-arm block for debugging."""

        if chunk.ndim != 2 or chunk.shape[1] < 8 or has_debug_metadata(chunk):
            return chunk

        dtype = chunk.dtype
        num_steps = chunk.shape[0]
        step_idx = np.arange(num_steps, dtype=dtype).reshape(-1, 1)
        iteration_col = np.full((num_steps, 1), iteration, dtype=dtype)
        left = chunk[:, :8]
        right = chunk[:, 8:]
        return np.concatenate([left, step_idx, iteration_col, right], axis=1)

    def _load_task_config(self, config_path: Optional[str], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if override is not None:
            return override
        if config_path is None:
            return {}
        try:
            return load_config(config_path)
        except FileNotFoundError:
            logger.error(f"PolicyConnector: task config file not found: {config_path}")
        except Exception as exc:
            logger.error(f"PolicyConnector: failed to load task config {config_path}: {exc}")
        return {}

    def _build_observation_metadata(self) -> Dict[str, Any]:
        camera_devices: List[str] = []
        robot_device: Optional[str] = None

        search_order = self.obs_devices if self.obs_devices else list(self.device_metadata.keys())
        for device_name in search_order:
            meta = self.device_metadata.get(device_name, {})
            cls_name = (meta.get('class') or meta.get('device_class') or '').lower()
            if not cls_name:
                # Fall back to the actual device name when metadata is missing
                cls_name = device_name.lower()
            if 'camera' in cls_name:
                camera_devices.append(device_name)
            elif 'robot' in cls_name and robot_device is None:
                robot_device = device_name

        return {
            'camera_devices': camera_devices,
            'robot_device': robot_device
        }

    # Recorder management -------------------------------------------------

    def _should_start_infer_recorder(self) -> bool:
        if not self.infer_recorder_config:
            return False
        enabled = self.infer_recorder_config.get('enabled', False)
        if not enabled:
            return False
        if CameraRecorder is None:
            logger.error("PolicyConnector: CameraRecorder module not available; cannot start inference recorder")
            return False
        return True

    def _start_infer_recorder(self) -> None:
        if self._infer_recorder_started or not self._should_start_infer_recorder():
            return

        recorder_cfg = self.infer_recorder_config

        recorder_kwargs: Dict[str, Any] = {
            'camera_id': int(recorder_cfg.get('camera_id', 0)),
            'save_path': recorder_cfg.get('save_path', './recordings'),
            'width': recorder_cfg.get('width'),
            'height': recorder_cfg.get('height'),
            'fps': recorder_cfg.get('fps'),
            'scale': recorder_cfg.get('scale', 1.0),
            'show_window': recorder_cfg.get('show_window', False),
            'enable_recording': recorder_cfg.get('enable_recording', True),
            'auto_start_recording': recorder_cfg.get('auto_start_recording', True),
            'record_key': recorder_cfg.get('record_key', 'r'),
            'realsense_serial': recorder_cfg.get('realsense_serial', 819612073139),
            'use_realsense': recorder_cfg.get('use_realsense',False),
            'window_name': recorder_cfg.get('window_name')
        }

        parent_conn, child_conn = Pipe()

        def _recorder_entry(connection) -> None:
            try:
                recorder = CameraRecorder(**recorder_kwargs)  # type: ignore[arg-type]
                recorder.start_display(ready_conn=connection)
            except Exception as exc:  # pragma: no cover - subprocess error reporting
                logger.error(f"PolicyConnector: Inference recorder process encountered an error: {exc}")
                try:
                    connection.send({'status': 'failed', 'error': str(exc)})
                except Exception:
                    pass
                finally:
                    try:
                        connection.close()
                    except Exception:
                        pass

        try:
            self.infer_recorder_process = Process(
                target=_recorder_entry,
                args=(child_conn,),
                name="InferenceCameraRecorder",
                daemon=True
            )
            self.infer_recorder_process.start()

            # Wait for recorder to signal readiness
            start_time = time.time()
            recorder_ready_timeout = float(self.infer_recorder_config.get('ready_timeout', 10.0))
            ready_info = None

            while time.time() - start_time < recorder_ready_timeout:
                if parent_conn.poll(0.1):
                    ready_info = parent_conn.recv()
                    break
            if ready_info is None:
                raise TimeoutError("Inference recorder did not signal readiness in time")

            if ready_info.get('status') != 'ready':
                error_msg = ready_info.get('error', 'unknown error')
                raise RuntimeError(f"Inference recorder failed to initialize: {error_msg}")

            self._infer_recorder_started = True

            # Log helpful recorder information
            record_key = ready_info.get('record_key', 'r')
            auto_recording = ready_info.get('auto_recording', False)
            show_window = ready_info.get('show_window', False)

            logger.info("PolicyConnector: Recorder window is visible and ready.")
            if show_window:
                logger.info(
                    f"PolicyConnector: Recorder controls — press '{record_key}' to toggle recording, 'c' for grayscale, 'q' to close the window."
                )
            if auto_recording:
                logger.info("PolicyConnector: Recorder auto-started recording; video is being saved.")
            else:
                logger.info(
                    f"PolicyConnector: Recorder waiting for manual recording start (key '{record_key}')."
                )

            if self.enable_visualization:
                logger.info("PolicyConnector: Visualization pipeline will start once recorder window is confirmed ready.")

        except Exception as exc:
            logger.error(f"PolicyConnector: Failed to start inference recorder process: {exc}")
            if self.infer_recorder_process is not None:
                try:
                    if self.infer_recorder_process.is_alive():
                        self.infer_recorder_process.terminate()
                        self.infer_recorder_process.join(timeout=2.0)
                except Exception:
                    pass
            self.infer_recorder_process = None
            self._infer_recorder_started = False
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass

    def _stop_infer_recorder(self) -> None:
        if not self._infer_recorder_started:
            return

        process = self.infer_recorder_process
        if process is None:
            return

        try:
            if process.is_alive():
                logger.info("PolicyConnector: Sending SIGINT to inference recorder process for graceful shutdown")
                try:
                    process.send_signal(signal.SIGINT)
                except AttributeError:
                    os.kill(process.pid, signal.SIGINT)

                process.join(timeout=5.0)

                if process.is_alive():
                    logger.warning("PolicyConnector: Recorder did not exit after SIGINT, terminating forcefully")
                    process.terminate()
                    process.join(timeout=2.0)

                if process.is_alive():
                    logger.error("PolicyConnector: Recorder still alive, force killing")
                    process.kill()
        except Exception as exc:
            logger.error(f"PolicyConnector: Error while stopping inference recorder process: {exc}")
        finally:
            self.infer_recorder_process = None
            self._infer_recorder_started = False


    def _requires_history(self) -> bool:
        obs_horizon = getattr(self.policy, 'obs_horizon', 1)
        return (
            isinstance(obs_horizon, int) and obs_horizon > 1 and
            hasattr(self.policy, 'obs_time_interval') and
            hasattr(self.policy, 'downsample')
        )


    def _collect_observation_history(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        obs_horizon = getattr(self.policy, 'obs_horizon', 1)
        if not isinstance(obs_horizon, int) or obs_horizon <= 1:
            return {k: [v] for k, v in latest_obs.items()}

        interval_sec = float(getattr(self.policy, 'obs_time_interval', 0.0))
        downsample = int(getattr(self.policy, 'downsample', 1))
        if interval_sec <= 0.0 or downsample <= 0:
            logger.warning("PolicyConnector: Invalid observation interval configuration; falling back to single observation")
            return {k: [v] for k, v in latest_obs.items()}

        dt_ns = int(interval_sec * downsample * 1e9)
        obs_list: List[Dict[str, np.ndarray]] = [latest_obs]
        history_obs_times = [latest_timestamp - (i + 1) * dt_ns for i in range(obs_horizon - 1)]

        for target_timestamp in history_obs_times:
            past_frame = self.find_frame_by_timestamp(target_timestamp)
            if past_frame is None or 'data' not in past_frame:
                return None
            past_obs = {key: value[1] for key, value in past_frame['data'].items()}
            obs_list.append(past_obs)

        if len(obs_list) != obs_horizon:
            return None

        obs_list.reverse()
        obs_batch: Dict[str, List[np.ndarray]] = {}
        for key in latest_obs.keys():
            try:
                obs_batch[key] = [step[key] for step in obs_list]
            except KeyError:
                logger.warning(f"PolicyConnector: Missing key {key} in historical observations")
                return None

        return obs_batch


    def _prepare_policy_input(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int
    ) -> Optional[Dict[str, Any]]:
        if not latest_obs:
            return None

        if self._requires_history():
            history = self._collect_observation_history(latest_obs, latest_timestamp)
            if history is None:
                return None
            return history

        return latest_obs


    def _parse_action_chunk(
        self,
        device_name: str,
        action_chunk: np.ndarray
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        logger.debug(f"PolicyConnector: _parse_action_chunk not implemented for device {device_name}")
        return None


    def _visualize_predicted_trajectory(
        self,
        device_name: str,
        action_chunk_dict: Dict[str, np.ndarray],
        timestamp_ns: int
    ) -> None:
        if not self.enable_visualization or device_name not in action_chunk_dict:
            return

        try:
            action_chunk = action_chunk_dict[device_name]
            parsed_actions = self._parse_action_chunk(device_name, action_chunk)
            if not parsed_actions:
                return

            chunk_key = f"{device_name}_chunk_{self.chunk_counter}"
            self.chunk_trajectories[chunk_key] = {
                'timestamp': timestamp_ns,
                'arms': parsed_actions
            }

            self._visualize_action_values(device_name, parsed_actions, timestamp_ns)
            self._visualize_chunk_trajectories(device_name)

            self.chunk_counter += 1
        except Exception as exc:
            logger.error(f"PolicyConnector: Error visualizing trajectory for {device_name}: {exc}")





    def _visualize_action_values(
        self,
        device_name: str,
        parsed_actions: Dict[str, Dict[str, np.ndarray]],
        timestamp_ns: int
    ) -> None:
        try:
            set_time_context(timestamp_ns, "timestamp")
            for arm_name, arm_data in parsed_actions.items():
                positions = arm_data.get('position')
                if positions is None:
                    continue
                quat_xyzw = arm_data.get("quaternion")
                r = R.from_quat(quat_xyzw)
                rpy = r.as_euler("xyz", degrees=False)
                grippers = arm_data.get('gripper')
                if grippers is None:
                    continue
                visualize_action_time_series(
                    "time_series/action_trends",
                    device_name,
                    arm_name,
                    positions,
                    grippers,
                    timestamp_ns,
                    rpy=rpy,
                    step_offset_ns=self.action_step_interval_ns,
                )

            set_time_context(timestamp_ns, "timestamp")
        except Exception as exc:
            logger.error(f"PolicyConnector: Error visualizing action values: {exc}")


    def _visualize_chunk_trajectories(self, device_name: str) -> None:
        try:
            max_chunks_to_show = 5
            device_chunks = [key for key in self.chunk_trajectories.keys() if device_name in key]
            # logger.info("visualize trajectories for robot")
            def _chunk_idx(chunk_key: str) -> int:
                try:
                    part = chunk_key.split("_chunk_")[-1]
                    num_str = ''.join(ch for ch in part if ch.isdigit())
                    return int(num_str) if num_str else -1
                except Exception:
                    return -1

            recent_chunks = sorted(device_chunks, key=_chunk_idx)[-max_chunks_to_show:]

            for chunk_idx, chunk_key in enumerate(recent_chunks):
                chunk_data = self.chunk_trajectories[chunk_key]
                chunk_timestamp = chunk_data['timestamp']
                chunk_arms = chunk_data['arms']

                denominator = max(1, len(recent_chunks) - 1)
                opacity = 0.3 + 0.7 * (chunk_idx / denominator)

                for arm_name, arm_data in chunk_arms.items():
                    positions = arm_data.get('position')
                    quaternions = arm_data.get('quaternion')
                    grippers = arm_data.get('gripper')

                    if positions is None or quaternions is None:
                        continue

                    if len(positions) < 2:
                        continue

                    chunk_arm_path = f"{_get_visualizer_world_path()}/chunk_trajectories/{chunk_key}_{arm_name}"

                    segment_colors: List[List[int]] = []
                    if grippers is not None:
                        for i in range(len(positions)):
                            gripper_width = float(grippers[i][0]) if grippers.ndim > 1 else float(grippers[i])
                            base_color = gripper_width_to_color(gripper_width)
                            r = int(base_color[0] * opacity)
                            g = int(base_color[1] * opacity)
                            b = int(base_color[2] * opacity)
                            segment_colors.append([r, g, b])
                    else:
                        default_palette = {
                            'left_arm': [255, 120, 120],
                            'right_arm': [120, 200, 255],
                            'single_arm': [200, 200, 200]
                        }
                        base_color = default_palette.get(arm_name, [200, 200, 200])
                        adjusted = [int(component * opacity) for component in base_color]
                        segment_colors = [adjusted[:] for _ in range(len(positions))]

                    visualize_trajectory_with_rotation(
                        chunk_arm_path,
                        positions,
                        quaternions,
                        segment_colors,
                        arrow_scale=0.02,
                        rotation_scale=0.03,
                        show_every_n=4
                    )

                    log_text_summary(
                        f"{chunk_arm_path}/info",
                        f"Chunk {chunk_key} | {arm_name} | Time: {chunk_timestamp/1e9:.3f}s"
                    )

            total_allowed = max_chunks_to_show * 2
            if len(self.chunk_trajectories) > total_allowed:
                all_chunks = sorted(self.chunk_trajectories.keys(), key=_chunk_idx)
                chunks_to_remove = all_chunks[:-total_allowed]
                for chunk_key in chunks_to_remove:
                    del self.chunk_trajectories[chunk_key]
        except Exception as exc:
            logger.error(f"PolicyConnector: Error visualizing chunk trajectories: {exc}")


    def _write_action_chunk_dict(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: int=-1) -> None:
        """Write dict-based action chunking to policy SHM."""
        if not self.policy_shm:
            return
        try:
            buf = self.policy_shm.buf
            timestamp_ns = time.time_ns()
            
            # Update manager header to indicate data is available
            device_count = len(self.policy_devices)
            manager_header = pack_policy_header(device_count, timestamp_ns)  # Use timestamp to indicate data availability
            buf[:POLICY_HEADER_SIZE] = manager_header
            
            # Calculate frame sizes for get_data_offset
            frame_sizes = [device['frame_size'] for device in self.policy_devices]
            
            # Write data for each device
            for i, device_info in enumerate(self.policy_devices):
                device_name = device_info['device_name']
                expected_shape = device_info['shape']  # [chunk_length, action_dim]
                
                # Get action chunk for this device
                if device_name not in action_chunk_dict:
                    logger.warning(f"No action chunk for device: {device_name}")
                    continue

                action_chunk = action_chunk_dict[device_name]

                # Ensure correct shape
                if len(action_chunk.shape) == 1:
                    action_chunk = action_chunk.reshape(1, -1)

                # Verify shape matches expected
                if action_chunk.shape != tuple(expected_shape):
                    logger.error(
                        f"Shape mismatch for {device_name}: got {action_chunk.shape}, expected {expected_shape}"
                    )
                    continue
                
                # Calculate data offset for this device
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                
                # Write frame header with observation timestamp
                frame_header = pack_frame_header(observation_timestamp)
                buf[data_offset:data_offset+FRAME_HEADER_SIZE] = frame_header
                
                # Write chunk data
                data_start = data_offset + FRAME_HEADER_SIZE
                chunk_bytes = action_chunk.astype(np.float64).tobytes()
                data_end = data_start + len(chunk_bytes)
                buf[data_start:data_end] = chunk_bytes
            
            logger.debug(f"Wrote action chunks to policy SHM: devices={list(action_chunk_dict.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to write action chunk dict: {e}")

    def _confirm_actions(self, action_chunk_dict: Dict[str, np.ndarray]) -> bool:
        """Optional manual gate before publishing actions to shared memory."""
        if not self.manual_confirm:
            return True

        try:
            logger.info("Manual confirmation required before publishing policy actions")
            for device_name, chunk in action_chunk_dict.items():
                preview = np.array2string(chunk, precision=4, suppress_small=True)
                logger.info(f"  Device {device_name} chunk:\n{preview}")

            while True:
                user_input = input("Press Enter=执行, 's'=跳过, 'q'=取消手动确认: ").strip().lower()
                if user_input in ("", "y", "yes"):
                    return True
                if user_input == "s":
                    logger.info("Operator requested to skip this chunk.")
                    return False
                if user_input == "q":
                    logger.info("Manual confirmation disabled by operator.")
                    self.manual_confirm = False
                    return True
                logger.info("Unrecognized input '%s'. Please按 Enter / 's' / 'q'.", user_input)

        except EOFError:
            logger.warning("Manual confirmation input unavailable; disabling manual mode.")
            self.manual_confirm = False
            return True

    def _assemble_observation_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """Assemble observation dictionary using atomic read operation."""
        try:
            # Use the new atomic read function
            all_data = self.read_all_device_data()
            if all_data is None:
                return None, None
            
            # Extract just the data arrays (not timestamps)
            obs_dict = {}
            all_obs_time = []
            for device_name, (timestamp, data) in all_data.items():
                obs_dict[device_name] = data
                if self.master_device is None:
                    all_obs_time.append(timestamp)

            # Compute the timestamp for this frame of observation
            if self.master_device is None:
                obs_time = int(np.array(all_obs_time).mean().item()) # use averaging time of all devices if the `master_device` is invalid
            else:
                obs_time = all_data[self.master_device][0]
            return obs_dict, obs_time
            
        except Exception as e:
            logger.error(f"Error assembling observation dict: {e}")
            return None, None

    def run(self) -> None:
        # Start inference recorder before connecting to SHM so recording includes initial frames
        self._start_infer_recorder()

        if not self.connect():
            self._stop_infer_recorder()
            logger.error("PolicyConnector failed to connect to summary SHM")
            return

        logger.info("PolicyConnector: Camera recorder is active; policy inference will begin now.")

        self._connect_policy_shm()

        logger.info(f"PolicyConnector running with {self.fps} Hz inference rate. Press Ctrl+C to stop...")
        logger.info(f"Connected to policy SHM: {self.policy_shm_name}")
        self.running = True
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        try:
            iteration = 0
            while self.running:
                obs_dict, obs_time = self._assemble_observation_dict()
                # print(list(obs_dict.keys()))
                if not obs_dict or obs_time is None:
                    continue
                
                policy_input = self._prepare_policy_input(obs_dict, obs_time)
                if policy_input is None:
                    continue

                _visualize_camera_frames(self.camera_devices, obs_dict, obs_time)
                _visualize_observation_poses(
                    self.robot_device_name,policy_input, obs_time
                )
                if hasattr(self.policy, "set_prediction_context"):
                    self.policy.set_prediction_context(timestamp_ns=obs_time)
                action_chunk_dict = self.policy.predict(
                    policy_input,
                    action_configs=self.action_configs,
                    chunk_length=self.original_chunk_length
                )
                # Apply latency compensation with different steps for robots and grippers
                if self.robot_latency_steps > 0 or self.gripper_latency_steps > 0:
                    for device_name in action_chunk_dict:
                        original_chunk = action_chunk_dict[device_name]
      

                        # Check if this is dual-arm data (16 dimensions)
                        if original_chunk.shape[1] == 16:
                            # Extract robot and gripper data
                            # Left arm: robot [0:7], gripper [7]
                            # Right arm: robot [8:15], gripper [15]
                            left_robot = original_chunk[:, 0:7]
                            left_gripper = original_chunk[:, 7:8]
                            right_robot = original_chunk[:, 8:15]
                            right_gripper = original_chunk[:, 15:16]

                            # Apply different latency compensation
                            left_robot_comp = left_robot[self.robot_latency_steps:, :]
                            left_gripper_comp = left_gripper[self.robot_latency_steps:, :]
                            right_robot_comp = right_robot[self.robot_latency_steps:, :]
                            right_gripper_comp = right_gripper[self.robot_latency_steps:, :]

                            # Truncate to chunk_length
                            left_robot_comp = left_robot_comp[:self.chunk_length, :]
                            left_gripper_comp = left_gripper_comp[:self.chunk_length, :]
                            right_robot_comp = right_robot_comp[:self.chunk_length, :]
                            right_gripper_comp = right_gripper_comp[:self.chunk_length, :]

                            # Recombine
                            original_chunk = np.concatenate(
                                [
                                    left_robot_comp,
                                    left_gripper_comp,
                                    right_robot_comp,
                                    right_gripper_comp,
                                ],
                                axis=1,
                            )

                            logger.debug(
                                f"Applied separate latency compensation for {device_name}: "
                                f"{original_chunk.shape} -> {original_chunk.shape}, "
                                f"robot_steps={self.robot_latency_steps}, gripper_steps={self.gripper_latency_steps}"
                            )
                        else:
                            # For other dimensions, use robot_latency_steps as default
                            original_chunk = original_chunk[
                                self.robot_latency_steps :, :
                            ]
                            original_chunk = original_chunk[: self.chunk_length, :]
                            logger.debug(
                                f"Applied uniform latency compensation for {device_name}: "
                                f"{original_chunk.shape} -> {original_chunk.shape}"
                            )

                        action_chunk_dict[device_name] = original_chunk
                else:
                    # No latency compensation, just truncate to chunk_length
                    for device_name in action_chunk_dict:
                        chunk = action_chunk_dict[device_name][:self.chunk_length, :]
                        action_chunk_dict[device_name] = chunk

                # TODO: remove this gripper width debug hack later
                action_chunk_dict['RizonRobot_1'][:, 7] = action_chunk_dict['RizonRobot_1'][:, 7] * 0.07 / 0.085
                action_chunk_dict['RizonRobot_1'][:, 15] = action_chunk_dict['RizonRobot_1'][:, 15] * 0.07 / 0.085

                # if self.enable_visualization:
                #     for device_name in action_chunk_dict.keys():
                #         self._visualize_predicted_trajectory(device_name, action_chunk_dict, obs_time)

                if not self._confirm_actions(action_chunk_dict):
                    logger.info("PolicyConnector: action chunk skipped by operator")
                else:
                #     # if iteration % 5 == 0:
                    self._write_action_chunk_dict(action_chunk_dict, obs_time)
                wait_for_next_iteration()
                iteration = (iteration + 1) % 10
                # logger.info(f"action_chunk: {action_chunk_dict['RizonRobot_2']}")

        except KeyboardInterrupt:
            logger.info("Stopping PolicyConnector...")
        finally:
            self._stop_infer_recorder()
            self.stop()
            if self.policy_shm:
                try:
                    self.policy_shm.close()
                except Exception:
                    pass


def load_policy_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy Connector - run a policy and write action chunking to policy SHM")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML path for policy connector config")

    args = parser.parse_args()

    cfg_yaml = load_policy_yaml(args.config)
    policy_cfg = cfg_yaml.get('policy', {})

    policy_class = policy_cfg.get('class', 'TinyMLPPolicy')
    policy_params = policy_cfg.get('params', {})
    fps = policy_cfg.get('fps', 50.0)
    chunk_length = policy_cfg.get('chunk_length', 10)
    original_chunk_length = policy_cfg.get('original_chunk_length', chunk_length)
    policy_shm_name = policy_cfg.get('policy_shm_name', 'policy_actions')
    robot_latency_steps = policy_cfg.get('robot_latency_steps', 0)
    gripper_latency_steps = policy_cfg.get('gripper_latency_steps', 0)


    obs_cfg = policy_cfg.get('obs', {})
    obs_devices = obs_cfg.get('devices')

    controls_cfg = policy_cfg.get('controls', [])

    master_device = policy_cfg.get('master_device', None)

    connector = PolicyConnector(
        summary_shm_name=cfg_yaml.get('summary_shm', 'device_summary_data'),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        fps=fps,
        original_chunk_length=original_chunk_length,
        chunk_length=chunk_length,
        policy_shm_name=policy_shm_name,
        master_device=master_device,
        robot_latency_steps=robot_latency_steps,
        gripper_latency_steps=gripper_latency_steps,
        infer_recorder_config=policy_cfg.get('infer_recorder')
    )

    connector.run()


if __name__ == "__main__":
    main() 
