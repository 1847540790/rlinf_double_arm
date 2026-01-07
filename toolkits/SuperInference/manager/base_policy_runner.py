#!/usr/bin/env python3
"""
Policy Runner - reads action chunking from policy and dispatches to robot control SHMs.

Author: Jun Lv, Zheng Wang
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import signal
from collections import deque
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    unpack_device_header, unpack_buffer_header, unpack_frame_header,
    DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    pack_policy_header, POLICY_HEADER_SIZE, unpack_policy_header,
    calculate_policy_shm_size, get_device_info_offset, get_policy_data_offset,
    create_policy_shm, connect_to_policy_shm, connect_to_shared_memory
)
from utils.logger_config import logger
from utils.config_parser import ensure_config_dict, load_config

from manager.utils import load_chunk_manager
from visualizers.action_visualizer import ActionVisualizer


class BasePolicyRunner:
    """
    Policy Runner for reading action chunking and dispatching to robot control SHMs.

    This class supports device-specific action delays to synchronize execution across
    devices with different latencies. Actions are queued per device, and devices with
    larger delays have their actions held longer before dispatch.

    Args:
        config: Configuration dictionary (alternative to config_path)
        config_path: Path to configuration YAML file
        policy_shm_name: Name of the policy shared memory
        robot_control_shms: List of robot control SHM configurations
        execution_fps: Execution frequency in Hz
        chunk_length: Length of action chunks
        chunk_manager: Chunk manager configuration
        enable_action_visualizer: Enable action visualization
        action_visualizer_config: Action visualizer configuration
        device_delays: Dict mapping device names to delay frames. Devices with higher
                      latency should have smaller delays, while devices with lower
                      latency should have larger delays to ensure synchronized execution.
                      Example: {'RizonRobot_1': 0, 'RizonRobot_2': 10}
                      This means RizonRobot_2 actions will be delayed by 10 frames.
    """

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            *,
            config_path: str = "config.yaml",
            policy_shm_name: str = "policy_actions",
            robot_control_shms: Optional[List[Dict[str, Any]]] = None,
            execution_fps: float = 100.0,
            chunk_length: int = 10,
            chunk_manager: Optional[Dict[str, Any]] = None,
            enable_action_visualizer: bool = False,
            action_visualizer_config: Optional[Dict[str, Any]] = None,
            device_delays: Optional[Dict[str, int]] = None,
            device_dimension_delays: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.config_path = config_path
        self.policy_shm_name = policy_shm_name
        self.robot_control_shms = robot_control_shms or []
        self.execution_fps = execution_fps
        self.chunk_length = chunk_length

        self.running = False
        self.policy_shm = None
        self.robot_states = []

        # Initialize execution state
        self.current_chunk_timestamp = 0.0
        self.current_step = 0
        self.current_action_chunk_dict = None  # Cache current chunk dict
        self.last_read_timestamp = 0  # Track last read timestamp
        self.chunk_manager = load_chunk_manager(chunk_manager or {}, execution_fps=self.execution_fps)
        # Retry settings for connecting to robot control SHMs (to avoid race with robot startup)
        self.retry_connect_secs: float = 0.5
        self.connect_timeout_secs: float = 60.0

        # Device delay management: device_name -> delay_frames (for backward compatibility)
        self.device_delays = device_delays or {}
        # Action queues for each device: device_name -> deque of actions
        self.device_action_queues: Dict[str, deque] = {}
        # Fine-grained delay config: device_name -> list of (indices, delay_frames)
        # Example: {'RizonRobot_1': [([7, 8, 9], 10), ([10, 11], 5)]}
        self.device_dimension_delays: Dict[str, List[Tuple[List[int], int]]] = {}
        # Store first action for each device to repeat during delay period
        self.device_first_actions: Dict[str, Optional[np.ndarray]] = {}

        # Parse dimension delays from config format
        if device_dimension_delays:
            for device_name, delay_configs in device_dimension_delays.items():
                parsed_delays = []
                for delay_config in delay_configs:
                    indices = delay_config.get('indices', [])
                    delay = delay_config.get('delay', 0)
                    # Support range notation: [start, end] means [start, start+1, ..., end]
                    # if len(indices) == 2 and isinstance(indices[0], int) and isinstance(indices[1], int):
                    #     indices = list(range(indices[0], indices[1] + 1))
                    parsed_delays.append((indices, delay))
                self.device_dimension_delays[device_name] = parsed_delays

        # Action visualizer
        self.enable_action_visualizer = enable_action_visualizer
        action_viz_cfg = action_visualizer_config or {}
        # Parse time_range if provided (can be None, or tuple/list [start, end])
        time_range = action_viz_cfg.get('time_range', None)
        if time_range is not None:
            if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                time_range = tuple(time_range)
            else:
                logger.warning(f"Invalid time_range format: {time_range}, expected [start, end] or None. Ignoring.")
                time_range = None

        self.action_visualizer = ActionVisualizer(
            max_history=action_viz_cfg.get('max_history', 500),
            update_interval_ms=action_viz_cfg.get('update_interval_ms', 50),
            figsize=tuple(action_viz_cfg.get('figsize', [12, 8])),
            window_size=action_viz_cfg.get('window_size', 100),
            enable=enable_action_visualizer,
            output_dir=action_viz_cfg.get('output_dir', './action_visualizations'),
            html_filename=action_viz_cfg.get('html_filename', 'action_visualization.html'),
            time_range=time_range
        )
        # Visualization updates are now handled automatically in background thread
        # No need for manual update counter

        if config is None:
            self.config = load_config(self.config_path)
        else:
            self.config = ensure_config_dict(config)

        # Initialize device action queues based on device delays
        self._initialize_device_queues()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"ActionExecutor initialized: policy_shm={policy_shm_name}, fps={execution_fps}")
        if self.device_delays:
            logger.info(f"Device delays configured: {self.device_delays}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.stop()

    def _initialize_device_queues(self) -> None:
        """
        Initialize action queues for each device based on configured delays.

        For devices with dimension-level delays, we create separate queues for each
        dimension group. For devices with only whole-action delays, we create a single queue.
        """
        # Get device names from robot_control_shms
        for robot_cfg in self.robot_control_shms:
            shm_name = robot_cfg.get('shm', '')
            # Extract device name from SHM name (e.g., "SimRobot_0_control" -> "SimRobot_0")
            device_name = shm_name.replace('_control', '') if shm_name.endswith('_control') else shm_name

            # Initialize first action storage
            self.device_first_actions[device_name] = None

            # Check if this device has dimension-level delays
            if device_name in self.device_dimension_delays:
                # Create a dict of queues: dimension_group_id -> deque
                # We'll use a special structure: dict with 'dimension_queues' key
                dimension_queues = {}
                for idx, (indices, delay) in enumerate(self.device_dimension_delays[device_name]):
                    dimension_queues[idx] = deque()
                # Also create a queue for dimensions not specified (default delay)
                dimension_queues['default'] = deque()
                self.device_action_queues[device_name] = dimension_queues
                logger.info(
                    f"Initialized dimension-level queues for {device_name}: {len(dimension_queues) - 1} groups + default")
            else:
                # Initialize single queue for whole action
                self.device_action_queues[device_name] = deque()

        logger.info(f"Initialized action queues for devices: {list(self.device_action_queues.keys())}")

    def start(self) -> bool:
        """Start the action executor."""
        if not self.connect():
            logger.error("Failed to connect, cannot start executor")
            return False

        logger.info("Starting ActionExecutor")
        self.running = True

        try:
            while self.running:
                self._update_execution()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

        return True

    def stop(self) -> None:
        """Stop the action executor."""
        self.running = False

        # Close action visualizer
        if self.enable_action_visualizer:
            try:
                self.action_visualizer.close()
            except Exception as e:
                logger.warning(f"Error closing action visualizer: {e}")

        # Close robot control SHMs
        for robot_state in self.robot_states:
            try:
                if robot_state['shm']:
                    robot_state['shm'].close()
            except Exception:
                pass

        # Close policy SHM
        if self.policy_shm:
            try:
                self.policy_shm.close()
            except Exception:
                pass

        logger.info("ActionExecutor stopped")

    def connect(self) -> bool:
        """Connect to robot control SHMs and create policy SHM."""
        try:
            # Connect to robot control SHMs
            for robot_cfg in self.robot_control_shms:
                robot_state = self._connect_robot_control(robot_cfg)
                if robot_state is None:
                    logger.error(f"Failed to connect to robot control SHM: {robot_cfg['shm']}")
                    return False
                self.robot_states.append(robot_state)

            # Create policy SHM
            self._create_policy_shm()
            logger.info("ActionExecutor connected successfully")
            return True

        except Exception as e:
            logger.error(f"ActionExecutor failed to connect: {e}")
            return False

    def _connect_robot_control(self, robot_cfg: Dict) -> Optional[Dict]:
        """Connect to a single robot control SHM using unified interface (read-write for policy runner)."""
        control_shm_name = robot_cfg.get('shm')
        if not control_shm_name:
            return None

        start_time = time.time()
        last_error: Optional[Exception] = None
        while True:
            try:
                # Use unified SHM interface for read-write connection (policy runner needs to write)
                robot_shm = connect_to_shared_memory(control_shm_name, read_only=False)
                buf = robot_shm.buf
                device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
                buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])

                action_dim = robot_cfg.get('shape', [1])[0]
                control_frame_size = device_header['frame_size']

                robot_state = {
                    'device_info': robot_cfg,
                    'shm': robot_shm,
                    'action_dim': action_dim,
                    'frame_size': control_frame_size,
                    'buffer_size': buffer_header['buffer_size'],
                }

                logger.info(f"Connected to robot control SHM: {control_shm_name} (READ-WRITE)")
                return robot_state
            except Exception as e:
                last_error = e
                if time.time() - start_time < self.connect_timeout_secs:
                    logger.info(
                        f"ActionExecutor: Waiting for control SHM '{control_shm_name}'... retry in {self.retry_connect_secs}s"
                    )
                    time.sleep(self.retry_connect_secs)
                    continue
                logger.error(f"Failed to connect to robot control SHM after retries: {e}")
                return None

    def _create_policy_shm(self) -> None:
        """Create policy SHM for PolicyConnector to write action chunking using unified interface."""
        try:
            # Calculate total action dimension and frame sizes for each device
            device_count = len(self.robot_states)
            frame_sizes = []

            for robot_state in self.robot_states:
                action_dim = robot_state['action_dim']
                # Each device frame: [FRAME_HEADER] + [chunk_data]
                chunk_data_size = self.chunk_length * action_dim * 8  # 8 bytes for float64, default date type in this shm is float64, which will be convert to right type.
                frame_size = FRAME_HEADER_SIZE + chunk_data_size
                frame_sizes.append(frame_size)

            # Use unified SHM creation interface
            self.policy_shm = create_policy_shm(device_count, frame_sizes)

            total_size = calculate_policy_shm_size(device_count, frame_sizes)
            logger.info(f"Created policy SHM: {self.policy_shm_name}, size={total_size} bytes")
            logger.info(
                f"Policy header: {POLICY_HEADER_SIZE} bytes, Device headers: {device_count * DEVICE_HEADER_SIZE} bytes, Data: {sum(frame_sizes)} bytes")

            # Initialize header
            self._initialize_policy_header()

        except Exception as e:
            logger.error(f"Failed to create policy SHM: {e}")
            raise

    def _initialize_policy_header(self) -> None:
        """Initialize policy SHM header with manager and device headers."""
        device_count = len(self.robot_states)

        # Pack manager header (timestamp will be updated when data is written)
        manager_header = pack_policy_header(device_count, 0)  # 0 timestamp initially

        # Write manager header to SHM
        self.policy_shm.buf[:POLICY_HEADER_SIZE] = manager_header

        # Pack and write device headers for each robot control
        for i, robot_state in enumerate(self.robot_states):
            robot_cfg = robot_state['device_info']
            shm_name = robot_cfg.get('shm', '')
            device_name = shm_name.replace('_control', '') if shm_name.endswith('_control') else shm_name

            action_dim = robot_state['action_dim']
            chunk_data_size = self.chunk_length * action_dim * 8  # 8 bytes for float64
            frame_size = FRAME_HEADER_SIZE + chunk_data_size

            # Use device_name as device_type for policy SHM
            device_header = pack_device_header(
                device_type=device_name,
                device_id=0,  # Not relevant for policy SHM
                fps=self.execution_fps,
                data_dtype="float64",
                shape=[self.chunk_length, action_dim],  # chunk shape
                frame_size=frame_size,
                hardware_latency_ms=0.0
            )

            # Write device header to SHM
            device_offset = get_device_info_offset(i)
            self.policy_shm.buf[device_offset:device_offset + DEVICE_HEADER_SIZE] = device_header

    def _update_execution(self) -> None:
        """
        Update execution loop - read action chunk, manage delays, and dispatch actions.

        This method handles the complete execution pipeline:
        1. Read new action chunks from policy SHM
        2. Get next action from chunk manager
        3. Enqueue actions to per-device delay queues
        4. Dequeue actions that have reached required delay
        5. Dispatch ready actions to robot control SHMs
        """
        start_time = time.time()

        try:
            # Read new action chunk from policy SHM if available
            chunk_result = self._read_latest_policy_frame()

            # Put new chunk into chunk manager if it's newer than the last one
            if chunk_result is not None:
                latest_chunk_dict, timestamp_ns = chunk_result
                if self.chunk_manager.is_new_chunk(timestamp_ns):
                    self.chunk_manager.put(latest_chunk_dict, timestamp_ns)
            # Get next action step from chunk manager
            if not self.chunk_manager.is_empty():
                action_dict = self.chunk_manager.get()
                if action_dict is not None:
                    # Visualize action if enabled
                    if self.enable_action_visualizer:
                        self.action_visualizer.add_action(action_dict)

                    # === Device Delay Management ===
                    # Enqueue actions to per-device queues for delay management
                    for device_name, action in action_dict.items():
                        if device_name in self.device_action_queues:
                            self._enqueue_action_with_delays(device_name, action)

                    # Dequeue actions from queues that have reached required delay depth
                    ready_actions = {}
                    for device_name in self.device_action_queues.keys():
                        ready_action = self._dequeue_action_with_delays(device_name)
                        if ready_action is not None:
                            ready_actions[device_name] = ready_action

                    # Dispatch ready actions to robot control SHMs

                    if ready_actions:
                        # logger.info(f"Dispatching {len(ready_actions)} actions to robots")
                        # logger.info(f"Ready actions: {ready_actions}")
                        self._dispatch_action_to_robots(ready_actions)

        except Exception as e:
            logger.error(f"Error in execution update: {e}")

        # Control execution frequency - account for processing time
        elapsed = time.time() - start_time
        target_interval = 1.0 / self.execution_fps
        sleep_time = max(0.0, target_interval - elapsed)

        if sleep_time > 0:
            time.sleep(sleep_time)

    def _enqueue_action_with_delays(self, device_name: str, action: np.ndarray) -> None:
        """
        Enqueue action with dimension-level delay support.

        If device has dimension-level delays configured, split the action and enqueue
        each dimension group separately. Otherwise, enqueue the whole action.

        Also stores the first action received for this device to use during delay period.
        """
        # Store first action if not already stored
        if self.device_first_actions[device_name] is None:
            self.device_first_actions[device_name] = action.copy()
            logger.info(f"Stored first action for {device_name} to use during delay period")

        queue_structure = self.device_action_queues[device_name]

        # Check if this device uses dimension-level delays
        if isinstance(queue_structure, dict):
            # Dimension-level delays
            dimension_delays = self.device_dimension_delays[device_name]

            # Track which dimensions have been assigned to groups
            assigned_dims = set()

            # Enqueue each dimension group
            for group_id, (indices, delay) in enumerate(dimension_delays):
                # Extract values for these dimensions
                dim_values = action[indices].copy()
                queue_structure[group_id].append(dim_values)
                assigned_dims.update(indices)
                logger.debug(f"Enqueued {device_name} dims {indices} to group {group_id}, "
                             f"queue_len={len(queue_structure[group_id])}, delay={delay}")

            # Enqueue remaining dimensions to default queue
            all_dims = set(range(len(action)))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                default_values = action[default_dims].copy()
                queue_structure['default'].append(default_values)
                logger.debug(f"Enqueued {device_name} dims {default_dims} to default queue, "
                             f"queue_len={len(queue_structure['default'])}")
        else:
            # Whole-action delay (backward compatibility)
            queue_structure.append(action.copy())
            delay_frames = self.device_delays.get(device_name, 0)
            logger.debug(
                f"Enqueued action for {device_name}, queue length: {len(queue_structure)}, delay: {delay_frames}")

    def _dequeue_action_with_delays(self, device_name: str) -> Optional[np.ndarray]:
        """
        Dequeue action with dimension-level delay support.

        If device has dimension-level delays configured, dequeue from each dimension
        queue and reconstruct the full action. Otherwise, dequeue the whole action.

        During the delay period (when queues are not ready), dimensions that need delay
        will use values from the first action received, while other dimensions use current values.
        """
        queue_structure = self.device_action_queues[device_name]
        first_action = self.device_first_actions.get(device_name)

        # Check if this device uses dimension-level delays
        if isinstance(queue_structure, dict):
            # Dimension-level delays
            dimension_delays = self.device_dimension_delays[device_name]

            # Check which queues are ready and which are not
            queues_ready = {}
            for group_id, (indices, delay) in enumerate(dimension_delays):
                queues_ready[group_id] = len(queue_structure[group_id]) > delay
                if not queues_ready[group_id]:
                    logger.debug(f"Queue not ready for {device_name} group {group_id} (dims {indices}): "
                                 f"current={len(queue_structure[group_id])}, required={delay + 1}")

            # Check default queue (no delay)
            default_delay = self.device_delays.get(device_name, 0)
            default_ready = len(queue_structure['default']) > default_delay
            if not default_ready:
                logger.debug(f"Default queue not ready for {device_name}: "
                             f"current={len(queue_structure['default'])}, required={default_delay + 1}")

            # If no queues are ready at all, return None (no action to dispatch yet)
            if not any(queues_ready.values()) and not default_ready:
                return None

            # Reconstruct action: use first action as base, then fill in ready dimensions
            # Determine total dimensions
            if first_action is not None:
                total_dims = len(first_action)
            else:
                # Fallback: calculate from queue structure
                total_dims = 0
                for group_id, (indices, _) in enumerate(dimension_delays):
                    total_dims = max(total_dims, max(indices) + 1)
                if queue_structure['default']:
                    assigned_dims = set()
                    for indices, _ in dimension_delays:
                        assigned_dims.update(indices)
                    default_count = len(queue_structure['default'][0])
                    total_dims = max(total_dims, len(assigned_dims) + default_count)

            # Start with first action as base (for dimensions that are still in delay period)
            if first_action is not None:
                reconstructed_action = first_action.copy()
            else:
                reconstructed_action = np.zeros(total_dims, dtype=np.float64)

            assigned_dims = set()

            # Fill in dimension groups (only if queue is ready)
            for group_id, (indices, delay) in enumerate(dimension_delays):
                if queues_ready[group_id]:
                    # Queue is ready, use dequeued value
                    dim_values = queue_structure[group_id].popleft()
                    reconstructed_action[indices] = dim_values
                    assigned_dims.update(indices)
                    logger.debug(f"Dequeued {device_name} group {group_id} (dims {indices}), "
                                 f"remaining={len(queue_structure[group_id])}")
                else:
                    # Queue not ready, keep first action values (already in reconstructed_action)
                    assigned_dims.update(indices)
                    logger.debug(
                        f"Using first action for {device_name} group {group_id} (dims {indices}) - queue not ready")

            # Fill in default dimensions (only if queue is ready)
            all_dims = set(range(total_dims))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                if default_ready:
                    # Queue is ready, use dequeued value
                    default_values = queue_structure['default'].popleft()
                    reconstructed_action[default_dims] = default_values
                    logger.debug(f"Dequeued {device_name} default (dims {default_dims}), "
                                 f"remaining={len(queue_structure['default'])}")
                else:
                    # Queue not ready, keep first action values
                    logger.debug(
                        f"Using first action for {device_name} default (dims {default_dims}) - queue not ready")

            return reconstructed_action
        else:
            # Whole-action delay (backward compatibility)
            delay_frames = self.device_delays.get(device_name, 0)

            if len(queue_structure) > delay_frames:
                ready_action = queue_structure.popleft()
                logger.debug(
                    f"Dequeued action for {device_name} (delay={delay_frames}, remaining in queue={len(queue_structure)})")
                return ready_action
            else:
                # During delay period, use first action if available
                if first_action is not None and len(queue_structure) > 0:
                    logger.debug(
                        f"Using first action for {device_name} during delay period (current: {len(queue_structure)}, required: {delay_frames + 1})")
                    return first_action.copy()
                else:
                    logger.debug(
                        f"Waiting for {device_name} queue to fill (current: {len(queue_structure)}, required: {delay_frames + 1})")
                    return None

    def _read_latest_policy_frame(self) -> Optional[Tuple[Dict[str, np.ndarray], int]]:
        """Read latest action chunk dict from policy SHM using standard manager/device structure."""
        try:
            if not self.policy_shm:
                return None

            buf = self.policy_shm.buf

            # Read manager header
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header['device_count']
            update_timestamp = manager_header['update_timestamp']

            # If timestamp is 0, no data has been written yet
            if update_timestamp == 0:
                return None

            # Calculate frame sizes for get_data_offset
            frame_sizes = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset + DEVICE_HEADER_SIZE])
                frame_sizes.append(device_header['frame_size'])

            # Read device headers and data
            action_chunk_dict = {}
            latest_timestamp = 0

            for i in range(device_count):
                # Read device header
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset + DEVICE_HEADER_SIZE])

                device_name = device_header['device_type']
                shape = device_header['shape']  # [chunk_length, action_dim]
                frame_size = device_header['frame_size']

                # Read device data
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)

                # Read frame header
                frame_header = unpack_frame_header(buf[data_offset:data_offset + FRAME_HEADER_SIZE])
                timestamp_ns = frame_header['timestamp_ns']

                # If timestamp is 0, no data for this device
                if timestamp_ns == 0:
                    continue

                # Read chunk data
                data_start = data_offset + FRAME_HEADER_SIZE
                data_end = data_start + (frame_size - FRAME_HEADER_SIZE)

                # Extract chunk data: buf is raw buffer, slicing works directly
                sliced_buf = buf[data_start:data_end]
                chunk_data = np.frombuffer(sliced_buf, dtype=np.float64)
                action_chunk = chunk_data.reshape(shape)

                action_chunk_dict[device_name] = action_chunk
                latest_timestamp = max(latest_timestamp, timestamp_ns)

            if not action_chunk_dict:
                return None

            return action_chunk_dict, latest_timestamp

        except Exception as e:
            logger.error(f"Failed to read policy frame: {e}")
            return None

    def _dispatch_action_to_robots(self, action_dict: Dict[str, np.ndarray]) -> None:
        """
        Dispatch action dict to robot control SHMs.

        This function simply writes the provided actions to their corresponding robot control SHMs.
        Delay management is handled in _update_execution().
        """
        if not self.robot_states or not action_dict:
            return

        for robot_state in self.robot_states:
            # Map robot config to device name
            robot_cfg = robot_state['device_info']
            shm_name = robot_cfg.get('shm', '')

            # Extract device name from SHM name (e.g., "SimRobot_0_control" -> "SimRobot_0")
            device_name = shm_name.replace('_control', '') if shm_name.endswith('_control') else shm_name

            # Find corresponding action in the dict
            if device_name in action_dict:
                robot_action = action_dict[device_name]
                self._write_action_to_robot(robot_state, robot_action)
                logger.debug(
                    f"Dispatched action to {device_name}: {robot_action[:3] if len(robot_action) > 3 else robot_action}")
            else:
                logger.debug(f"No action found for device {device_name} in this dispatch cycle")

    def _write_action_to_robot(self, robot_state: Dict[str, Any], action: np.ndarray) -> None:
        """Write action to robot control SHM."""
        try:
            robot_shm = robot_state['shm']
            buf = robot_shm.buf

            # Read current buffer state
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            write_index = buffer_header['write_index']
            buffer_size = buffer_header['buffer_size']
            frame_size = robot_state['frame_size']

            # Calculate frame offset and write frame header
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + write_index * frame_size
            current_time_ns = time.time_ns()
            frame_header = pack_frame_header(current_time_ns)
            buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = frame_header

            # Write action data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + (frame_size - FRAME_HEADER_SIZE)

            action_bytes = action.tobytes()
            if len(action_bytes) != (frame_size - FRAME_HEADER_SIZE):
                data_len = frame_size - FRAME_HEADER_SIZE
                if len(action_bytes) > data_len:
                    action_bytes = action_bytes[:data_len]
                else:
                    action_bytes = action_bytes + b"\x00" * (data_len - len(action_bytes))

            buf[data_start:data_end] = action_bytes

            # Update buffer header
            new_write_index = (write_index + 1) % buffer_size
            new_current_frames_count = min(buffer_header['current_frames_count'] + 1, buffer_size)
            new_buffer_header = pack_buffer_header(buffer_size, new_current_frames_count, new_write_index)
            buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = new_buffer_header

        except Exception as e:
            logger.error(f"Failed to write action to robot: {e}")