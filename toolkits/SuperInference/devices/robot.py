#!/usr/bin/env python3
"""
Robot Device - Base robot class and PyBullet-based simulated robot.

Author: Jun Lv,Zixi Ying
"""

import time
import numpy as np
import multiprocessing.shared_memory as shm
from typing import Optional, Dict, Any, Union, Tuple, List
import rerun as rr
from devices.base import BaseDevice
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    calculate_device_shm_size, DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    get_dtype, create_control_shm, connect_to_control_shm, connect_to_device_shm,
    connect_to_shared_memory
)
from utils.logger_config import logger
from utils.transform import compute_deviation, similarity_transform, axis_similarity_transform
from scipy.spatial.transform import Rotation as R
from utils.transform import scalar_transform
from utils.rerun_log_utils import log_shared_rerun_timestamp
from utils.rerun_visualization import set_time_context
from utils.calibration_utils import load_tcp_umi_transforms


class BaseRobot(BaseDevice):
    """
    Base robot class that extends BaseDevice with control capabilities.
    Provides dual SHM buffers: one for state data (inherited) and one for control commands.
    """
    
    def __init__(self, device_id=0, data_shape=(7,), control_shape=(7,), 
                 fps=100.0, data_dtype=np.float64, control_dtype=np.float64, command_fps = 100.0,
                 buffer_size=100, control_buffer_size=50, hardware_latency_ms=0.0, 
                 control_shm_name: Union[str, List[str], None] = None, airexo_type="ver2", 
                 **kwargs):
        """
        Initialize the base robot.
        
        Args:
            device_id: Unique identifier for this robot instance
            data_shape: Shape of robot state data (e.g., joint positions, velocities)
            control_shape: Shape of control command data (e.g., target joint positions)
            fps: Control frequency in Hz
            data_dtype: Data type for state data (string or numpy dtype)
            control_dtype: Data type for control data (string or numpy dtype)
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            control_shm_name: Name of existing control SHM to connect to (if None, will create new one)
            airexo_type: Version of airexo ("ver1", "ver2", None)
            **kwargs: Additional arguments passed to BaseDevice
        """
        # Initialize base device for state data
        super().__init__(device_id=device_id, data_shape=data_shape, fps=fps, 
                        data_dtype=data_dtype, buffer_size=buffer_size, hardware_latency_ms=hardware_latency_ms, **kwargs)
        # Robot command executer fps
        self.command_fps = command_fps
        
        # Robot-specific attributes
        self.airexo_type = airexo_type
        # Convert control_shape to tuple if it's a list
        self.control_shape = tuple(control_shape) if isinstance(control_shape, list) else control_shape
        
        self.control_dtype = control_dtype
            
        self.control_buffer_size = control_buffer_size
        
        # Control buffer management
        self.control_shared_memory = None

        # Calculate control buffer layout
        self._calculate_control_buffer_layout()

        self.control_shm_names = control_shm_name
        if isinstance(self.control_shm_names, str):
            self.control_shm_names = [self.control_shm_names]
        
    def _calculate_control_buffer_layout(self) -> None:
        """Calculate the layout of the control command buffer."""
        # Calculate control data size per command
        numpy_control_dtype = get_dtype(self.control_dtype)
        bytes_per_element = np.dtype(numpy_control_dtype).itemsize
        control_data_size = np.prod(self.control_shape) * bytes_per_element
        
        # Control frame header: timestamp (8 bytes)
        self.control_frame_header_size = FRAME_HEADER_SIZE
        self.control_frame_data_size = control_data_size
        self.control_frame_size = self.control_frame_header_size + self.control_frame_data_size
        
        # Pre-calculate control frame offsets
        self.control_frame_offsets = [
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * self.control_frame_size 
            for i in range(self.control_buffer_size)
        ]
        
        # Total control SHM size: device_header + buffer_header + control_frames
        self.total_control_shm_size = calculate_device_shm_size(self.control_buffer_size, self.control_frame_data_size)
        
        # Control shared memory name
        self.default_control_shm_name = f"{self.device_name}_{self.device_id}_control"
        
    def _create_control_shared_memory(self) -> None:
        """Create shared memory for control commands using unified interface."""
        # Phase 1: Create writable SHM for initialization
        control_shm = create_control_shm(
            device_name=self.device_name,
            device_id=self.device_id,
            buffer_size=self.control_buffer_size,
            frame_data_size=self.control_frame_data_size,
            readonly_after_creation=False  # Initially writable for initialization
        )
        
        # Store in control shared memory dict (don't overwrite user-specified control_shm_names)
        if self.control_shm_names is None:
            self.control_shm_names = [self.default_control_shm_name]
        self.control_shared_memory = {self.default_control_shm_name: control_shm}
        
        logger.info(f"Created control shared memory: {self.default_control_shm_name} ({self.total_control_shm_size:,} bytes)")
        
        # Phase 2: Initialize control buffer header
        self._initialize_control_buffer_header()
        logger.info(f"Initialized control buffer header")
        
        # Phase 3: Convert to read-only after initialization
        from utils.shm_utils import make_readonly
        readonly_control_shm = make_readonly(control_shm)
        self.control_shared_memory[self.default_control_shm_name] = readonly_control_shm
        
        logger.info(f"Control SHM converted to READ-ONLY after initialization")
        
    def _connect_to_existing_control_shm(self, shm_name: Optional[str] = None) -> bool:
        """
        Connect to an existing control shared memory in READ-ONLY mode using unified interface.
        
        Args:
            shm_name: Name of the existing control SHM (if None, uses default name)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to connect to the SHM directly
            # Let the unified interface handle the connection logic
            self.control_shared_memory[shm_name] = connect_to_shared_memory(shm_name, read_only=True)
            logger.info(f"Connected to control SHM: {shm_name} (READ-ONLY)")
            
            # Validate the connected SHM
            if not self._validate_control_shm():
                logger.warning(f"Control SHM validation failed for: {shm_name}")
                self.control_shared_memory[shm_name].close()
                self.control_shared_memory[shm_name] = None
                return False
                
            logger.info(f"Successfully connected to control source: {shm_name} (READ-ONLY)")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Control shared memory not found: {shm_name}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to control shared memory {shm_name}: {e}")
            return False
            
    def _validate_control_shm(self) -> bool:
        """
        Validate the connected control shared memory.
        
        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            from utils.shm_utils import unpack_device_header, unpack_buffer_header
            
            for shm_name in self.control_shm_names:
                # Read and validate device header
                device_header_data = self.control_shared_memory[shm_name].buf[:DEVICE_HEADER_SIZE]
                device_info = unpack_device_header(device_header_data)

                # Check control shape
                shm_shape = device_info['shape']
                if shm_shape != self.control_shape:
                    logger.error(f"Control shape mismatch: expected {self.control_shape}, got {shm_shape}")
                    return False

                # Check control dtype
                shm_dtype_str = device_info['data_dtype']  # dtype is at index 3
                if shm_dtype_str != self.control_dtype:
                    logger.error(f"Control dtype mismatch: expected {self.control_dtype}, got {shm_dtype_str}")
                    return False

                # Read and validate buffer header
                buffer_header_data = self.control_shared_memory[shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE]
                buffer_info = unpack_buffer_header(buffer_header_data)

                # Check buffer size
                shm_buffer_size = buffer_info['buffer_size']
                if shm_buffer_size != self.control_buffer_size:
                    logger.error(f"Control buffer size mismatch: expected {self.control_buffer_size}, got {shm_buffer_size}")
                    return False

                # Recalculate frame offsets based on connected SHM
                shm_frame_size = device_info['frame_size']  # frame_size is at index 5
                self.control_frame_size = shm_frame_size
                self.control_frame_data_size = shm_frame_size - FRAME_HEADER_SIZE

                self.control_frame_offsets = [
                    DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * self.control_frame_size
                    for i in range(self.control_buffer_size)
                ]

                logger.info(f"Control SHM validation successful")
                logger.info(f"  - Shape: {shm_shape}")
                logger.info(f"  - Dtype: {shm_dtype_str}")
                logger.info(f"  - Buffer size: {shm_buffer_size}")
                logger.info(f"  - Current count: {buffer_info['current_frames_count']}")
                logger.info(f"  - Write index: {buffer_info['write_index']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating control SHM: {e}")
            return False
            
    def _get_control_buffer_info(self,shm_name: str) -> Optional[tuple]:
        """
        Get current control buffer information from SHM.
        
        Returns:
            tuple: (current_frames_count, write_index) or (None, None) if error
        """
        try:
            from utils.shm_utils import unpack_buffer_header
            
            # Read buffer header from SHM
            buffer_header_data = self.control_shared_memory[shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE]
            buffer_info = unpack_buffer_header(buffer_header_data)
            
            current_frames_count = buffer_info['current_frames_count']
            write_index = buffer_info['write_index']
            
            return current_frames_count, write_index
            
        except Exception as e:
            logger.error(f"Error reading control buffer info: {e}")
            return None, None
        
    def _initialize_control_buffer_header(self) -> None:
        """Initialize the control buffer header with metadata."""
        # Pack device header for control buffer
        device_header = pack_device_header(
            f"{self.device_name}_control", self.device_id, self.fps, 
            self.control_dtype, self.control_shape, self.control_frame_size
        )
        
        # Pack buffer header (initialize with zeros)
        buffer_header = pack_buffer_header(self.control_buffer_size, 0, 0)
        
        # Write headers to control SHM
        self.control_shared_memory[self.default_control_shm_name].buf[:DEVICE_HEADER_SIZE] = device_header
        self.control_shared_memory[self.default_control_shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE] = buffer_header
        
    def read_latest_control_command(self,shm_name: str) -> Optional[np.ndarray]:
        """
        Read the latest control command from the control buffer.
        
        Returns:
            tuple: (timestamp_ns, control_array) or (None, None) if no command available
        """
        if not self.control_shared_memory[shm_name]:
            return None, None
            
        # Get current buffer info from SHM
        current_frames_count, write_index = self._get_control_buffer_info(shm_name)
        if current_frames_count is None or current_frames_count == 0:
            return None, None
            
        try:
            # Calculate frame offset for latest command
            latest_index = (write_index - 1) % self.control_buffer_size
            frame_offset = self.control_frame_offsets[latest_index]
            
            # Read frame header
            frame_header = pack_frame_header(0)  # Placeholder
            frame_header_data = self.control_shared_memory[shm_name].buf[frame_offset:frame_offset+FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(frame_header_data, byteorder='little')
            
            # Read control data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + self.control_frame_data_size
            numpy_control_dtype = get_dtype(self.control_dtype)
            # Extract control data: DataChannel.buf returns raw buffer, slicing works directly
            sliced_buf = self.control_shared_memory[shm_name].buf[data_start:data_end]
            control_array = np.frombuffer(sliced_buf, dtype=numpy_control_dtype)
            # Reshape if needed
            if len(self.control_shape) > 0:
                control_array = control_array.reshape(self.control_shape)
                
            return timestamp_ns, control_array
            
        except Exception as e:
            logger.error(f"Error reading control command: {e}")
            return None, None
            
    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state. To be implemented by subclasses.
        
        Returns:
            numpy.ndarray: Current robot state
        """
        raise NotImplementedError("Subclasses must implement get_robot_state()")
        
    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command. To be implemented by subclasses.
        
        Args:
            control_array: Control command to execute
        """
        raise NotImplementedError("Subclasses must implement execute_control_command()")
        
    def _cleanup_control_shared_memory(self) -> None:
        """Clean up control shared memory resources."""
        for shm_name in self.control_shm_names:
            if self.control_shared_memory[shm_name]:
                try:
                    self.control_shared_memory[shm_name].close()
                    self.control_shared_memory[shm_name].unlink()
                    logger.info(f"Cleaned up control shared memory: {shm_name}")
                except (FileNotFoundError, PermissionError):
                    pass  # Already cleaned up or no permission
                except Exception as e:
                    logger.error(f"Error cleaning up control shared memory: {e}")
                finally:
                    self.control_shared_memory[shm_name] = None
                
    def start_server(self) -> None:
        """Start the robot server and run control loop."""
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_id} is already running")
            return
            
        logger.info(f"Starting {self.device_name}_{self.device_id} server...")
        self.running = True
        
        # Create state shared memory
        self._create_shared_memory()  # State buffer
        if self.control_shm_names is not None:
            self.control_shared_memory = {shm_name: None for shm_name in self.control_shm_names}
            for shm_name in self.control_shm_names:
                self._connect_to_existing_control_shm(shm_name)
        else:
            logger.info(f'enter to continue...none')
            self._create_control_shared_memory()  # Control buffer
        
        if not self.shared_memory or not any(self.control_shared_memory.values()):
            self.running = False
            raise RuntimeError("Failed to create/connect to shared memories")
            
        logger.info(f"Robot {self.device_name}_{self.device_id} server started successfully")
        
        # Main control loop
        try:
            assert self.fps > self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running:
                start_time = time.time()
                
                control_dict = {}
                for shm_name in self.control_shm_names:
                    # Read latest control command
                    timestamp_ns, control_array = self.read_latest_control_command(shm_name)
                    control_dict[shm_name] = control_array
                
                if control_dict[shm_name] is not None:
                    if iteration % command_iteration == 0:
                    # Execute control command
                        # logger.info(f"start execute: {control_dict}")
                        self.execute_control_command(control_dict)
                iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())
                    
                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            self.stop_server()
            
    def stop_server(self) -> None:
        """Stop the robot server and clean up resources."""
        if not self.running:
            return
            
        self.running = False
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()
        logger.info(f"Stopped {self.device_name}_{self.device_id} server")
        
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class SimRobot(BaseRobot):
    """
    Simulated robot using PyBullet for physics simulation.
    Controls a robotic arm and provides joint state feedback.
    """
    
    def __init__(self, device_id=0, urdf_path=None, joint_indices=None,
                 data_shape=(7,), control_shape=(7,), fps=100.0, 
                 data_dtype=np.float64, control_dtype=np.float64,
                 buffer_size=100, control_buffer_size=50, enable_gui=False, 
                 control_shm_name=None,ctrl_type="joint", **kwargs):
        """
        Initialize the simulated robot.
        
        Args:
            device_id: Unique identifier for this robot instance
            urdf_path: Path to robot URDF file (if None, uses default KUKA iiwa)
            joint_indices: List of joint indices to control (if None, uses all joints)
            data_shape: Shape of robot state data (joint positions, velocities, etc.)
            control_shape: Shape of control command data (target joint positions)
            fps: Control frequency in Hz
            data_dtype: Data type for state data
            control_dtype: Data type for control data
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            enable_gui: Enable PyBullet GUI for visualization
            control_shm_name: Name of existing control SHM to connect to (if None, will create new one)
            enable_eef_control: Enable end-effector control
            **kwargs: Additional arguments passed to BaseRobot
        """
        # Initialize base robot
        super().__init__(device_id=device_id, data_shape=data_shape, 
                        control_shape=control_shape, fps=fps,
                        data_dtype=data_dtype, control_dtype=control_dtype,
                        buffer_size=buffer_size, control_buffer_size=control_buffer_size, 
                        control_shm_name=control_shm_name, **kwargs)
        
        # Robot-specific attributes
        self.device_name = "SimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui
        
        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint","eef_delta","eef_abs"]
        
        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"
        
        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_id}_control"
        
        # PyBullet simulation
        self.physics_client = None
        self.robot_id = None
        self.num_joints = 0
        self.joint_names = []
        self.joint_limits = []

        # Initialize PyBullet
        self._initialize_pybullet()
        
    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation."""
        try:
            import pybullet as p
            import pybullet_data
            
            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Set physics parameters
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / self.fps)
            
            # Load robot URDF
            if self.urdf_path is None:
                # Use default KUKA iiwa robot
                self.urdf_path = "kuka_iiwa/model.urdf"
                
            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0])
            logger.info(f"Loaded robot URDF: {self.urdf_path}")
            
            # Get joint information
            self.num_joints = p.getNumJoints(self.robot_id)
            logger.info(f"Robot has {self.num_joints} joints")
            
            # Get joint names and limits
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                self.joint_names.append(joint_info[1].decode('utf-8'))
                
                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
                
            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                self.joint_indices = []
                for i in range(self.num_joints):
                    joint_info = p.getJointInfo(self.robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.joint_indices.append(i)
                        
            logger.info(f"Controlling joints: {self.joint_indices}")
            logger.info(f"Joint names: {[self.joint_names[i] for i in self.joint_indices]}")
            
            # Set initial joint positions to zero
            initial_positions = [0.0] * len(self.joint_indices)
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)
            

            
            logger.info("PyBullet simulation initialized successfully")
            link_state = p.getLinkState(self.robot_id, 6, computeForwardKinematics=True)
            self.init_pose = np.concatenate([np.array(link_state[4]), np.array(link_state[5])]) # in world frame

        except Exception as e:
            logger.error(f"Error initializing PyBullet: {e}")
            raise
            
    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities).
        
        Returns:
            numpy.ndarray: Current robot state [joint_positions, joint_velocities]
        """
        try:
            import pybullet as p
            
            if self.robot_id is None:
                return None
                
            # Get joint states
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            
            # Extract positions and velocities
            positions = [state[0] for state in joint_states]
            velocities = [state[1] for state in joint_states]
            
            # Combine into state array
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(positions + velocities, dtype=numpy_data_dtype)
            
            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)
                
            return state_array
            
        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return None
            
    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions).
        
        Args:
            control_array: Target joint positions
        """
        try:
            import pybullet as p
            
            if self.robot_id is None:
                return

            assert len(list(control_dict.keys())) == 1, "Simulated robot only supports control command from a single teleoperator device"
            control_array = list(control_dict.values())[0]
            if control_array is None:
                p.stepSimulation()
                return
            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]
                
            # If end-effector control is enabled, use inverse kinematics
            if self.ctrl_type == "eef_abs":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=True)
            elif self.ctrl_type == "eef_delta":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=False)
            else:
                target_positions = control_data
                
            # Limit the number of positions to the number of controlled joints
            num_controlled_joints = len(self.joint_indices)
            if len(target_positions) > num_controlled_joints:
                target_positions = target_positions[:num_controlled_joints]
            elif len(target_positions) < num_controlled_joints:
                # Pad with current positions if not enough targets provided
                current_states = p.getJointStates(self.robot_id, self.joint_indices)
                current_positions = [state[0] for state in current_states]
                target_positions.extend(current_positions[len(target_positions):])
                
            # Apply joint limits
            for i, (joint_idx, target_pos) in enumerate(zip(self.joint_indices, target_positions)):
                lower_limit, upper_limit = self.joint_limits[joint_idx]
                target_positions[i] = np.clip(target_pos, lower_limit, upper_limit)
                
            # Reset joint states
            for joint_idx, target_pos in zip(self.joint_indices, target_positions):
                p.resetJointState(self.robot_id, joint_idx, target_pos, 0.0)
            
            # Step simulation
            p.stepSimulation()
            
        except Exception as e:
            logger.error(f"Error executing control command: {e}")

    def _compute_ik_iterative_single_arm(self, control_data,abs=True):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics for single arm.

        Args:
            control_data: End-effector pose [pos, orn] where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)

        Returns:
            target_positions: Joint positions for the arm
        """
        try:
            import pybullet as p

            # Assume control_data contains [pos(3), orn(4)] = 7 elements minimum
            if len(control_data) < 7:  # 3+4 = 7 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            # Extract target pose
            target_pos = control_data[0:3]
            target_orn = control_data[3:7]
            target_pos,target_orn = compute_deviation([0,0,0],[0,0,0,0],target_pos, target_orn) if abs else compute_deviation(self.init_pose[0:3],self.init_pose[3:7],target_pos, target_orn)

            # Get end-effector link index (assume last link)
            end_effector_link = self.num_joints - 1

            # Iterative IK parameters - relaxed tolerances for better convergence
            max_iterations = 20  # Reduced outer iteration count
            position_tolerance = 1e-3  # Relaxed position tolerance to 1mm
            orientation_tolerance = 1e-2  # Relaxed orientation tolerance

            # Get current joint positions as initial guess
            current_joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            q0 = [state[0] for state in current_joint_states]
            
            # Iterative IK with PyBullet
            for iteration in range(max_iterations):
                # Calculate IK with PyBullet
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    end_effector_link,
                    target_pos,
                    target_orn,
                    lowerLimits=[self.joint_limits[i][0] for i in self.joint_indices],
                    upperLimits=[self.joint_limits[i][1] for i in self.joint_indices],
                    jointRanges=[self.joint_limits[i][1] - self.joint_limits[i][0] for i in self.joint_indices],
                    restPoses=q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )
                
                # Convert to list and apply joint limits
                joint_positions = list(joint_positions[:len(self.joint_indices)])
                for i, joint_idx in enumerate(self.joint_indices):
                    if joint_idx < len(self.joint_limits):
                        lower_limit, upper_limit = self.joint_limits[joint_idx]
                        joint_positions[i] = np.clip(joint_positions[i], lower_limit, upper_limit)
                
                # Check error
                link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                current_pos = link_state[4]
                current_orn = link_state[5]
                
                pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                dot_product = np.clip(np.dot(current_orn, target_orn), -1.0, 1.0)
                orn_error = 2 * np.sqrt(max(0.0, 1 - dot_product ** 2))
                
                # Check convergence - using relaxed tolerances
                if pos_error < position_tolerance:  # Focus mainly on position convergence
                    break

            # Print final orientation error
            if joint_positions is not None:
                # Calculate final error after IK iterations
                final_link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                final_current_pos = final_link_state[4]
                final_current_orn = final_link_state[5]
                
                final_pos_error = np.linalg.norm(np.array(final_current_pos) - np.array(target_pos))
                final_dot_product = np.clip(np.dot(final_current_orn, target_orn), -1.0, 1.0)
                final_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_dot_product ** 2))
                final_orn_error_deg = np.degrees(final_orn_error_rad)
                
                # Take only the controlled joints
                controlled_positions = joint_positions[:len(self.joint_indices)]
                return list(controlled_positions)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data

        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data

    def stop_server(self) -> None:
        """Stop the robot server and clean up PyBullet resources."""
        if self.physics_client is not None:
            import pybullet as p
            p.disconnect(self.physics_client)
            self.physics_client = None
            self.robot_id = None
            logger.info("PyBullet simulation disconnected")
            
        super().stop_server()
        
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class DualArmSimRobot(BaseRobot):
    """
    Dual-arm simulated robot using PyBullet for physics simulation.
    Controls two robotic arms and provides joint state feedback for both arms.
    """
    
    def __init__(self, device_id=0, urdf_path=None, joint_indices=None,
                 data_shape=(14,), control_shape=(14,), fps=100.0, 
                 data_dtype=np.float64, control_dtype=np.float64,
                 buffer_size=100, control_buffer_size=50, enable_gui=False, 
                 control_shm_name=None, left_arm_position=[0.2, 0, 1.2], 
                 right_arm_position=[-0.2, 0, 1.2], ctrl_type = "joint",
                 **kwargs):
        """
        Initialize the dual-arm simulated robot.
        
        Args:
            device_id: Unique identifier for this robot instance
            urdf_path: Path to robot URDF file (if None, uses default Flexiv Rizon4)
            joint_indices: List of joint indices to control (if None, uses all joints)
            data_shape: Shape of robot state data (joint positions, velocities for both arms)
            control_shape: Shape of control command data (target joint positions for both arms)
            fps: Control frequency in Hz
            data_dtype: Data type for state data
            control_dtype: Data type for control data
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            enable_gui: Enable PyBullet GUI for visualization
            control_shm_name: Name of existing control SHM to connect to (if None, will create new one)
            left_arm_position: Position of left arm base [x, y, z]
            right_arm_position: Position of right arm base [x, y, z]
            **kwargs: Additional arguments passed to BaseRobot
        """
        # Initialize base robot
        super().__init__(device_id=device_id, data_shape=data_shape, 
                        control_shape=control_shape, fps=fps,
                        data_dtype=data_dtype, control_dtype=control_dtype,
                        buffer_size=buffer_size, control_buffer_size=control_buffer_size, 
                        control_shm_name=control_shm_name, **kwargs)
        
        # Robot-specific attributes
        self.device_name = "DualArmSimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui
        self.left_arm_position = left_arm_position
        self.right_arm_position = right_arm_position
        
        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint", "eef_delta", "eef_abs"]
        
        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"
        
        # RTB robot models removed - using PyBullet IK only
        
        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_id}_control"
        
        # PyBullet simulation
        self.physics_client = None
        self.left_robot_id = None
        self.right_robot_id = None
        self.left_num_joints = 0
        self.right_num_joints = 0
        self.left_joint_names = []
        self.right_joint_names = []
        self.left_joint_limits = []
        self.right_joint_limits = []
        self.left_joint_indices = []
        self.right_joint_indices = []

        # Initialize PyBullet
        self._initialize_pybullet()
        # time.sleep(3)
        
    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation for dual arms."""
        try:
            import pybullet as p
            import pybullet_data
            
            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for dual-arm visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Set physics parameters
            p.setGravity(0, 0, 0)
            p.setTimeStep(1.0 / self.fps)
            
            # Load robot URDF for both arms
            if self.urdf_path is None:
                # Use default Flexiv Rizon4 robot
                self.urdf_path = "./assests/robot/rizon/flexiv_Rizon4_kinematics.urdf"
                
            # Load left arm
            self.left_orientation = p.getQuaternionFromEuler([np.pi/2, -np.pi/2, 0])
            self.left_robot_id = p.loadURDF(self.urdf_path, self.left_arm_position, self.left_orientation)
            logger.info(f"Loaded left arm URDF: {self.urdf_path} at {self.left_arm_position}")
            
            # Load right arm
            self.right_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
            self.right_robot_id = p.loadURDF(self.urdf_path, self.right_arm_position, self.right_orientation)
            logger.info(f"Loaded right arm URDF: {self.urdf_path} at {self.right_arm_position}")
            
            # Get joint information for left arm
            self.left_num_joints = p.getNumJoints(self.left_robot_id)
            logger.info(f"Left arm has {self.left_num_joints} joints")
            
            # Get joint information for right arm
            self.right_num_joints = p.getNumJoints(self.right_robot_id)
            logger.info(f"Right arm has {self.right_num_joints} joints")
            
            # Get joint names and limits for left arm
            for i in range(self.left_num_joints):
                joint_info = p.getJointInfo(self.left_robot_id, i)
                self.left_joint_names.append(joint_info[1].decode('utf-8'))
                
                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.left_joint_limits.append((lower_limit, upper_limit))
                
            # Get joint names and limits for right arm
            for i in range(self.right_num_joints):
                joint_info = p.getJointInfo(self.right_robot_id, i)
                self.right_joint_names.append(joint_info[1].decode('utf-8'))
                
                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.right_joint_limits.append((lower_limit, upper_limit))
                
            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                # Left arm joint indices
                self.left_joint_indices = []
                for i in range(self.left_num_joints):
                    joint_info = p.getJointInfo(self.left_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.left_joint_indices.append(i)
                        
                # Right arm joint indices
                self.right_joint_indices = []
                for i in range(self.right_num_joints):
                    joint_info = p.getJointInfo(self.right_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.right_joint_indices.append(i)
            else:
                # Use provided joint indices (assuming they're split between arms)
                mid_point = len(self.joint_indices) // 2
                self.left_joint_indices = self.joint_indices[:mid_point]
                self.right_joint_indices = self.joint_indices[mid_point:]
                        
            logger.info(f"Controlling left arm joints: {self.left_joint_indices}")
            logger.info(f"Left arm joint names: {[self.left_joint_names[i] for i in self.left_joint_indices]}")
            logger.info(f"Controlling right arm joints: {self.right_joint_indices}")
            logger.info(f"Right arm joint names: {[self.right_joint_names[i] for i in self.right_joint_indices]}")
            
            # Set initial joint positions to zero for both arms
            left_initial_positions = [1] * len(self.left_joint_indices)
            right_initial_positions = [1] * len(self.right_joint_indices)
            
            for i, joint_idx in enumerate(self.left_joint_indices):
                p.resetJointState(self.left_robot_id, joint_idx, left_initial_positions[i], 0.0)
            
            for i, joint_idx in enumerate(self.right_joint_indices):
                p.resetJointState(self.right_robot_id, joint_idx,  right_initial_positions[i], 0.0)

            start_time = time.time()

            while time.time() < start_time+1:
                left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_left = np.concatenate(
                    [np.array(left_link_state[4]), np.array(left_link_state[5])])  # in world frame

                right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_right = np.concatenate(
                    [np.array(right_link_state[4]), np.array(right_link_state[5])])  # in world frame

            left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_left = np.concatenate([np.array(left_link_state[4]), np.array(left_link_state[5])]) # in world frame

            right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_right = np.concatenate([np.array(right_link_state[4]), np.array(right_link_state[5])]) # in world frame

            logger.info("Dual-arm PyBullet simulation initialized successfully")

            
        except Exception as e:
            logger.error(f"Error initializing dual-arm PyBullet: {e}")
            raise
            
    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities for both arms).
        
        Returns:
            numpy.ndarray: Current robot state [left_joint_positions, left_joint_velocities, 
                          right_joint_positions, right_joint_velocities]
        """
        try:
            import pybullet as p
            
            if self.left_robot_id is None or self.right_robot_id is None:
                return None
                
            # Get joint states for left arm
            left_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)
            
            # Get joint states for right arm
            right_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)
            
            # Extract positions and velocities for left arm
            left_positions = [state[0] for state in left_joint_states]
            left_velocities = [state[1] for state in left_joint_states]
            
            # Extract positions and velocities for right arm
            right_positions = [state[0] for state in right_joint_states]
            right_velocities = [state[1] for state in right_joint_states]
            
            # Combine into state array: [left_pos, left_vel, right_pos, right_vel]
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(left_positions + left_velocities + right_positions + right_velocities, 
                                 dtype=numpy_data_dtype)
            
            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)
                
            return state_array
            
        except Exception as e:
            logger.error(f"Error getting dual-arm robot state: {e}")
            return None
            
    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions for both arms).
        
        Args:
            control_array: Target joint positions for both arms
        """
        try:
            import pybullet as p
            
            if self.left_robot_id is None or self.right_robot_id is None:
                return
                
            assert len(list(control_dict.keys())) == 1, "Simulated robot only supports control command from a single teleoperator device"
            control_array = list(control_dict.values())[0]

            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]

            # If end-effector control is enabled, use inverse kinematics
            if self.ctrl_type == "eef_abs":
                target_positions = self._compute_ik_iterative(control_data, abs=True)
            elif self.ctrl_type == "eef_delta":
                target_positions = self._compute_ik_iterative(control_data, abs=False)
            else:
                target_positions = control_data
                
            # Split into left and right arm targets
            left_joint_count = len(self.left_joint_indices)
            left_target_positions = target_positions[:left_joint_count]
            right_target_positions = target_positions[left_joint_count:]
            
            # Reset joint states for left arm
            for joint_idx, target_pos in zip(self.left_joint_indices, left_target_positions):
                p.resetJointState(self.left_robot_id, joint_idx, target_pos, 0.0)

            # Reset joint states for right arm
            for joint_idx, target_pos in zip(self.right_joint_indices, right_target_positions):
                p.resetJointState(self.right_robot_id, joint_idx, target_pos, 0.0)
            
            # Step simulation
            p.stepSimulation()
            
        except Exception as e:
            logger.error(f"Error executing dual-arm control command: {e}")

    def _compute_ik_iterative(self, control_data,abs):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics.
        
        Args:
            control_data: End-effector poses for both arms [left_pos, left_orn, right_pos, right_orn]
                         where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)
        
        Returns:
            target_positions: Joint positions for both arms
        """
        try:
            import pybullet as p
            # Assume control_data contains [left_pos(3), left_orn(4), right_pos(3), right_orn(4)]
            if len(control_data) < 14:  # 3+4+3+4 = 14 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            left_target_pos = control_data[0:3]
            left_target_orn = control_data[3:7]
            left_target_pos, left_target_orn = compute_deviation(self.left_arm_position, self.left_orientation,
                                                                 left_target_pos, left_target_orn)  if abs else compute_deviation(self.init_pose_left[0:3],self.init_pose_left[3:7],left_target_pos, left_target_orn)

            right_target_pos = control_data[7:10]
            right_target_orn = control_data[10:14]
            right_target_pos, right_target_orn = compute_deviation(self.right_arm_position, self.right_orientation,
                                                                 right_target_pos, right_target_orn) if abs else compute_deviation(self.init_pose_right[0:3],self.init_pose_right[3:7],right_target_pos, right_target_orn)
            left_end_effector_link = 6
            right_end_effector_link = 6

            # Iterative IK parameters
            max_iterations = 20
            position_tolerance = 1e-3
            orientation_tolerance = 1e-2
            
            # Get current joint positions as initial guess
            left_current_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)
            left_q0 = [state[0] for state in left_current_joint_states]
            right_current_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)
            right_q0 = [state[0] for state in right_current_joint_states]
            
            # Compute IK for left arm with iteration
            for iteration in range(max_iterations):
                left_joint_positions = p.calculateInverseKinematics(
                    self.left_robot_id,
                    left_end_effector_link,
                    left_target_pos,
                    left_target_orn,
                    lowerLimits=[self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    upperLimits=[self.left_joint_limits[i][1] for i in self.left_joint_indices],
                    jointRanges=[self.left_joint_limits[i][1] - self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    restPoses=left_q0,  # Use current state as rest pose
                    maxNumIterations=10,  # Increase internal iterations per step
                    residualThreshold=1e-5  # Relaxed residual threshold
                )
                
                # Convert to list and apply joint limits
                left_joint_positions = list(left_joint_positions[:len(self.left_joint_indices)])
                for i, joint_idx in enumerate(self.left_joint_indices):
                    if joint_idx < len(self.left_joint_limits):
                        lower_limit, upper_limit = self.left_joint_limits[joint_idx]
                        left_joint_positions[i] = np.clip(left_joint_positions[i], lower_limit, upper_limit)
                
                # Check error
                left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                left_current_pos = left_link_state[4]
                left_current_orn = left_link_state[5]
                
                left_pos_error = np.linalg.norm(np.array(left_current_pos) - np.array(left_target_pos))
                left_dot_product = np.clip(np.dot(left_current_orn, left_target_orn), -1.0, 1.0)
                left_orn_error = 2 * np.sqrt(max(0.0, 1 - left_dot_product ** 2))
                
                # Update initial guess
                left_q0 = left_joint_positions[:len(self.left_joint_indices)]
                
                # Check convergence
                if left_pos_error < position_tolerance:
                    break
            
            # Compute IK for right arm with iteration
            for iteration in range(max_iterations):
                right_joint_positions = p.calculateInverseKinematics(
                    self.right_robot_id,
                    right_end_effector_link,
                    right_target_pos,
                    right_target_orn,
                    lowerLimits=[self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    upperLimits=[self.right_joint_limits[i][1] for i in self.right_joint_indices],
                    jointRanges=[self.right_joint_limits[i][1] - self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    restPoses=right_q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )
                
                # Convert to list and apply joint limits
                right_joint_positions = list(right_joint_positions[:len(self.right_joint_indices)])
                for i, joint_idx in enumerate(self.right_joint_indices):
                    if joint_idx < len(self.right_joint_limits):
                        lower_limit, upper_limit = self.right_joint_limits[joint_idx]
                        right_joint_positions[i] = np.clip(right_joint_positions[i], lower_limit, upper_limit)
                
                # Check error
                right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                right_current_pos = right_link_state[4]
                right_current_orn = right_link_state[5]
                right_pos_error = np.linalg.norm(np.array(right_current_pos) - np.array(right_target_pos))
                right_dot_product = np.clip(np.dot(right_current_orn, right_target_orn), -1.0, 1.0)
                right_orn_error = 2 * np.sqrt(max(0.0, 1 - right_dot_product ** 2))
                
                # Update initial guess
                right_q0 = right_joint_positions[:len(self.right_joint_indices)]
                
                # Check convergence
                if right_pos_error < position_tolerance:
                    logger.debug(f"Right arm IK converged at iteration {iteration+1}, pos_error: {right_pos_error:.6f}")
                    break
            
            # Print final orientation errors for both arms
            if left_joint_positions is not None and right_joint_positions is not None:
                # Calculate final errors for left arm
                final_left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                final_left_current_orn = final_left_link_state[5]
                final_left_dot_product = np.clip(np.dot(final_left_current_orn, left_target_orn), -1.0, 1.0)
                final_left_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_left_dot_product ** 2))
                final_left_orn_error_deg = np.degrees(final_left_orn_error_rad)
                
                # Calculate final errors for right arm
                final_right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                final_right_current_orn = final_right_link_state[5]
                final_right_dot_product = np.clip(np.dot(final_right_current_orn, right_target_orn), -1.0, 1.0)
                final_right_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_right_dot_product ** 2))
                final_right_orn_error_deg = np.degrees(final_right_orn_error_rad)
                # Take only the controlled joints
                left_controlled = left_joint_positions[:len(self.left_joint_indices)]
                right_controlled = right_joint_positions[:len(self.right_joint_indices)]
                return list(left_controlled) + list(right_controlled)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data
                
        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data
    

            
    def stop_server(self) -> None:
        """Stop the dual-arm robot server and clean up resources."""
        self.running = False
        
        # Clean up PyBullet
        if self.physics_client is not None:
            try:
                import pybullet as p
                p.disconnect()
                logger.info("Disconnected from PyBullet")
            except:
                pass
            self.physics_client = None
            
        # Clean up shared memory
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()
        
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.stop_server()


class RizonRobot(BaseRobot):
    """
    Rizon Robot Device - Flexiv Rizon robot interface for SuperInference.

    Inherits from BaseRobot and provides real-time robot state monitoring and joint position control.
    """

    def __init__(self, device_id=0, robot_sn="Rizon4s-123456", data_shape=(14,),
                 control_shape=(7,), fps=100.0, data_dtype=np.float64, control_dtype=np.float64,
                 buffer_size=100, control_buffer_size=50, hardware_latency_ms=10.0,
                 control_shm_name=None, control_type="joint", home=None, max_velocity=None,tool_name="",
                 max_acceleration=None, mirror_mode=False, utilize_gripper=False,
                 gripper_name="", delta_eef_motion_scaler=None, impedance_parameter=None, using_tcp_pose=False,
                 max_linear_vel=None, max_linear_acc=None, max_angular_vel=None, max_angular_acc=None, **kwargs):
        """
        Initialize Rizon Robot Device.

        Args:
            device_id: Unique device identifier
            robot_sn: Robot serial number (e.g., "Rizon4s-123456") or list for dual arm
            data_shape: Shape of robot state data (22 for single arm, 44 for dual arm)
            control_shape: Shape of control command data (7 for single arm, 14 for dual arm, +1 per arm if gripper enabled)
            fps: Update frequency in Hz
            data_dtype: Data type for robot states
            control_dtype: Data type for control commands
            buffer_size: Size of state data buffer
            control_buffer_size: Size of control command buffer
            hardware_latency_ms: Hardware latency compensation
            control_shm_name: Name of shared memory for control commands
            control_type: Control type - "joint", "eef_abs", or "eef_delta"
            home: Home configuration dict with 'joints' and/or 'pose' keys
            max_velocity: Maximum joint velocities [rad/s]
            max_acceleration: Maximum joint accelerations [rad/s²]
            mirror_mode: Mirror mode for dual arm robots
            utilize_gripper: Enable gripper control (requires control_shape: [8] single, [16] dual)
            gripper_name: Name of the gripper to control (e.g., "GRAV-G6-10")
            delta_eef_motion_scaler: Dict with 'trans' and 'rot' scaling factors for eef_delta mode
            impedance_parameter: Cartesian impedance parameters [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z] for EEF control mode
            max_linear_vel: Maximum linear velocity for EEF control [m/s]
            max_linear_acc: Maximum linear acceleration for EEF control [m/s²]
            max_angular_vel: Maximum angular velocity for EEF control [rad/s]
            max_angular_acc: Maximum angular acceleration for EEF control [rad/s²]
            **kwargs: Additional arguments passed to BaseRobot
        """
        super().__init__(device_id=device_id, data_shape=data_shape, control_shape=control_shape,
                        fps=fps, data_dtype=data_dtype, control_dtype=control_dtype,
                        buffer_size=buffer_size, control_buffer_size=control_buffer_size,
                        hardware_latency_ms=hardware_latency_ms, control_shm_name=control_shm_name, **kwargs)

        # Robot configuration
        self.robot_sn = robot_sn
        self.control_type = control_type
        assert self.control_type in ["joint", "eef_abs", "eef_delta", "eef_del"], f"Invalid control_type: {control_type}"
        self.default_control_shm_name = f"{self.device_name}_{self.device_id}_control"
        self.mirror_mode = mirror_mode
        
        # Gripper configuration
        self.utilize_gripper = utilize_gripper
        self.gripper_name = gripper_name
        self.tool_name = tool_name if tool_name is not None else self.gripper_name
        self.last_gripper_width = [0.1,0.1]
        self.using_tcp_pose = using_tcp_pose
        # Delta EEF motion scaler configuration
        self.delta_eef_motion_scaler = delta_eef_motion_scaler
        if self.delta_eef_motion_scaler is not None:
            logger.info(f"Delta EEF motion scaler configured: trans={self.delta_eef_motion_scaler.get('trans', 1.0)}, rot={self.delta_eef_motion_scaler.get('rot', 1.0)}")
            
        # Cartesian impedance configuration
        self.impedance_parameter = impedance_parameter
        if self.control_type in ["eef_abs", "eef_delta", "eef_del"]:
            if self.impedance_parameter is not None:
                logger.info(f"Cartesian impedance configured: {self.impedance_parameter}")
            else:
                # Default values: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
                self.impedance_parameter = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
                logger.info(f"Using default cartesian : {self.impedance_parameter}")
        else:
            # Joint control mode - impedance parameters not used
            self.impedance_parameter = None
            logger.info("Joint control mode - Impedance parameters not applicable")
            
        # Motion limits configuration for EEF control
        if self.control_type in ["eef_abs", "eef_delta", "eef_del"]:
            # Linear motion limits
            self.max_linear_vel = max_linear_vel if max_linear_vel is not None else 1.0
            self.max_linear_acc = max_linear_acc if max_linear_acc is not None else 2.0
            
            # Angular motion limits
            self.max_angular_vel = max_angular_vel if max_angular_vel is not None else 2.0
            self.max_angular_acc = max_angular_acc if max_angular_acc is not None else 5.0
            
            logger.info(f"Motion limits configured - Linear: vel={self.max_linear_vel} m/s, acc={self.max_linear_acc} m/s²")
            logger.info(f"Motion limits configured - Angular: vel={self.max_angular_vel} rad/s, acc={self.max_angular_acc} rad/s²")
        else:
            # Joint control mode - motion limits not used
            self.max_linear_vel = None
            self.max_linear_acc = None
            self.max_angular_vel = None
            self.max_angular_acc = None
            logger.info("Joint control mode - Motion limits not applicable")
            
        # Determine if this is dual arm based on robot_sn and control_shape
        self.is_dual_arm = isinstance(robot_sn, list) and control_shape[0] >= 14

        # Validate control_shape based on gripper usage
        if self.utilize_gripper:
            if self.is_dual_arm:
                assert control_shape[0] == 16, f"Dual arm with gripper requires control_shape[0] == 16, got {control_shape[0]}"
            else:
                assert control_shape[0] == 8, f"Single arm with gripper requires control_shape[0] == 8, got {control_shape[0]}"
            assert self.gripper_name, "gripper_name must be specified when utilize_gripper=True"
        else:
            if self.is_dual_arm:
                assert control_shape[0] == 14, f"Dual arm without gripper requires control_shape[0] == 14, got {control_shape[0]}"
            else:
                assert control_shape[0] == 7, f"Single arm without gripper requires control_shape[0] == 7, got {control_shape[0]}"

        if self.mirror_mode:
            assert self.is_dual_arm, "Mirror mode requires dual arm robots"
            assert self.control_type == "eef_delta", "Mirror mode requires eef_delta control type"
        
        # Robot interfaces
        if self.is_dual_arm:
            self.robot_left = None
            self.robot_right = None
            self.left_sn = robot_sn[0] if isinstance(robot_sn, list) else robot_sn
            self.right_sn = robot_sn[1] if isinstance(robot_sn, list) and len(robot_sn) > 1 else robot_sn
            logger.info(f"Initializing dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")
        else:
            self.robot = None
            logger.info(f"Initializing single-arm Rizon robot: {robot_sn}")
            
        # Gripper interfaces
        self.gripper = None
        self.gripper_left = None
        self.gripper_right = None
        self.tool = None
        self.tool_left = None
        self.tool_right = None

        # Set default velocity and acceleration limits
        num_joints = 14 if self.is_dual_arm else 7
        if max_velocity is None:
            self.max_velocity = [1.0] * num_joints  # Conservative velocity limits [rad/s]
        else:
            self.max_velocity = max_velocity

        if max_acceleration is None:
            self.max_acceleration = [2.0] * num_joints  # Conservative acceleration limits [rad/s²]
        else:
            self.max_acceleration = max_acceleration

        # Robot status
        self.is_operational = False
        self.is_enabled = False
        self.current_mode = None
        
        # Parse home configuration
        self.home_joints = None
        self.home_joints_left = None
        self.home_joints_right = None
        self.home_presets = None
        self.home_preset_index = None
        self.home_pose = None
        self.home_pose_left = None
        self.home_pose_right = None
        self.wb_R_left = R.from_euler('z', np.pi) if not self.mirror_mode else R.from_euler('z', np.pi/2)
        self.wb_R_right = R.from_euler('z', np.pi/2) if not self.mirror_mode else R.from_euler('z', np.pi)

        # Process home configuration
        if home is not None and isinstance(home, dict):
            self._configure_home(home)

        # Import flexivrdk
        try:
            import flexivrdk
            self.flexivrdk = flexivrdk
        except ImportError as e:
            logger.error(f"Required dependency not available: {e}")
            logger.error("Install with: pip install flexivrdk")
            raise

    def _configure_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure home configuration from YAML dict."""
        try:
            if self.is_dual_arm:
                self._configure_dual_arm_home(home_cfg)
            else:
                self._configure_single_arm_home(home_cfg)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(f"Failed to configure home parameters: {exc}")
            raise

    def _configure_dual_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure dual-arm home joints and poses."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets for dual-arm configuration must be a non-empty list")
            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            joints_left = selected_preset.get("joints_left")
            joints_right = selected_preset.get("joints_right")
            pose_left = selected_preset.get("pose_left")
            pose_right = selected_preset.get("pose_right")

            if joints_left is not None:
                self.home_joints_left = joints_left
                logger.info(f"Configured home joints for left arm (preset {preset_index}): {self.home_joints_left}")
            if joints_right is not None:
                self.home_joints_right = joints_right
                logger.info(f"Configured home joints for right arm (preset {preset_index}): {self.home_joints_right}")
            if pose_left is not None:
                self.home_pose_left = pose_left
                logger.info(f"Configured home pose for left arm (preset {preset_index}): {self.home_pose_left}")
            if pose_right is not None:
                self.home_pose_right = pose_right
                logger.info(f"Configured home pose for right arm (preset {preset_index}): {self.home_pose_right}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints_left" in home_cfg:
                self.home_joints_left = home_cfg["joints_left"]
                logger.info(f"Configured home joints for left arm: {self.home_joints_left}")
            if "joints_right" in home_cfg:
                self.home_joints_right = home_cfg["joints_right"]
                logger.info(f"Configured home joints for right arm: {self.home_joints_right}")
            if "pose_left" in home_cfg:
                self.home_pose_left = home_cfg["pose_left"]
                logger.info(f"Configured home pose for left arm: {self.home_pose_left}")
            if "pose_right" in home_cfg:
                self.home_pose_right = home_cfg["pose_right"]
                logger.info(f"Configured home pose for right arm: {self.home_pose_right}")

    def _configure_single_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure single-arm home joints and pose."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets must be a non-empty list")

            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            if "joints" in selected_preset:
                self.home_joints = selected_preset["joints"]
                logger.info(f"Configured home joints (preset {preset_index}): {self.home_joints}")
            if "pose" in selected_preset:
                self.home_pose = selected_preset["pose"]
                logger.info(f"Configured home pose (preset {preset_index}): {self.home_pose}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints" in home_cfg:
                self.home_joints = home_cfg["joints"]
                logger.info(f"Configured home joints: {self.home_joints}")
            if "pose" in home_cfg:
                self.home_pose = home_cfg["pose"]
                logger.info(f"Configured home pose: {self.home_pose}")

    def _connect_robot(self) -> bool:
        """Connect to Rizon robot(s)."""
        try:
            if self.is_dual_arm:
                return self._connect_dual_arm_robots()
            else:
                return self._connect_single_robot()
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            return False

    def _connect_single_robot(self) -> bool:
        """Connect to single Rizon robot."""
        try:
            logger.info(f"Connecting to Rizon robot: {self.robot_sn}")

            # Create robot interface
            logger.info("Creating robot interface...")
            self.robot = self.flexivrdk.Robot(self.robot_sn)
            logger.info("Robot interface created successfully")

            # Check for faults and enable
            if not self._check_and_enable_robot(self.robot, "Robot"):
                return False

            self.is_enabled = True
            
            # Initialize gripper if enabled
            if self.utilize_gripper:
                if not self._initialize_gripper():
                    logger.error("Failed to initialize gripper")
                    return False
            
            self.is_operational = True
            logger.info("Robot connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to single robot: {e}")
            return False

    def _connect_dual_arm_robots(self) -> bool:
        """Connect to dual arm Rizon robots."""
        try:
            logger.info(f"Connecting to dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")

            # Create left robot interface
            logger.info("Creating left robot interface...")
            self.robot_left = self.flexivrdk.Robot(self.left_sn)
            logger.info("Left robot interface created successfully")

            # Create right robot interface
            logger.info("Creating right robot interface...")
            self.robot_right = self.flexivrdk.Robot(self.right_sn)
            logger.info("Right robot interface created successfully")

            # Check for faults and enable both robots
            if not self._check_and_enable_robot(self.robot_left, "Left robot"):
                return False
            if not self._check_and_enable_robot(self.robot_right, "Right robot"):
                return False

            self.is_enabled = True
            
            # Initialize grippers if enabled
            if self.utilize_gripper:
                if not self._initialize_dual_grippers():
                    logger.error("Failed to initialize grippers")
                    return False
            
            self.is_operational = True
            logger.info("Dual-arm robots connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to dual-arm robots: {e}")
            return False

    def _check_and_enable_robot(self, robot, name: str) -> bool:
        """Check for faults and enable a robot."""
        try:
            # Check for faults
            logger.info(f"Checking {name} status...")
            if robot.fault():
                logger.warning(f"{name} has faults, attempting to clear...")
                if not robot.ClearFault():
                    logger.error(f"Failed to clear {name} faults")
                    return False
                logger.info(f"{name} faults cleared successfully")
            else:
                logger.info(f"{name} status: No faults detected")

            # Enable robot
            logger.info(f"Enabling {name}...")
            robot.Enable()

            # Wait for operational status
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            logger.info(f"Waiting for {name} to become operational...")

            while not robot.operational():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"Timeout waiting for {name} to become operational after {timeout} seconds")
                    return False

                # Show progress every 5 seconds
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    remaining = timeout - elapsed
                    logger.info(f"Still waiting for {name}... {remaining:.1f}s remaining")

                time.sleep(1.0)

            logger.info(f"{name} is now operational")
            return True

        except Exception as e:
            logger.error(f"Failed to enable {name}: {e}")
            return False
    
    def _initialize_gripper(self) -> bool:
        """Initialize gripper for single arm robot."""
        try:
            logger.info(f"Initializing gripper [{self.gripper_name}]")
            
            # Create gripper interface
            self.gripper = self.flexivrdk.Gripper(self.robot)
            
            # Create tool interface
            self.tool = self.flexivrdk.Tool(self.robot)
            
            # Enable gripper
            logger.info(f"Enabling gripper [{self.gripper_name}]")
            self.gripper.Enable(self.gripper_name)
            
            # Print gripper parameters
            logger.info("Gripper parameters:")
            logger.info(f"  name: {self.gripper.params().name}")
            logger.info(f"  min_width: {round(self.gripper.params().min_width, 3)}")
            logger.info(f"  max_width: {round(self.gripper.params().max_width, 3)}")
            logger.info(f"  min_force: {round(self.gripper.params().min_force, 3)}")
            logger.info(f"  max_force: {round(self.gripper.params().max_force, 3)}")
            logger.info(f"  min_vel: {round(self.gripper.params().min_vel, 3)}")
            logger.info(f"  max_vel: {round(self.gripper.params().max_vel, 3)}")
            
            # Switch robot tool to gripper
            logger.info(f"Switching robot tool to [{self.tool_name}]")
            self.tool.Switch(self.tool_name)
            self.gripper.Move(0.085, 0.1, 100)
            logger.info("Gripper initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize gripper: {e}")
            return False

    def _initialize_dual_grippers(self) -> bool:
        """Initialize grippers for dual arm robots."""
        try:
            logger.info(f"Initializing grippers [{self.gripper_name}] for dual arms")
            
            # Create gripper interfaces
            self.gripper_left = self.flexivrdk.Gripper(self.robot_left)
            self.gripper_right = self.flexivrdk.Gripper(self.robot_right)
            
            # Create tool interfaces
            self.tool_left = self.flexivrdk.Tool(self.robot_left)
            self.tool_right = self.flexivrdk.Tool(self.robot_right)
            
            # Enable grippers
            logger.info(f"Enabling left gripper [{self.gripper_name}]")
            self.gripper_left.Enable(self.gripper_name)
            
            logger.info(f"Enabling right gripper [{self.gripper_name}]")
            self.gripper_right.Enable(self.gripper_name)
            
            # Print gripper parameters for left arm
            logger.info("Left gripper parameters:")
            logger.info(f"  name: {self.gripper_left.params().name}")
            logger.info(f"  min_width: {round(self.gripper_left.params().min_width, 3)}")
            logger.info(f"  max_width: {round(self.gripper_left.params().max_width, 3)}")
            
            # Switch robot tools to grippers
            logger.info(f"Switching left robot tool to [{self.gripper_name}]")
            self.tool_left.Switch(self.tool_name)
            
            logger.info(f"Switching right robot tool to [{self.gripper_name}]")
            self.tool_right.Switch(self.tool_name)

            self.gripper_left.Move(0.085, 0.1, 100)
            self.gripper_right.Move(0.085, 0.1, 100)

            logger.info("Dual grippers initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dual grippers: {e}")
            return False

    def _switch_to_joint_control(self) -> bool:
        """Switch robot(s) to joint position control mode."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to joint control
                self.robot_left.SwitchMode(self.flexivrdk.Mode.NRT_JOINT_POSITION)
                self.robot_right.SwitchMode(self.flexivrdk.Mode.NRT_JOINT_POSITION)
                logger.info("Switched dual-arm robots to joint position control mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to joint control
                self.robot.SwitchMode(self.flexivrdk.Mode.NRT_JOINT_POSITION)
                logger.info("Switched robot to joint position control mode")

            # Warning: Cartesian impedance parameters are not used in joint control mode
            logger.warning("Joint control mode active - Impedance parameters are not applied")

            self.current_mode = "NRT_JOINT_POSITION"
            return True

        except Exception as e:
            logger.error(f"Failed to switch to joint control mode: {e}")
            return False

    def _switch_to_eef_control(self) -> bool:
        """Switch robot(s) to end-effector control mode."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to Cartesian control
                self.robot_left.SwitchMode(self.flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                self.robot_right.SwitchMode(self.flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                self.robot_left.SetForceControlAxis([False, False, False, False, False, False])
                self.robot_right.SetForceControlAxis([False, False, False, False, False, False])
                self.robot_left.SetCartesianImpedance(self.impedance_parameter)
                self.robot_right.SetCartesianImpedance(self.impedance_parameter)
                logger.info("Switched dual-arm robots to Cartesian control mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to Cartesian control
                # print("before:", self.robot.mode())
                self.robot.SwitchMode(self.flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                # print("after:", self.robot.mode())
                self.robot.SetForceControlAxis([False, False, False, False, False, False])
                self.robot.SetCartesianImpedance(self.impedance_parameter)
                logger.info("Switched robot to Cartesian control mode")

            self.current_mode = "NRT_CARTESIAN_MOTION_FORCE"
            return True

        except Exception as e:
            logger.error(f"Failed to switch to end-effector control mode: {e}")
            return False

    def _set_home_pose(self) -> bool:
        """Set home pose from config or current pose if not configured."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                
                # If home poses not configured, use current poses
                if self.home_pose_left is None or self.home_pose_right is None:
                    logger.info("Home poses not configured, using current poses")
                    
                    # Check if home_joints are configured - if so, move there first and record poses
                    if self.home_joints_left is not None or self.home_joints_right is not None:
                        logger.info("Moving to configured home joint positions")
                        try:
                            # Ensure joint control mode before sending joint positions
                            return_to_eef = self.control_type in ["eef_abs", "eef_delta", "eef_del"]
                            if self.current_mode != "NRT_JOINT_POSITION":
                                if not self._switch_to_joint_control():
                                    raise RuntimeError("Failed to switch to joint position control before homing")
                            # Move left arm to home joint positions if configured
                            if self.home_joints_left is not None:
                                logger.info(f"Moving left arm to home joints: {self.home_joints_left}")
                                target_vel = [0.0] * 7
                                self.robot_left.SendJointPosition(
                                    self.home_joints_left,
                                    target_vel,
                                    target_vel,
                                    self.max_velocity[:7],
                                    self.max_acceleration[:7]
                                )
                            
                            # Move right arm to home joint positions if configured
                            if self.home_joints_right is not None:
                                logger.info(f"Moving right arm to home joints: {self.home_joints_right}")
                                self.robot_right.SendJointPosition(
                                    self.home_joints_right,
                                    target_vel,
                                    target_vel,
                                    self.max_velocity[7:14],
                                    self.max_acceleration[7:14]
                                )
                            
                            # Wait for motion to complete
                            time.sleep(3.0)
                            logger.info("Dual-arm robots moved to home joint positions")
                            
                            # Now get the poses at these joint configurations and set as home poses
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()
                            
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            if self.home_pose_left is None:
                                self.home_pose_left = [left_tcp[0], left_tcp[1], left_tcp[2], 
                                                     left_tcp[4], left_tcp[5], left_tcp[6], left_tcp[3]]
                                logger.info(f"Recorded home pose for left arm from home joints: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")
                            
                            if self.home_pose_right is None:
                                self.home_pose_right = [right_tcp[0], right_tcp[1], right_tcp[2], 
                                                      right_tcp[4], right_tcp[5], right_tcp[6], right_tcp[3]]
                                logger.info(f"Recorded home pose for right arm from home joints: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")

                            # Switch back to EEF control if needed
                            if return_to_eef and self.current_mode != "NRT_CARTESIAN_MOTION_FORCE":
                                if not self._switch_to_eef_control():
                                    logger.warning("Failed to switch back to EEF control after homing")
                            
                        except Exception as e:
                            logger.error(f"Failed to move to home joints: {e}")
                            # Fall back to using current poses
                            logger.info("Falling back to using current poses as home poses")
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()
                            
                            if self.home_pose_left is None:
                                self.home_pose_left = [left_tcp[0], left_tcp[1], left_tcp[2], 
                                                     left_tcp[4], left_tcp[5], left_tcp[6], left_tcp[3]]
                            if self.home_pose_right is None:
                                self.home_pose_right = [right_tcp[0], right_tcp[1], right_tcp[2], 
                                                      right_tcp[4], right_tcp[5], right_tcp[6], right_tcp[3]]
                    
                    # If still no home poses, get current poses
                    if self.home_pose_left is None or self.home_pose_right is None:
                        # Get current poses for both arms
                        left_states = self.robot_left.states()
                        right_states = self.robot_right.states()
                        
                        # Set home poses (TCP pose: [x, y, z, qw, qx, qy, qz] -> [x, y, z, qx, qy, qz, qw])
                        left_tcp = left_states.tcp_pose.copy()
                        right_tcp = right_states.tcp_pose.copy()
                        
                        if self.home_pose_left is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_left = [left_tcp[0], left_tcp[1], left_tcp[2], 
                                                 left_tcp[4], left_tcp[5], left_tcp[6], left_tcp[3]]
                        
                        if self.home_pose_right is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_right = [right_tcp[0], right_tcp[1], right_tcp[2], 
                                                  right_tcp[4], right_tcp[5], right_tcp[6], right_tcp[3]]
                
                logger.info(f"Using dual-arm home poses:")
                logger.info(f"  Left: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")
                logger.info(f"  Right: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")
            else:
                if not self.robot:
                    return False
                
                # Check if home_joints is configured - if so, move there first and record pose
                if self.home_joints is not None:
                    logger.info(f"Moving to configured home joints: {self.home_joints}")
                    try:
                        # Ensure joint control mode before sending joint positions
                        return_to_eef = self.control_type in ["eef_abs", "eef_delta", "eef_del"]
                        if self.current_mode != "NRT_JOINT_POSITION":
                            if not self._switch_to_joint_control():
                                raise RuntimeError("Failed to switch to joint position control before homing")

                        target_vel = [0.0] * 7
                        target_acc = [0.0] * 7
                        # Move to home joint positions
                        self.robot.SendJointPosition(
                            self.home_joints, target_vel, target_acc,
                            self.max_velocity, self.max_acceleration
                        )
                        # Wait for motion to complete
                        time.sleep(3.0)
                        logger.info("Robot moved to home joint positions")
                        
                        # Now get the pose at this joint configuration and set as home_pose
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()
                        
                        # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                        self.home_pose = [tcp[0], tcp[1], tcp[2], tcp[4], tcp[5], tcp[6], tcp[3]]
                        logger.info(f"Recorded home pose from home joints: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")

                        # Switch back to EEF control if needed
                        if return_to_eef and self.current_mode != "NRT_CARTESIAN_MOTION_FORCE":
                            if not self._switch_to_eef_control():
                                logger.warning("Failed to switch back to EEF control after homing")
                        
                    except Exception as e:
                        logger.error(f"Failed to move to home joints: {e}")
                        # Fall back to using current pose
                        logger.info("Falling back to using current pose as home pose")
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()
                        self.home_pose = [tcp[0], tcp[1], tcp[2], tcp[4], tcp[5], tcp[6], tcp[3]]
                
                # If home pose not configured, use current pose
                if self.home_pose is None:
                    logger.info("Home pose not configured, using current pose")
                    
                    # Get current pose for single arm
                    states = self.robot.states()
                    tcp = states.tcp_pose.copy()
                    
                    # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                    self.home_pose = [tcp[0], tcp[1], tcp[2], tcp[4], tcp[5], tcp[6], tcp[3]]
                
                logger.info(f"Using home pose: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set home pose: {e}")
            return False

    def _go_home(self) -> bool:
        """Move robot(s) to home pose."""
        try:
            if self.control_type == "joint":
                logger.info("Joint control mode - skipping go_home")
                return True
                
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right or not self.home_pose_left or not self.home_pose_right:
                    logger.error("Dual-arm robots or home poses not initialized")
                    return False
                
                # Move both arms to home poses
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                left_target = [self.home_pose_left[0], self.home_pose_left[1], self.home_pose_left[2],
                             self.home_pose_left[6], self.home_pose_left[3], self.home_pose_left[4], self.home_pose_left[5]]
                right_target = [self.home_pose_right[0], self.home_pose_right[1], self.home_pose_right[2],
                              self.home_pose_right[6], self.home_pose_right[3], self.home_pose_right[4], self.home_pose_right[5]]
                
                logger.info("Moving dual-arm robots to home poses...")
                self.robot_left.SendCartesianMotionForce(left_target)
                self.robot_right.SendCartesianMotionForce(right_target)
                
                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Dual-arm robots moved to home poses")
            else:
                if not self.robot or not self.home_pose:
                    logger.error("Robot or home pose not initialized")
                    return False
                
                # Move single arm to home pose
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                target = [self.home_pose[0], self.home_pose[1], self.home_pose[2],
                         self.home_pose[6], self.home_pose[3], self.home_pose[4], self.home_pose[5]]
                
                logger.info("Moving robot to home pose...")
                self.robot.SendCartesianMotionForce(
                    target,
                    max_linear_vel=self.max_linear_vel,
                    max_linear_acc=self.max_linear_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc
                )
                
                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Robot moved to home pose")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to go home: {e}")
            return False

    def get_robot_state(self) -> np.ndarray:
        """Get current robot states."""
        try:
            if not self.is_operational:
                return np.zeros(self.data_shape, dtype=self.data_dtype)

            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)
                
                # Get states from both robots
                left_states = self.robot_left.states()
                right_states = self.robot_right.states()
                
                # Combine joint positions and velocities for both arms
                left_positions = np.array(left_states.q, dtype=self.data_dtype)
                left_velocities = np.array(left_states.dq, dtype=self.data_dtype)
                left_eef_pose = np.array(left_states.tcp_pose, dtype=self.data_dtype)
                left_gripper_width = np.array([self.gripper_left.states().width], dtype=self.data_dtype) if self.utilize_gripper else np.array([0], dtype=self.data_dtype)
                right_positions = np.array(right_states.q, dtype=self.data_dtype)
                right_velocities = np.array(right_states.dq, dtype=self.data_dtype)
                right_eef_pose = np.array(right_states.tcp_pose, dtype=self.data_dtype)
                right_gripper_width = np.array([self.gripper_right.states().width], dtype=self.data_dtype) if self.utilize_gripper else np.array([0], dtype=self.data_dtype)
                # Combine into single array: [left_pos, left_vel, right_pos, right_vel]
                combined_states = np.concatenate([left_positions, left_velocities, left_eef_pose, left_gripper_width,
                                                right_positions, right_velocities, right_eef_pose, right_gripper_width])
            else:
                if not self.robot:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)
                
                states = self.robot.states()
                if self.using_tcp_pose:
                    ### Using tcp pose
                    # Combine tcp pose and gripper width
                    joint_positions = np.array(states.q, dtype=self.data_dtype)
                    joint_velocities = np.array(states.dq, dtype=self.data_dtype)
                    eef_pose = np.array(states.tcp_pose, dtype=self.data_dtype)
                    gripper_width = np.array([self.gripper.states().width], dtype=self.data_dtype) if self.utilize_gripper else np.array([0], dtype=self.data_dtype)
                    
                    combined_states = np.concatenate([joint_positions,joint_velocities,eef_pose,gripper_width])
                else:
                    ### Using joint 
                    # Combine joint positions and velocities
                    joint_positions = np.array(states.q, dtype=self.data_dtype)
                    joint_velocities = np.array(states.dq, dtype=self.data_dtype)
                    
                    # Combine into single array: [pos1, pos2, ..., pos7, vel1, vel2, ..., vel7]
                    combined_states = np.concatenate([joint_positions, joint_velocities])

            return combined_states

        except Exception as e:
            logger.error(f"Failed to get robot states: {e}")
            return np.zeros(self.data_shape, dtype=self.data_dtype)

    def get_control_array(self, control_dict: Dict[str, np.ndarray]) -> np.ndarray:
        try:
            if 'RizonRobot_2_control' in control_dict:
                return control_dict['RizonRobot_2_control']
            elif 'RizonRobot_1_control' in control_dict:
                return control_dict["RizonRobot_1_control"]
            seq_dict = {}
            """Get control array from control dictionary."""
            for key_name in control_dict.keys():
                if self.is_dual_arm:
                    if "Airexoskeleton" in key_name or "ViveTracker" in key_name:
                        seq_dict[key_name] = 0 if "0" in key_name else 2
                    else:
                        assert "RotaryEncoder" in key_name
                        seq_dict[key_name] = 1 if "0" in key_name else 3
                        # TODO: handle gripper width scaling in a better way
                        control_dict[key_name] = min(control_dict[key_name]/1000.,0.085) # convert mm to m
                else:
                    if "Airexoskeleton" in key_name or "ViveTracker" in key_name:
                        seq_dict[key_name] = 0
                    else:
                        assert "RotaryEncoder" in key_name
                        seq_dict[key_name] = 1
                        # TODO: handle gripper width scaling in a better way
                        control_dict[key_name] = min(control_dict[key_name]/1000.,0.085) # convert mm to m

            sorted_keys = sorted(control_dict.keys(), key=lambda x: seq_dict[x])
            control_array = np.concatenate([control_dict[key] for key in sorted_keys])
            assert len(control_array.shape) == 1
            return control_array
        except Exception as e:
            # logger.error(f"Failed to get control array: {e}")
            return None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """Execute control command based on control type."""
        try:
            # get control array from action dict
            control_array = self.get_control_array(control_dict)
            # logger.info(f"control_array: {control_array}")
            if control_array is None:
                return
            if not self.is_operational:
                return

            if self.control_type == "joint":
                self._execute_joint_command(control_array)
            elif self.control_type in ["eef_abs", "eef_delta", "eef_del"]:
                self._execute_eef_command(control_array)
            else:
                logger.error(f"Unknown control type: {self.control_type}")

        except Exception as e:
            logger.error(f"Failed to execute control command: {e}")

    def _execute_joint_command(self, control_array: np.ndarray) -> None:
        """Execute joint position command."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return
                
                # Ensure we have 14 joint positions for dual arm
                if len(control_array) != 14:
                    logger.warning(f"Expected 14 joint positions for dual arm, got {len(control_array)}")
                    return
                
                # Split into left and right arm commands
                left_pos = control_array[:7].tolist()
                right_pos = control_array[7:14].tolist()
                
                left_vel = [0.0] * 7
                left_acc = [0.0] * 7
                right_vel = [0.0] * 7
                right_acc = [0.0] * 7
                
                # Send commands to both robots
                self.robot_left.SendJointPosition(
                    left_pos, left_vel, left_acc,
                    self.max_velocity[:7], self.max_acceleration[:7]
                )
                self.robot_right.SendJointPosition(
                    right_pos, right_vel, right_acc,
                    self.max_velocity[7:14], self.max_acceleration[7:14]
                )
            else:
                if not self.robot:
                    return
                
                # Ensure we have 7 joint positions
                if len(control_array) != 7:
                    logger.warning(f"Expected 7 joint positions, got {len(control_array)}")
                    return
                
                # Convert to list
                target_pos = control_array.tolist()
                target_vel = [0.0] * 7
                target_acc = [0.0] * 7
                
                # Send command
                self.robot.SendJointPosition(
                    target_pos, target_vel, target_acc,
                    self.max_velocity, self.max_acceleration
                )

        except Exception as e:
            logger.error(f"Failed to execute joint command: {e}")

    def _execute_eef_command(self, control_array: np.ndarray) -> None:
        """Execute end-effector command."""
        try:
            if self.is_dual_arm:
                # logger.info(f"dual arm: {self.is_dual_arm}")
                if not self.robot_left or not self.robot_right:
                    return

                # Determine expected array size based on gripper usage
                expected_size = 16 if self.utilize_gripper else 14
                if len(control_array) != expected_size:
                    logger.warning(f"Expected {expected_size} elements for dual arm EEF control, got {len(control_array)}")
                    return
                
                # Split into left and right arm commands (pose + gripper + pose + gripper format)
                if self.utilize_gripper:
                    left_command = control_array[:7]
                    left_gripper_width = control_array[7] 
                    right_command = control_array[8:15]
                    right_gripper_width = control_array[15] 
                else:
                    left_command = control_array[:7]
                    right_command = control_array[7:14]
                # Process each arm
                left_target = self._process_eef_command(left_command, self.home_pose_left, "left")
                right_target = self._process_eef_command(right_command, self.home_pose_right, "right")
                # logger.info(f"left_target: {left_target}")
                # logger.info(f"right_target: {right_target}")
                if left_target is not None and right_target is not None:
                    self.robot_left.SendCartesianMotionForce(left_target, 
                                                           max_linear_vel=self.max_linear_vel, 
                                                           max_linear_acc=self.max_linear_acc, 
                                                           max_angular_vel=self.max_angular_vel, 
                                                           max_angular_acc=self.max_angular_acc)
                    self.robot_right.SendCartesianMotionForce(right_target, 
                                                            max_linear_vel=self.max_linear_vel, 
                                                            max_linear_acc=self.max_linear_acc, 
                                                            max_angular_vel=self.max_angular_vel, 
                                                            max_angular_acc=self.max_angular_acc)
                    
                    # Control grippers if enabled
                    if self.utilize_gripper and self.gripper_left and self.gripper_right:
                        self.gripper_left.Move(min(left_gripper_width,0.085), 0.1, 100)
                        self.gripper_right.Move(min(right_gripper_width,0.085), 0.1, 100)
            else:
                if not self.robot:
                    return
                # logger.info(f"dual arm: {self.is_dual_arm}")
                # Determine expected array size based on gripper usage
                expected_size = 8 if self.utilize_gripper else 7
                if len(control_array) != expected_size:
                    logger.warning(f"Expected {expected_size} elements for EEF control, got {len(control_array)}")
                    return
                
                # Split command and gripper width
                if self.utilize_gripper:
                    eef_command = control_array[:7]
                    gripper_width = control_array[7]
                    gripper_width_value = float(gripper_width)
                    logger.debug(f"Single arm gripper control: gripper_width={gripper_width}, control_array={control_array}")
                else:
                    eef_command = control_array
                
                # Process single arm command
                target = self._process_eef_command(eef_command, self.home_pose, "single")
                
                if target is not None:
                    # print(self.robot.mode())
                    self.robot.SendCartesianMotionForce(target, 
                                                      max_linear_vel=self.max_linear_vel, 
                                                      max_linear_acc=self.max_linear_acc, 
                                                      max_angular_vel=self.max_angular_vel, 
                                                      max_angular_acc=self.max_angular_acc)

                    # Control gripper if enabled
                    if self.utilize_gripper and self.gripper:
                        # logger.info(f"Moving gripper to width: {gripper_width}")
                        self.gripper.Move(min(gripper_width,0.085), 0.1, 100)
                    elif self.utilize_gripper and not self.gripper:
                        logger.warning("Gripper enabled but gripper object not available")

        except Exception as e:
            logger.error(f"Failed to execute EEF command: {e}")

    def _process_eef_command(self, command: np.ndarray, home_pose, arm_name: str):
        """Process end-effector command based on control type."""
        try:
            if home_pose is None:
                logger.error(f"Home pose not set for {arm_name} arm")
                return None
            
            # Extract position and quaternion from command [x, y, z, qx, qy, qz, qw]
            cmd_pos = command[:3]
            cmd_quat = command[3:7]
            if self.control_type == "eef_abs":
                # Absolute positioning
                target_pos = cmd_pos
                target_quat = cmd_quat
                threshold = 0.50
                if arm_name == "left":
                    crt_pos = self.robot_left.states().tcp_pose[:3]
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                elif arm_name == "right":
                    crt_pos = self.robot_right.states().tcp_pose[:3]
                else:
                    crt_pos = self.robot.states().tcp_pose[:3]
                delta_move = np.linalg.norm(crt_pos - target_pos)
                if delta_move > threshold:
                    logger.error(f"{arm_name} arm's move is too large {delta_move}")
                    return None

            elif self.control_type == "eef_delta":
                # Apply motion scaling if configured
                if self.delta_eef_motion_scaler is not None:
                    cmd_pos, cmd_quat = scalar_transform(
                        cmd_pos.copy(), cmd_quat.copy(),
                        self.delta_eef_motion_scaler['trans'],
                        self.delta_eef_motion_scaler['rot']
                    )
                if self.airexo_type == "ver1":
                    # if (self.mirror_mode and arm_name == "right") or (not self.mirror_mode and arm_name == "left"):
                    if arm_name == "left":
                        _ , cmd_quat = similarity_transform(None, cmd_quat.copy())
                        if not self.mirror_mode:
                            cmd_pos = R.from_euler('xyz', np.array([0.0, 0.0, np.pi])).apply(cmd_pos.copy())
                    if self.mirror_mode:
                        _ , cmd_quat = similarity_transform(None, cmd_quat.copy(),mirror_mode = True)
                    # Delta positioning - apply transformation relative to home pose

                    # Current home position and orientation
                    home_pos = np.array(home_pose[:3])
                    home_quat = np.array(home_pose[3:7])

                    # Apply position delta - first rotate cmd_pos by home rotation, then add to home position
                    home_rot = R.from_quat(home_quat)
                    if arm_name == "left":
                        delta_pos = self.wb_R_left.inv().apply(cmd_pos.copy())
                        if self.mirror_mode:
                            delta_pos[1] *= -1
                        target_pos = home_pos + delta_pos
                    else:
                        delta_pos = self.wb_R_right.inv().apply(cmd_pos.copy())
                        if self.mirror_mode:
                            delta_pos[1] *= -1
                        target_pos = home_pos + delta_pos
                else:
                    # if (self.mirror_mode and arm_name == "right") or (not self.mirror_mode and arm_name == "left"):
                    if arm_name == "right":
                        _ , cmd_quat = axis_similarity_transform(None, cmd_quat.copy(),"xz")
                    if self.mirror_mode:
                        _ , cmd_quat = similarity_transform(None, cmd_quat.copy(),mirror_mode = True)
                    # Delta positioning - apply transformation relative to home pose

                    # Current home position and orientation
                    home_pos = np.array(home_pose[:3])
                    home_quat = np.array(home_pose[3:7])

                    # Apply position delta - first rotate cmd_pos by home rotation, then add to home position
                    home_rot = R.from_quat(home_quat)
                    if arm_name == "left":
                        delta_pos = self.wb_R_left.inv().apply(cmd_pos.copy())
                        if self.mirror_mode:
                            delta_pos[0] *= -1
                        target_pos = home_pos + delta_pos
                    else:
                        delta_pos = self.wb_R_right.inv().apply(cmd_pos.copy())
                        if self.mirror_mode:
                            delta_pos[1] *= -1
                        target_pos = home_pos + delta_pos
                # Apply rotation delta (right multiply: home_rot * delta_rot)
                delta_rot = R.from_quat(cmd_quat.copy())
                target_rot = home_rot * delta_rot
                target_quat = target_rot.as_quat()


            elif self.control_type == "eef_del":
                if self.delta_eef_motion_scaler is not None:
                    cmd_pos, cmd_quat = scalar_transform(
                        cmd_pos.copy(), cmd_quat.copy(),
                        self.delta_eef_motion_scaler['trans'],
                        self.delta_eef_motion_scaler['rot']
                    )
                cmd_rot = R.from_quat(cmd_quat)
                # Current home position and orientation
                # TODO: distinguish left robot and right robot
                crt_pose = self.robot_left.states().tcp_pose
                crt_pos = crt_pose[:3]
                crt_quat = crt_pose[3:7]
                crt_rot = R.from_quat(crt_quat)
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                target_pos = crt_pos + crt_rot.apply(cmd_pos.copy())
                target_rot = crt_rot * cmd_rot
                target_quat = target_rot.as_quat()

            else:
                logger.error(f"Unknown EEF control type: {self.control_type}")
                return None
            
            # Convert to Flexiv format [x, y, z, qw, qx, qy, qz]
            flexiv_pose = [target_pos[0], target_pos[1], target_pos[2],
                          target_quat[3], target_quat[0], target_quat[1], target_quat[2]]
            
            return flexiv_pose
            
        except Exception as e:
            logger.error(f"Failed to process EEF command for {arm_name}: {e}")
            return None

    def start_server(self) -> bool:
        """Start the Rizon robot server."""
        try:
            # Connect to robot
            if not self._connect_robot():
                logger.error("Failed to connect to robot")
                return False

            # Switch to appropriate control mode
            if self.control_type == "joint":
                if not self._switch_to_joint_control():
                    logger.error("Failed to switch to joint control mode")
                    return False
            elif self.control_type in ["eef_abs", "eef_delta", "eef_del"]:
                if not self._switch_to_eef_control():
                    logger.error("Failed to switch to end-effector control mode")
                    return False
                
                # Set home pose for EEF control (from config or current pose)
                if not self._set_home_pose():
                    logger.error("Failed to set home pose")
                    return False
                
                # Always go to home position for EEF control
                logger.info("Moving to home position for EEF control...")
                if not self._go_home():
                    logger.error("Failed to go to home position")
                    return False

            # Start base robot server
            super().start_server()
            logger.info(f"Rizon robot device {self.device_id} started with {self.control_type} control")
            return True

        except Exception as e:
            logger.error(f"Failed to start Rizon robot device: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the Rizon robot server."""
        try:
            # Disable robot(s) if connected
            if self.is_dual_arm:
                if self.robot_left and self.is_enabled:
                    try:
                        self.robot_left.Disable()
                        logger.info("Left robot disabled")
                    except:
                        pass
                if self.robot_right and self.is_enabled:
                    try:
                        self.robot_right.Disable()
                        logger.info("Right robot disabled")
                    except:
                        pass
            else:
                if self.robot and self.is_enabled:
                    try:
                        self.robot.Disable()
                        logger.info("Robot disabled")
                    except:
                        pass

            # Stop base robot server
            super().stop_server()
            logger.info(f"Rizon robot device {self.device_id} stopped")

        except Exception as e:
            logger.error(f"Error stopping Rizon robot device: {e}")

    def get_status(self) -> dict:
        """Get device status information."""
        status = super().get_status()
        status.update({
            'robot_sn': self.robot_sn,
            'control_type': self.control_type,
            'is_dual_arm': self.is_dual_arm,
            'is_operational': self.is_operational,
            'is_enabled': self.is_enabled,
            'current_mode': self.current_mode,
            'control_shm_connected': all(shm is not None for shm in self.control_shared_memory.values()) if self.control_shared_memory else False,
            'home_pose_set': (self.home_pose is not None) if not self.is_dual_arm else 
                           (self.home_pose_left is not None and self.home_pose_right is not None),
            'impedance_parameter': self.impedance_parameter if self.control_type in ["eef_abs", "eef_delta", "eef_del"] else "Not applicable (joint mode)",
            'motion_limits': {
                'max_linear_vel': self.max_linear_vel,
                'max_linear_acc': self.max_linear_acc,
                'max_angular_vel': self.max_angular_vel,
                'max_angular_acc': self.max_angular_acc
            } if self.control_type in ["eef_abs", "eef_delta", "eef_del"] else "Not applicable (joint mode)"
        })
        return status


if __name__ == "__main__":
    main() 