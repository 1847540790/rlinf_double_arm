#!/usr/bin/env python3
"""
Teleoperator device for robot control with GUI sliders.

Author: Jun Lv, Zixi Ying
"""

import time
import numpy as np
import multiprocessing.shared_memory as shm
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List, Tuple, Optional
from devices.base import BaseDevice
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    calculate_device_shm_size, DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    get_dtype
)
from utils.logger_config import logger
from scipy.spatial.transform import Rotation as R
from utils.time_control import precise_loop_timing
from utils.transform import compute_deviation, compute_deviation_3

# Try to import exoskeleton module, handle import error gracefully
try:
    from exoskeleton import AngleEncoder
    EXOSKELETON_AVAILABLE = True
except ImportError:
    logger.warning("exoskeleton module not available. Airexoskeleton classes will not work.")
    EXOSKELETON_AVAILABLE = False
    AngleEncoder = None

class JointSlider(BaseDevice):
    """
    Joint slider teleoperator device for controlling robot joint positions.
    Provides GUI sliders for real-time joint control.
    """
    # This is all wirtten by Cursor, not reviewed
    
    def __init__(self, device_id=0, fps=10.0, num_joints=7,
                 data_dtype=np.float64, buffer_size=100, hardware_latency_ms=0.0,
                 joint_ranges=None):
        """
        Initialize the joint slider teleoperator.
        
        Args:
            device_id: Unique identifier for this device instance
            fps: Update frequency in Hz
            num_joints: Number of joints to control
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            joint_ranges: List of (min, max) ranges for each joint (default: [-pi, pi] for all)
        """
        # Initialize base device with joint position data
        super().__init__(device_id=device_id, data_shape=(num_joints,), fps=fps,
                        data_dtype=data_dtype, buffer_size=buffer_size, 
                        hardware_latency_ms=hardware_latency_ms)
        
        # Override device name
        self.device_name = "JointSlider"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Joint control attributes
        self.num_joints = num_joints
        
        # Set default joint ranges if not provided
        if joint_ranges is None:
            self.joint_ranges = [(-np.pi, np.pi)] * num_joints  # Default: -pi to pi
        else:
            self.joint_ranges = joint_ranges
            
        # Current joint positions (initialize to center of ranges)
        self.current_joints = []
        for min_val, max_val in self.joint_ranges:
            self.current_joints.append((min_val + max_val) / 2)
        
        # GUI elements
        self.root = None
        self.sliders = []
        self.labels = []
        self.value_labels = []
        
        # Threading
        self.running = False
        self.gui_thread = None
        self.lock = threading.Lock()
        
        logger.info(f"Joint Slider teleoperator initialized")
        logger.info(f"Number of joints: {num_joints}")
        logger.info(f"Joint ranges: {self.joint_ranges}")
        logger.info(f"Current positions: {[f'{pos:.3f}' for pos in self.current_joints]}")
        logger.info("")
        
    def _generate_joint_array(self) -> np.ndarray:
        """
        Generate current joint position array.
        
        Returns:
            numpy.ndarray: Current joint positions
        """
        with self.lock:
            numpy_dtype = get_dtype(self.data_dtype)
            return np.array(self.current_joints, dtype=numpy_dtype)
    
    def _update_joint(self, joint_idx: int, value: float) -> None:
        """
        Update joint position.
        
        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        with self.lock:
            if 0 <= joint_idx < self.num_joints:
                min_val, max_val = self.joint_ranges[joint_idx]
                self.current_joints[joint_idx] = np.clip(value, min_val, max_val)
                logger.debug(f"Joint {joint_idx}: {self.current_joints[joint_idx]:.3f}")
    
    def _create_gui(self) -> None:
        """Create the GUI with sliders for each joint."""
        self.root = tk.Tk()
        self.root.title(f"Joint Slider Control - Device {self.device_id}")
        self.root.geometry("400x600")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Joint Position Control", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create sliders for each joint
        for i in range(self.num_joints):
            min_val, max_val = self.joint_ranges[i]
            current_val = self.current_joints[i]
            
            # Joint label
            joint_label = ttk.Label(main_frame, text=f"Joint {i}:")
            joint_label.grid(row=i+1, column=0, sticky=tk.W, pady=5)
            
            # Slider
            slider = ttk.Scale(
                main_frame,
                from_=min_val,
                to=max_val,
                value=current_val,
                orient=tk.HORIZONTAL,
                length=200,
                command=lambda val, idx=i: self._on_slider_change(idx, val)
            )
            slider.grid(row=i+1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
            
            # Value label
            value_label = ttk.Label(main_frame, text=f"{current_val:.3f}")
            value_label.grid(row=i+1, column=2, sticky=tk.W, padx=(5, 0), pady=5)
            
            self.sliders.append(slider)
            self.labels.append(joint_label)
            self.value_labels.append(value_label)
        
        # Reset button
        reset_button = ttk.Button(main_frame, text="Reset All", command=self._reset_all)
        reset_button.grid(row=self.num_joints+1, column=0, columnspan=3, pady=20)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=self.num_joints+2, column=0, columnspan=3, pady=10)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
    def _on_slider_change(self, joint_idx: int, value: str) -> None:
        """Handle slider value change."""
        try:
            value = float(value)
            self._update_joint(joint_idx, value)
            
            # Update value label
            if joint_idx < len(self.value_labels):
                self.value_labels[joint_idx].config(text=f"{value:.3f}")
                
        except ValueError:
            logger.warning(f"Invalid slider value: {value}")
    
    def _reset_all(self) -> None:
        """Reset all joints to center of their ranges."""
        with self.lock:
            for i in range(self.num_joints):
                min_val, max_val = self.joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.current_joints[i] = center_val
                
                # Update slider and label
                if i < len(self.sliders):
                    self.sliders[i].set(center_val)
                if i < len(self.value_labels):
                    self.value_labels[i].config(text=f"{center_val:.3f}")
                    
            logger.info("All joints reset to center")
    
    def _on_closing(self) -> None:
        """Handle window closing."""
        logger.info("GUI window closing...")
        self.running = False
        if self.root:
            self.root.quit()
    
    def _gui_loop(self) -> None:
        """Main GUI loop."""
        try:
            self._create_gui()
            logger.info("GUI created successfully")
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Running", foreground="green")
            
            # Start GUI main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error in GUI loop: {e}")
            self.running = False
    
    def start_server(self) -> None:
        """Start the joint slider teleoperator server."""
        if self.running:
            logger.info(f"Joint Slider teleoperator {self.device_id} is already running")
            return
        
        logger.info(f"Starting Joint Slider teleoperator {self.device_id} server...")
        self.running = True
        
        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        
        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        
        # Start GUI thread
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()
        
        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main data writing loop
        while self.running:
            try:
                # Get current joint positions and write to SHM
                joint_array = self._generate_joint_array()
                timestamp_ns = time.time_ns()
                self._write_array_to_shm_with_timestamp(joint_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()
                
            except Exception as e:
                logger.error(f"Error in data writing: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the joint slider teleoperator server."""
        if not self.running:
            return
        
        logger.info(f"Stopping Joint Slider teleoperator {self.device_id} server...")
        self.running = False
        
        # Close GUI
        if self.root:
            try:
                self.root.quit()
            except:
                pass
        
        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2.0)
        
        self._cleanup_shared_memory()
        logger.info("Server stopped")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "num_joints": self.num_joints,
            "joint_ranges": self.joint_ranges,
            "current_joints": self.current_joints.copy()
        })
        return info


class DualArmJointSlider(BaseDevice):
    """
    Dual-arm joint slider teleoperator device for controlling two robot arms.
    Provides GUI sliders for real-time joint control of both arms.
    """
    
    def __init__(self, device_id=0, fps=10.0, left_arm_joints=7, right_arm_joints=7,
                 data_dtype=np.float64, buffer_size=100, hardware_latency_ms=0.0,
                 left_joint_ranges=None, right_joint_ranges=None):
        """
        Initialize the dual-arm joint slider teleoperator.
        
        Args:
            device_id: Unique identifier for this device instance
            fps: Update frequency in Hz
            left_arm_joints: Number of joints in left arm
            right_arm_joints: Number of joints in right arm
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            left_joint_ranges: List of (min, max) ranges for left arm joints (default: [-pi, pi] for all)
            right_joint_ranges: List of (min, max) ranges for right arm joints (default: [-pi, pi] for all)
        """
        # Initialize base device with joint position data for both arms
        total_joints = left_arm_joints + right_arm_joints
        super().__init__(device_id=device_id, data_shape=(total_joints,), fps=fps,
                        data_dtype=data_dtype, buffer_size=buffer_size, 
                        hardware_latency_ms=hardware_latency_ms)
        
        # Override device name
        self.device_name = "DualArmJointSlider"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Joint control attributes
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.total_joints = total_joints
        
        # Set default joint ranges if not provided
        if left_joint_ranges is None:
            self.left_joint_ranges = [(-np.pi, np.pi)] * left_arm_joints  # Default: -pi to pi
        else:
            self.left_joint_ranges = left_joint_ranges
            
        if right_joint_ranges is None:
            self.right_joint_ranges = [(-np.pi, np.pi)] * right_arm_joints  # Default: -pi to pi
        else:
            self.right_joint_ranges = right_joint_ranges
            
        # Current joint positions (initialize to center of ranges)
        self.left_current_joints = []
        for min_val, max_val in self.left_joint_ranges:
            self.left_current_joints.append((min_val + max_val) / 2)
            
        self.right_current_joints = []
        for min_val, max_val in self.right_joint_ranges:
            self.right_current_joints.append((min_val + max_val) / 2)
        
        # GUI elements
        self.root = None
        self.left_sliders = []
        self.right_sliders = []
        self.left_labels = []
        self.right_labels = []
        self.left_value_labels = []
        self.right_value_labels = []
        
        # Threading
        self.running = False
        self.gui_thread = None
        self.lock = threading.Lock()
        
        logger.info(f"Dual-Arm Joint Slider teleoperator initialized")
        logger.info(f"Left arm joints: {left_arm_joints}")
        logger.info(f"Right arm joints: {right_arm_joints}")
        logger.info(f"Total joints: {total_joints}")
        logger.info(f"Left joint ranges: {self.left_joint_ranges}")
        logger.info(f"Right joint ranges: {self.right_joint_ranges}")
        logger.info(f"Left current positions: {[f'{pos:.3f}' for pos in self.left_current_joints]}")
        logger.info(f"Right current positions: {[f'{pos:.3f}' for pos in self.right_current_joints]}")
        logger.info("")
        
    def _generate_joint_array(self) -> np.ndarray:
        """
        Generate current joint position array for both arms.
        
        Returns:
            numpy.ndarray: Current joint positions [left_arm_joints, right_arm_joints]
        """
        with self.lock:
            numpy_dtype = get_dtype(self.data_dtype)
            # Combine left and right arm joint positions
            all_joints = self.left_current_joints + self.right_current_joints
            return np.array(all_joints, dtype=numpy_dtype)
    
    def _update_left_joint(self, joint_idx: int, value: float) -> None:
        """
        Update left arm joint position.
        
        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        if 0 <= joint_idx < self.left_arm_joints:
            min_val, max_val = self.left_joint_ranges[joint_idx]
            with self.lock:
                self.left_current_joints[joint_idx] = np.clip(value, min_val, max_val)
    
    def _update_right_joint(self, joint_idx: int, value: float) -> None:
        """
        Update right arm joint position.
        
        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        if 0 <= joint_idx < self.right_arm_joints:
            min_val, max_val = self.right_joint_ranges[joint_idx]
            with self.lock:
                self.right_current_joints[joint_idx] = np.clip(value, min_val, max_val)
    
    def _create_gui(self) -> None:
        """Create the dual-arm GUI with sliders for both arms."""
        self.root = tk.Tk()
        self.root.title(f"Dual-Arm Joint Slider - Device {self.device_id}")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        
        # Left Arm Section
        left_frame = ttk.LabelFrame(main_frame, text="Left Arm", padding="5")
        left_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right Arm Section
        right_frame = ttk.LabelFrame(main_frame, text="Right Arm", padding="5")
        right_frame.grid(row=0, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Create sliders for left arm
        for i in range(self.left_arm_joints):
            # Joint label
            label = ttk.Label(left_frame, text=f"Joint {i+1}")
            label.grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            self.left_labels.append(label)
            
            # Slider
            min_val, max_val = self.left_joint_ranges[i]
            slider = ttk.Scale(left_frame, from_=min_val, to=max_val, 
                             value=self.left_current_joints[i],
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self._on_left_slider_change(idx, val))
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
            self.left_sliders.append(slider)
            
            # Value label
            value_label = ttk.Label(left_frame, text=f"{self.left_current_joints[i]:.3f}")
            value_label.grid(row=i, column=2, sticky=tk.W)
            self.left_value_labels.append(value_label)
        
        # Create sliders for right arm
        for i in range(self.right_arm_joints):
            # Joint label
            label = ttk.Label(right_frame, text=f"Joint {i+1}")
            label.grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            self.right_labels.append(label)
            
            # Slider
            min_val, max_val = self.right_joint_ranges[i]
            slider = ttk.Scale(right_frame, from_=min_val, to=max_val, 
                             value=self.right_current_joints[i],
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self._on_right_slider_change(idx, val))
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
            self.right_sliders.append(slider)
            
            # Value label
            value_label = ttk.Label(right_frame, text=f"{self.right_current_joints[i]:.3f}")
            value_label.grid(row=i, column=2, sticky=tk.W)
            self.right_value_labels.append(value_label)
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(20, 0))
        
        # Reset button (single button like single arm)
        reset_all_btn = ttk.Button(button_frame, text="Reset All", command=self._reset_all)
        reset_all_btn.grid(row=0, column=0)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        logger.info("Dual-arm GUI created successfully")
    
    def _on_left_slider_change(self, joint_idx: int, value: str) -> None:
        """Callback for left arm slider changes."""
        value = float(value)
        self._update_left_joint(joint_idx, value)
        
        # Update value label
        if 0 <= joint_idx < len(self.left_value_labels):
            self.left_value_labels[joint_idx].config(text=f"{value:.3f}")
    
    def _on_right_slider_change(self, joint_idx: int, value: str) -> None:
        """Callback for right arm slider changes."""
        value = float(value)
        self._update_right_joint(joint_idx, value)
        
        # Update value label
        if 0 <= joint_idx < len(self.right_value_labels):
            self.right_value_labels[joint_idx].config(text=f"{value:.3f}")
    
    def _reset_all(self) -> None:
        """Reset all joints to center of their ranges."""
        with self.lock:
            # Reset left arm joints
            for i in range(self.left_arm_joints):
                min_val, max_val = self.left_joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.left_current_joints[i] = center_val
                
                # Update slider and label
                if i < len(self.left_sliders):
                    self.left_sliders[i].set(center_val)
                if i < len(self.left_value_labels):
                    self.left_value_labels[i].config(text=f"{center_val:.3f}")
            
            # Reset right arm joints
            for i in range(self.right_arm_joints):
                min_val, max_val = self.right_joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.right_current_joints[i] = center_val
                
                # Update slider and label
                if i < len(self.right_sliders):
                    self.right_sliders[i].set(center_val)
                if i < len(self.right_value_labels):
                    self.right_value_labels[i].config(text=f"{center_val:.3f}")
                    
            logger.info("All joints reset to center")
    
    def _on_closing(self) -> None:
        """Handle GUI window close event."""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def _gui_loop(self) -> None:
        """Run the GUI main loop in a separate thread."""
        try:
            self._create_gui()
            logger.info("Dual-arm GUI created successfully")
            
            # Start GUI main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error in GUI loop: {e}")
            self.running = False
    
    def start_server(self) -> None:
        """Start the dual-arm joint slider server."""
        if self.running:
            logger.info(f"Dual-Arm Joint Slider teleoperator {self.device_id} is already running")
            return
        
        logger.info(f"Starting Dual-Arm Joint Slider teleoperator {self.device_id} server...")
        self.running = True
        
        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        
        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        
        # Start GUI in separate thread
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()
        
        logger.info("Dual-arm GUI started in separate thread")
        logger.info("Use the sliders to control both robot arms")
        logger.info("Press Ctrl+C to stop...")
        
        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main data loop
        while self.running:
            try:
                # Generate current joint array
                joint_array = self._generate_joint_array()
                timestamp_ns = time.time_ns()
                self._write_array_to_shm_with_timestamp(joint_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()
                
            except Exception as e:
                logger.error(f"Error in dual-arm joint slider server: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the dual-arm joint slider server."""
        self.running = False
        
        # Close GUI
        if self.root:
            try:
                self.root.quit()
            except:
                pass
        
        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2)
        
        # Clean up shared memory
        self._cleanup_shared_memory()
        
        logger.info("Dual-Arm Joint Slider server stopped")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device_name': self.device_name,
            'device_id': self.device_id,
            'fps': self.fps,
            'data_shape': self.data_shape,
            'data_dtype': self.data_dtype,
            'buffer_size': self.buffer_size,
            'hardware_latency_ms': self.hardware_latency_ms,
            'left_arm_joints': self.left_arm_joints,
            'right_arm_joints': self.right_arm_joints,
            'total_joints': self.total_joints,
            'left_joint_ranges': self.left_joint_ranges,
            'right_joint_ranges': self.right_joint_ranges,
            'current_left_joints': self.left_current_joints,
            'current_right_joints': self.right_current_joints
        }


class Airexoskeleton(BaseDevice):
    """
    Air exoskeleton device for reading joint angle data from physical exoskeleton.
    Reads joint angles and publishes to shared memory.
    """
    
    def __init__(self, device_id=0, fps=100.0, num_joints=7,
                 data_dtype=np.float64, buffer_size=100, hardware_latency_ms=5.0,
                 port="/dev/ttyUSB0", baudrate=1000000, encoder_ids=None,
                 ctrl_type = "joint",arm_type="left",scalar_first_quat_ctrl = False,
                 utilize_gripper=False, utilize_button=False, button_key='space',
                 airexo_type="ver1"):
        """
        Initialize the air exoskeleton device.
        
        Args:
            device_id: Unique identifier for this device instance
            fps: Update frequency in Hz (default: 100.0 to match encoder refresh rate)
            num_joints: Number of joints to read (default: 7)
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            port: Serial port for encoder communication (default: /dev/ttyUSB0)
            baudrate: Baudrate for serial communication (default: 1000000)
            encoder_ids: List of encoder IDs (default: [1,2,3,4,5,6,7])
            ctrl_type: Control type ("joint", "eef_delta", "eef_abs")
            arm_type: Arm type ("left" or "right")
            scalar_first_quat_ctrl: Whether to use scalar first quaternion control
            utilize_gripper: Whether to enable gripper support
            utilize_button: Whether to enable button control
            button_key: Key for button control
            airexo_type: Version of airexo ("ver1" or "ver2")
        """
        # Initialize base device with joint angle data
        super().__init__(device_id=device_id, data_shape=(num_joints+int(utilize_gripper),), fps=fps,
                        data_dtype=data_dtype, buffer_size=buffer_size, 
                        hardware_latency_ms=hardware_latency_ms)
        
        # Override device name
        self.device_name = "Airexoskeleton"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Exoskeleton attributes
        self.num_joints = num_joints
        self.port = port
        self.baudrate = baudrate
        self.encoder_ids = encoder_ids if encoder_ids else list(range(1, num_joints + 1))
        
        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint", "eef_delta", "eef_abs"]
        self.arm_type = arm_type  # "left" or "right"
        
        # Store airexo type
        self.airexo_type = airexo_type
        
        # Button control support
        self.utilize_button = utilize_button
        if self.utilize_button:
            assert self.ctrl_type == "eef_delta", "Button control requires ctrl_type == 'eef_delta'"
            from utils.button import Button
            self.button = Button(button_key)

        self.state = True  # True: running, False: paused
        self.last_state = True
        self.detach_time = None  # Time when detach occurred
        self.resume_cooldown = 1.0  # Minimum time (seconds) between detach and resume
        # Pose storage for smooth transition
        self.detach_pos = None
        self.detach_quat = None
        self.resume_pos = None
        self.resume_quat = None
        
        # Gripper support
        self.utilize_gripper = utilize_gripper
        if self.utilize_gripper:
            assert self.data_shape[0] == 8, f"Gripper enabled requires data_shape[0] == 8, got {self.data_shape[0]}"
        else:
            assert self.data_shape[0] == 7, f"No gripper requires data_shape[0] == 7, got {self.data_shape[0]}"
        
        # Initialize gripper data storage
        self.gripper_data = 0.0
        # Initialize joint angle processor
        from utils.exoskeleton_utils import create_processor_for_arm
        self.angle_processor = create_processor_for_arm(arm_type, num_joints, data_dtype, airexo_type)
        self.scalar_first_quat_ctrl = scalar_first_quat_ctrl
        
        # Initialize kinematics if EEF is enabled
        if self.ctrl_type in ["eef_delta", "eef_abs"]:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from utils.exoskeleton_utils import ExoskeletonKinematics
                self.kinematics = ExoskeletonKinematics(arm_type, airexo_type)
                logger.info(f"Exoskeleton kinematics model initialized for EEF output with airexo_type: {airexo_type}.")
            except ImportError as e:
                logger.warning(f"Could not import kinematics module for EEF output: {e}")
                self.ctrl_type = "joint"
            
        # Current joint angles (initialize to zeros)
        self.current_angles = [0.0] * num_joints
        self.home_pos = None
        self.home_rot = None
        self.home_j = None
        
        # Encoder instance
        self.encoder = None
        self.encoder_connected = False
        
        # Threading
        self.running = False
        self.lock = threading.Lock()
        
        logger.info(f"Air Exoskeleton device initialized")
        logger.info(f"Device ID: {device_id}")
        logger.info(f"Number of joints: {num_joints}")
        logger.info(f"Port: {port}")
        logger.info(f"Baudrate: {baudrate}")
        logger.info(f"Encoder IDs: {self.encoder_ids}")
        
    def _test_port(self) -> bool:
        """Test if the serial port is accessible."""
        try:
            import serial
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                logger.info(f"Port {self.port} is accessible")
                return True
        except Exception as e:
            logger.error(f"Port {self.port} test failed: {e}")
            return False
    
    def _connect_encoder(self) -> bool:
        """Connect to the angle encoder."""
        max_wait_time = 3  # Increase wait time to 3 seconds
        max_retries = 2    # Add retry mechanism
        if not EXOSKELETON_AVAILABLE:
            logger.error("exoskeleton module not available")
            return False
            
        # Test port accessibility first
        if not self._test_port():
            logger.error(f"Cannot access port {self.port}")
            return False
            
        for retry in range(max_retries):
            try:
                if retry > 0:
                    logger.info(f"Retry {retry + 1}/{max_retries} connecting to encoder on {self.port}")
                    time.sleep(1)  # Wait between retries
                
                self.encoder = AngleEncoder(
                    ids=self.encoder_ids,
                    port=self.port,
                    baudrate=self.baudrate
                )
                
                # Test connection by reading once
                test_result = self.encoder.get_info()
                if test_result is not None:
                    state = test_result.get('state', None)
                    logger.info(f"Initial encoder state: {state}, full result: {test_result}")
                    
                    if state == 1:  # OK status
                        self.encoder_connected = True
                        logger.info(f"Encoder connected successfully on {self.port}")
                        logger.info(f"Initial reading: {test_result}")
                        # Use angle processor to process home joint angles
                        self.home_j = self.angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                        if self.arm_type == "left":
                            self.home_pos, self.home_rot = self.kinematics.forward_kinematics_left(self.home_j.tolist())
                        elif self.arm_type == "right":
                            self.home_pos, self.home_rot = self.kinematics.forward_kinematics_right(self.home_j.tolist())
                        return True
                    else:
                        logger.info(f"Waiting for encoder to become ready (current state: {state})...")
                        t_s = time.time()
                        while time.time() < t_s + max_wait_time:
                            test_result = self.encoder.get_info()
                            if test_result is not None and isinstance(test_result, dict):
                                state = test_result.get('state', None)
                                if state == 1:
                                    self.encoder_connected = True
                                    logger.info(f"Encoder connected successfully on {self.port}")
                                    logger.info(f"Final reading: {test_result}")
                                    # Use angle processor to process home joint angles
                                    self.home_j = self.angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                    if self.arm_type == "left":
                                        self.home_pos, self.home_rot = self.kinematics.forward_kinematics_left(
                                            self.home_j.tolist())
                                    elif self.arm_type == "right":
                                        self.home_pos, self.home_rot = self.kinematics.forward_kinematics_right(
                                            self.home_j.tolist())
                                    return True
                            time.sleep(0.2)  # Slightly longer delay
                        
                        logger.warning(f"Encoder not ready after {max_wait_time}s wait")
                        logger.warning(f"Final state: {state} (expected 1 for OK)")
                        logger.warning(f"Port: {self.port}, Baudrate: {self.baudrate}, IDs: {self.encoder_ids}")
                        if test_result:
                            logger.warning(f"Final test result: {test_result}")
                else:
                    logger.error("Failed to get initial reading from encoder")
                    
            except Exception as e:
                logger.error(f"Error connecting to encoder (attempt {retry + 1}): {e}")
                if retry == max_retries - 1:  # Last retry
                    return False
                    
        return False
    
    def _disconnect_encoder(self) -> None:
        """Disconnect from the angle encoder."""
        if self.encoder:
            try:
                self.encoder.stop_streaming()
                self.encoder = None
                self.encoder_connected = False
                logger.info("Encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting encoder: {e}")
    
    def _read_joint_angles(self) -> np.ndarray:
        """
        Read current joint angles from encoder.
        
        Returns:
            numpy.ndarray: Current joint angles
        """
        if not self.encoder_connected or not self.encoder:
            return self.angle_processor.get_zero_array()
        
        try:
            # Read angles from encoder - returns dict with 'state' and 'joint_pose'
            encoder_info = self.encoder.get_info()
            
            if encoder_info is not None and isinstance(encoder_info, dict):
                state = encoder_info.get('state', None)
                joint_pose = encoder_info.get('joint_pose', None)
                
                # Check if state is OK (assuming 1 means OK based on dual arm implementation)
                if state == 1 and joint_pose is not None and len(joint_pose) >= self.num_joints:
                    # Process joint angles (excluding gripper data if present)
                    if self.utilize_gripper and len(joint_pose) >= 8:
                        # Extract joint angles (first 7) and gripper data (last 1)
                        joint_data = joint_pose[:7]
                        gripper_data = joint_pose[7]
                        processed_angles = self.angle_processor.process_raw_angles(joint_data)
                        # Process gripper data using process_gripper function
                        from utils.exoskeleton_utils import process_gripper
                        self.gripper_data = process_gripper(gripper_data, self.arm_type, self.airexo_type)
                        logger.debug(f"Processed gripper data: raw={gripper_data}, processed={self.gripper_data}")
                    else:
                        # Process all data as joint angles
                        processed_angles = self.angle_processor.process_raw_angles(joint_pose)
                    
                    # Combine data based on gripper usage (following DualAirexoskeleton pattern)
                    if not self.utilize_gripper:
                        result = processed_angles
                    else:
                        # Add gripper data if enabled
                        assert hasattr(self, 'gripper_data'), "Gripper data not found"
                        result = np.concatenate([processed_angles, [self.gripper_data]])
                    
                    with self.lock:
                        self.current_angles = result.tolist()
                    
                    return result.copy()
                else:
                    # Invalid state or data, set to zero
                    zero_array = self.angle_processor.get_zero_array()
                    with self.lock:
                        self.current_angles = zero_array.tolist()
                    if state != 1:
                        logger.error(f"Encoder state error: {state}")
                    return zero_array
            else:
                logger.warning("Invalid response format from encoder (expected dict with 'state' and 'joint_pose')")
                return self.angle_processor.get_zero_array()
                
        except Exception as e:
            logger.error(f"Error reading joint angles: {e}")
            return self.angle_processor.get_zero_array()
    
    def _compute_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose
        
        Args:
            joint_angles: Joint angle array (may include gripper data)
            
        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw] + [gripper_width] if gripper enabled
        """
        if not self.ctrl_type in ["eef_abs","eef_delta"] or not hasattr(self, 'kinematics'):
            if self.utilize_gripper:
                return np.zeros(8, dtype=get_dtype(self.data_dtype))  # 7 elements for pos + quat + 1 for gripper
            else:
                return np.zeros(7, dtype=get_dtype(self.data_dtype))  # 7 elements for pos + quat
        
        try:
            # Extract joint angles (first 7 elements) for kinematics computation
            # joint_angles may include gripper data as the last element
            if self.utilize_gripper and len(joint_angles) > 7:
                joint_angles_for_kinematics = joint_angles[:7]
            else:
                joint_angles_for_kinematics = joint_angles
            
            # Use kinematics to compute end-effector pose
            if self.arm_type == "left":
                pos, rot = self.kinematics.forward_kinematics_left(joint_angles_for_kinematics.tolist())
            elif self.arm_type == "right":
                pos, rot = self.kinematics.forward_kinematics_right(joint_angles_for_kinematics.tolist())
            else:
                raise ValueError(f"Invalid arm_type: {self.arm_type}")
            
            # Convert rotation matrix to quaternion
            quaternion = R.from_matrix(rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                quaternion = np.roll(quaternion, 1)

            if self.ctrl_type == "eef_delta":
                # Handle button state transitions
                if self.utilize_button:
                    # Check state transitions
                    if self.last_state and not self.state:  # running -> paused
                        if self.detach_pos is not None:
                            self.home_rot = (R.from_quat(self.resume_quat) * R.from_quat(self.detach_quat).inv() * R.from_matrix(self.home_rot)).as_matrix()
                            if self.arm_type == "left":
                                self.home_pos = self.home_pos + self.resume_pos - self.detach_pos
                            else:
                                self.home_pos = self.home_pos + self.resume_pos - self.detach_pos

                        self.detach_pos = pos.copy()
                        self.detach_quat = quaternion.copy()
                        self.detach_time = time.time()  # Record detach time

                        self.resume_pos = None
                        self.resume_quat = None
                        logger.info(f"Recorded detach pose: pos={self.detach_pos}, quat={self.detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        detach_dt = time.time() - self.detach_time
                        # Check cooldown time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.resume_pos = pos.copy()
                            self.resume_quat = quaternion.copy()
                            logger.info(f"Recorded resume pose: pos={self.resume_pos}, quat={self.resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Resume cooldown active, {remaining_time:.2f}s remaining")
                    # Use compute_deviation_3 after resume if we have all poses
                    if (self.detach_pos is not None and self.resume_pos is not None):
                        home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                        _ , quaternion = compute_deviation_3(
                            None, home_rot_quat,
                            None, self.detach_quat,
                            None, self.resume_quat,
                            None, quaternion
                        )
                        if self.arm_type == "left":
                            pos = self.kinematics.wb_R_left.apply(pos - self.home_pos + self.detach_pos - self.resume_pos)
                        else:
                            pos = self.kinematics.wb_R_right.apply(pos - self.home_pos + self.detach_pos - self.resume_pos)
                    else:
                        # Fall back to original compute_deviation
                        home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                        _ , quaternion = compute_deviation(None, home_rot_quat, None, quaternion, first_inv=True)
                        pos = self.kinematics.wb_R_left.apply(pos - self.home_pos) if self.arm_type == "left" else self.kinematics.wb_R_right.apply(pos - self.home_pos)
                else:
                    # Normal computation without button
                    home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                    _ , quaternion = compute_deviation(None, home_rot_quat, None, quaternion, first_inv=True)
                    pos = self.kinematics.wb_R_left.apply(pos - self.home_pos) if self.arm_type == "left" else self.kinematics.wb_R_right.apply(pos - self.home_pos)

            # Combine position and quaternion
            eef_pose = np.concatenate([pos, quaternion])
            
            # Add gripper data if enabled (following DualAirexoskeleton pattern)
            if not self.utilize_gripper:
                # Return pose without gripper data
                return eef_pose.astype(get_dtype(self.data_dtype))
            else:
                # Add gripper data if enabled
                assert hasattr(self, 'gripper_data'), "Gripper data not found"
                logger.debug(f"Using self.gripper_data: {self.gripper_data}")
                eef_pose = np.concatenate([eef_pose, [self.gripper_data]])
                return eef_pose.astype(get_dtype(self.data_dtype))
            
        except Exception as e:
            logger.error(f"Error computing end-effector pose: {e}")
            return np.zeros(7, dtype=get_dtype(self.data_dtype))
    
    def start_server(self) -> None:
        """Start the air exoskeleton device server."""
        if self.running:
            logger.info(f"Air Exoskeleton device {self.device_id} is already running")
            return
        
        logger.info(f"Starting Air Exoskeleton device {self.device_id} server...")
        self.running = True
        
        # Start button listener if enabled
        if self.utilize_button:
            self.button.start_listener()
            logger.info(f"Button listener started for key: {self.button.key}")
        
        # Connect to encoder
        if not self._connect_encoder():
            raise RuntimeError("Failed to connect to exoskeleton encoder.")
        
        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        

        
        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        
        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main data reading loop
        while self.running:
            try:
                self.last_state = self.state
                # Check button press and update state
                if self.utilize_button:
                    if self.button.is_pressed():
                        logger.info(f"Button pressed: {'PAUSED' if not self.state else 'RUNNING'}")
                        self.state = not self.state
                # Read current joint angles
                angle_array = self._read_joint_angles() # range from -pi to pi
                # print(f"angle_array: {angle_array}")
                timestamp_ns = time.time_ns()
                if not self.utilize_button or (self.utilize_button and (self.state or self.last_state)):
                    # Write end-effector pose to SHM if EEF control is enabled
                    if self.ctrl_type in ["eef_delta", "eef_abs"]:
                        eef_pose = self._compute_end_effector_pose(angle_array)
                        logger.debug(f"Writing EEF pose to SHM: {eef_pose}, length: {len(eef_pose)}")
                        self._write_array_to_shm_with_timestamp(eef_pose, timestamp_ns)
                        # print(f"eef_pose: {eef_pose}")
                    else:
                        # Write joint angles to SHM (original behavior)
                        self._write_array_to_shm_with_timestamp(angle_array, timestamp_ns)
                
                # Wait for next iteration using precise timing
                wait_for_next_iteration()
                
            except Exception as e:
                logger.error(f"Error in data reading: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the air exoskeleton device server."""
        if not self.running:
            return
        
        logger.info(f"Stopping Air Exoskeleton device {self.device_id} server...")
        self.running = False
        
        # Stop button listener if enabled
        if self.utilize_button:
            self.button.stop_listener()
            logger.info("Button listener stopped")
        
        # Disconnect encoder
        self._disconnect_encoder()
        
        self._cleanup_shared_memory()
        logger.info("Server stopped")
    
    def get_device_info(self) -> Dict[str,Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "num_joints": self.num_joints,
            "port": self.port,
            "baudrate": self.baudrate,
            "encoder_ids": self.encoder_ids,
            "current_angles": self.current_angles.copy(),
            "encoder_connected": self.encoder_connected
        })
        return info


class DualAirexoskeleton(BaseDevice):
    """
    Dual air exoskeleton device for reading joint angle data from two physical exoskeletons.
    Reads joint angles from both arms and publishes to shared memory.
    """
    
    def __init__(self, device_id=0, fps=100.0, left_arm_joints=7, right_arm_joints=7,
                 data_dtype=np.float64, buffer_size=100, hardware_latency_ms=5.0,
                 left_port="/dev/ttyUSB0", right_port="/dev/ttyUSB1", 
                 baudrate=1000000, left_encoder_ids=None, right_encoder_ids=None,
                 ctrl_type = "joint",scalar_first_quat_ctrl = False,
                 utilize_gripper=False, utilize_button=False, button_key='space',
                 airexo_type="ver1"):
        """
        Initialize the dual air exoskeleton device.
        
        Args:
            device_id: Unique identifier for this device instance
            fps: Update frequency in Hz (default: 100.0 to match encoder refresh rate)
            left_arm_joints: Number of joints in left arm (default: 7)
            right_arm_joints: Number of joints in right arm (default: 7)
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            left_port: Serial port for left arm encoder (default: /dev/ttyUSB0)
            right_port: Serial port for right arm encoder (default: /dev/ttyUSB1)
            baudrate: Baudrate for serial communication (default: 1000000)
            left_encoder_ids: List of encoder IDs for left arm (default: [1,2,3,4,5,6,7])
            right_encoder_ids: List of encoder IDs for right arm (default: [1,2,3,4,5,6,7])
            ctrl_type: Control type ("joint", "eef_delta", "eef_abs")
            scalar_first_quat_ctrl: Whether to use scalar first quaternion control
            utilize_gripper: Whether to enable gripper support
            utilize_button: Whether to enable button control
            button_key: Key for button control
            airexo_type: Version of airexo ("ver1" or "ver2")
        """
        # Initialize base device with joint angle data for both arms
        total_joints = left_arm_joints + right_arm_joints
        super().__init__(device_id=device_id, data_shape=(total_joints,) if not utilize_gripper else (total_joints+2,), fps=fps,
                        data_dtype=data_dtype, buffer_size=buffer_size, 
                        hardware_latency_ms=hardware_latency_ms)
        
        # Override device name
        self.device_name = "DualAirexoskeleton"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Exoskeleton attributes
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.total_joints = total_joints
        self.left_port = left_port
        self.right_port = right_port
        self.baudrate = baudrate
        
        # Encoder IDs
        self.left_encoder_ids = left_encoder_ids if left_encoder_ids else list(range(1, left_arm_joints + 1))
        self.right_encoder_ids = right_encoder_ids if right_encoder_ids else list(range(1, right_arm_joints + 1))
        
        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint", "eef_delta", "eef_abs"]
        
        # Gripper support
        self.utilize_gripper = utilize_gripper
        if self.utilize_gripper:
            assert self.data_shape[0] == 16, f"Gripper enabled requires data_shape[0] == 16, got {self.data_shape[0]}"
        else:
            assert self.data_shape[0] == 14, f"No gripper requires data_shape[0] == 14, got {self.data_shape[0]}"
            
        # Store airexo type
        self.airexo_type = airexo_type
        
        # Initialize joint angle processors for each arm separately
        from utils.exoskeleton_utils import create_processor_for_arm
        self.left_angle_processor = create_processor_for_arm("left", 7, data_dtype, airexo_type)
        self.right_angle_processor = create_processor_for_arm("right", 7, data_dtype, airexo_type)
        self.scalar_first_quat_ctrl = scalar_first_quat_ctrl
        
        # Initialize kinematics if EEF is enabled
        if self.ctrl_type in ["eef_delta", "eef_abs"]:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from utils.exoskeleton_utils import ExoskeletonKinematics
                self.kinematics = ExoskeletonKinematics("all", airexo_type)
                logger.info(f"Dual arm exoskeleton kinematics model initialized for EEF output with airexo_type: {airexo_type}.")
            except ImportError as e:
                logger.warning(f"Could not import kinematics module for dual EEF output: {e}")
                self.ctrl_type = "joint"
        
        # Current joint angles (initialize to zeros)
        self.left_current_angles = [0.0] * left_arm_joints
        self.right_current_angles = [0.0] * right_arm_joints
        
        # Gripper data storage
        self.left_gripper_data = 0.0
        self.right_gripper_data = 0.0
        self.home_pos_left = None
        self.home_rot_left = None
        self.home_j_left = None
        self.home_pos_right = None
        self.home_rot_right = None
        self.home_j_right = None
        
        # Encoder instances
        self.left_encoder = None
        self.right_encoder = None
        self.left_encoder_connected = False
        self.right_encoder_connected = False
        
        # Button control configuration
        self.utilize_button = utilize_button
        self.button_key = button_key
        
        # State management (always present, independent of utilize_button)
        self.state = True  # True = running, False = paused
        self.last_state = True
        self.detach_time = None  # Time when detach occurred
        self.resume_cooldown = 1.0  # Minimum time (seconds) between detach and resume
        
        # Pose storage for smooth transitions (always present)
        self.left_detach_pos = None
        self.left_detach_quat = None
        self.left_resume_pos = None
        self.left_resume_quat = None
        
        self.right_detach_pos = None
        self.right_detach_quat = None
        self.right_resume_pos = None
        self.right_resume_quat = None
        
        # Initialize button only if enabled
        if self.utilize_button:
            # Assert that ctrl_type must be eef_delta for button control
            assert self.ctrl_type == "eef_delta", f"Button control requires ctrl_type='eef_delta', got '{self.ctrl_type}'"
            
            # Import and initialize button
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.button import Button
            self.button = Button(self.button_key)
            
            logger.info(f"Button control enabled with key: {self.button_key}")
        
        # Threading
        self.running = False
        self.lock = threading.Lock()
        self.bias = np.array([126.03, 251.19, 323.26, 281.4, 314.38, 114.15, 61.52, 352.52, 152.57, 72.15, 352.4, 277.20, 358.7, 272.8])
        
        logger.info(f"Dual Air Exoskeleton device initialized")
        logger.info(f"Device ID: {device_id}")
        logger.info(f"Left arm joints: {left_arm_joints}")
        logger.info(f"Right arm joints: {right_arm_joints}")
        logger.info(f"Total joints: {total_joints}")
        logger.info(f"Left port: {left_port}")
        logger.info(f"Right port: {right_port}")
        logger.info(f"Baudrate: {baudrate}")
        logger.info(f"Left encoder IDs: {self.left_encoder_ids}")
        logger.info(f"Right encoder IDs: {self.right_encoder_ids}")
        
    def _connect_encoders(self) -> Tuple[bool,bool]:
        """Connect to both angle encoders."""
        max_wait_time = 3  # Increased wait time for encoder initialization
        if not EXOSKELETON_AVAILABLE:
            logger.error("exoskeleton module not available")
            return False, False
            
        left_connected = False
        right_connected = False
        
        # Connect left encoder
        try:
            self.left_encoder = AngleEncoder(
                ids=self.left_encoder_ids,
                port=self.left_port,
                baudrate=self.baudrate
            )
            # Test connection by reading once
            test_result = self.left_encoder.get_info()
            if test_result is not None and isinstance(test_result, dict):
                state = test_result.get('state', None)
                if state == 1:
                    self.left_encoder_connected = True
                    left_connected = True
                    logger.info(f"Left encoder connected successfully on {self.left_port}")
                    logger.info(f"Left initial reading: {test_result}")
                    # Use left arm angle processor for home position
                    self.home_j_left = self.left_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                    self.home_pos_left, self.home_rot_left = self.kinematics.forward_kinematics_left(
                        self.home_j_left.tolist())
         
                else:
                    t_s = time.time()
                    while time.time() < t_s + max_wait_time:
                        test_result = self.left_encoder.get_info()
                        if test_result is not None and isinstance(test_result, dict):
                            state = test_result.get('state', None)
                            if state == 1:
                                left_connected = True
                                self.left_encoder_connected = True
                                logger.info(f"Left encoder connected successfully on {self.left_port}")
                                logger.info(f"Left initial reading: {test_result}")
                                # Use left arm angle processor for home position
                                self.home_j_left = self.left_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                self.home_pos_left, self.home_rot_left = self.kinematics.forward_kinematics_left(self.home_j_left.tolist())
      
                                break
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    if not self.left_encoder_connected:
                        logger.warning(f"Left encoder not ready, state: {state} (expected 1 for OK)")
            else:
                logger.error("Failed to get initial reading from left encoder")
                
        except Exception as e:
            logger.error(f"Error connecting to left encoder: {e}")
        
        # Connect right encoder
        try:
            self.right_encoder = AngleEncoder(
                ids=self.right_encoder_ids,
                port=self.right_port,
                baudrate=self.baudrate
            )
            # Test connection by reading once
            test_result = self.right_encoder.get_info()
            if test_result is not None and isinstance(test_result, dict):
                state = test_result.get('state', None)
                if state == 1:
                    self.right_encoder_connected = True
                    right_connected = True
                    logger.info(f"Right encoder connected successfully on {self.right_port}")
                    logger.info(f"Right initial reading: {test_result}")
                    # Use right arm angle processor for home position
                    self.home_j_right = self.right_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                    self.home_pos_right, self.home_rot_right = self.kinematics.forward_kinematics_right(
                        self.home_j_right.tolist())
                else:
                    t_s = time.time()
                    while time.time() < t_s + max_wait_time:
                        test_result = self.right_encoder.get_info()
                        if test_result is not None and isinstance(test_result, dict):
                            state = test_result.get('state', None)
                            if state == 1:
                                right_connected = True
                                self.right_encoder_connected = True
                                logger.info(f"Right encoder connected successfully on {self.right_port}")
                                logger.info(f"Right initial reading: {test_result}")
                                # Use right arm angle processor for home position
                                self.home_j_right = self.right_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                self.home_pos_right, self.home_rot_right = self.kinematics.forward_kinematics_right(
                                    self.home_j_right.tolist())
                                break
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    if not self.right_encoder_connected:
                        logger.warning(f"Right encoder not ready, state: {state} (expected 1 for OK)")
            else:
                logger.error("Failed to get initial reading from right encoder")
                
        except Exception as e:
            logger.error(f"Error connecting to right encoder: {e}")
        
        return left_connected, right_connected
    
    def _disconnect_encoders(self) -> None:
        """Disconnect from both angle encoders."""
        # Disconnect left encoder
        if self.left_encoder:
            try:
                self.left_encoder.stop_streaming()
                self.left_encoder = None
                self.left_encoder_connected = False
                logger.info("Left encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting left encoder: {e}")
        
        # Disconnect right encoder
        if self.right_encoder:
            try:
                self.right_encoder.stop_streaming()
                self.right_encoder = None
                self.right_encoder_connected = False
                logger.info("Right encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting right encoder: {e}")
    
    def _read_joint_angles(self) -> None:
        """
        Read current joint angles from both encoders.
        
        Returns:
            numpy.ndarray: Current joint angles [left_arm_joints, right_arm_joints]
        """
        left_angles = np.zeros(self.left_arm_joints, dtype=get_dtype(self.data_dtype))
        right_angles = np.zeros(self.right_arm_joints, dtype=get_dtype(self.data_dtype))
        
        # Read left encoder
        if self.left_encoder_connected and self.left_encoder:
            try:
                encoder_info = self.left_encoder.get_info()
                if encoder_info is not None and isinstance(encoder_info, dict):
                    state = encoder_info.get('state', None)
                    joint_pose = encoder_info.get('joint_pose', None)
                    
                    # Check if state is OK (assuming 1 means OK based on the example)
                    if state == 1 and joint_pose is not None and len(joint_pose) >= self.left_arm_joints:
                        # Process left arm joint angles (excluding gripper data if present)
                        if self.utilize_gripper and len(joint_pose) >= 8:
                            # Extract joint angles (first 7) and gripper data (last 1)
                            left_joint_data = joint_pose[:7]
                            raw_left_gripper_data = joint_pose[7]
                            left_processed = self.left_angle_processor.process_raw_angles(left_joint_data)
                            # Process gripper data using process_gripper function
                            from utils.exoskeleton_utils import process_gripper
                            self.left_gripper_data = process_gripper(raw_left_gripper_data, "left", self.airexo_type)
                        else:
                            # Process all data as joint angles
                            left_processed = self.left_angle_processor.process_raw_angles(joint_pose)
                        
                        with self.lock:
                            self.left_current_angles = left_processed.tolist()
                            left_angles[:] = left_processed
                    else:
                        # Invalid state or data, set to zero
                        with self.lock:
                            zero_array = np.zeros(self.left_arm_joints, dtype=get_dtype(self.data_dtype))
                            self.left_current_angles = zero_array.tolist()
                            left_angles.fill(0.0)
                        if state != 1:
                            logger.error(f"Left encoder state error: {state}")
            except Exception as e:
                logger.error(f"Error reading left encoder: {e}")
        
        # Read right encoder
        if self.right_encoder_connected and self.right_encoder:
            try:
                encoder_info = self.right_encoder.get_info()
                if encoder_info is not None and isinstance(encoder_info, dict):
                    state = encoder_info.get('state', None)
                    joint_pose = encoder_info.get('joint_pose', None)
                    
                    # Check if state is OK (assuming 1 means OK based on the example)
                    if state == 1 and joint_pose is not None and len(joint_pose) >= self.right_arm_joints:
                        # Process right arm joint angles (excluding gripper data if present)
                        if self.utilize_gripper and len(joint_pose) >= 8:
                            # Extract joint angles (first 7) and gripper data (last 1)
                            right_joint_data = joint_pose[:7]
                            raw_right_gripper_data = joint_pose[7]
                            right_processed = self.right_angle_processor.process_raw_angles(right_joint_data)
                            # Process gripper data using process_gripper function
                            from utils.exoskeleton_utils import process_gripper
                            self.right_gripper_data = process_gripper(raw_right_gripper_data, "right", self.airexo_type)
                        else:
                            # Process all data as joint angles
                            right_processed = self.right_angle_processor.process_raw_angles(joint_pose)
                        
                        with self.lock:
                            self.right_current_angles = right_processed.tolist()
                            right_angles[:] = right_processed
                    else:
                        # Invalid state or data, set to zero
                        with self.lock:
                            zero_array = np.zeros(self.right_arm_joints, dtype=get_dtype(self.data_dtype))
                            self.right_current_angles = zero_array.tolist()
                            right_angles.fill(0.0)
                        if state != 1:
                            logger.error(f"Right encoder state error: {state}")
            except Exception as e:
                logger.error(f"Error reading right encoder: {e}")
        
        # Combine both arms' data
        if not self.utilize_gripper:
            all_angles = np.concatenate([left_angles, right_angles])
        else:
            # Add gripper data if enabled
            assert hasattr(self, 'left_gripper_data') and hasattr(self, 'right_gripper_data'), "Gripper data not found"
            all_angles = np.concatenate([left_angles, [self.left_gripper_data], right_angles, [self.right_gripper_data]])
     
        return all_angles
    
    def _compute_dual_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute dual-arm end-effector poses
        
        Args:
            joint_angles: Dual-arm joint angle array [left_arm_joints + right_arm_joints]
            
        Returns:
            Dual-arm end-effector poses [left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw,
                                        right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw]
        """
        if not self.ctrl_type in ["eef_abs","eef_delta"] or not hasattr(self, 'kinematics'):
            if self.utilize_gripper:
                return np.zeros(16, dtype=get_dtype(self.data_dtype))  # 7 elements per arm + 2 gripper
            else:
                return np.zeros(14, dtype=get_dtype(self.data_dtype))  # 7 elements per arm
        
        try:
            # Separate left and right arm joint angles
            left_angles = joint_angles[:self.left_arm_joints]
            right_angles = joint_angles[self.left_arm_joints:] if not self.utilize_gripper else joint_angles[self.left_arm_joints+1:self.left_arm_joints+self.right_arm_joints+1]
            
            # Compute left arm end-effector pose
            left_pos, left_rot = self.kinematics.forward_kinematics_left(left_angles.tolist())
            left_quaternion = R.from_matrix(left_rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                left_quaternion = np.roll(left_quaternion,1)
            if self.ctrl_type == "eef_delta":
                # Handle button state transitions for left arm
                if self.utilize_button:
                    # Check state transitions for left arm
                    if self.last_state and not self.state:  # running -> paused
                        if self.left_detach_pos is not None:
                            # Update home pose using resume and detach poses
                            self.home_rot_left = (R.from_quat(self.left_resume_quat) * R.from_quat(self.left_detach_quat).inv() * R.from_matrix(self.home_rot_left)).as_matrix()
                            self.home_pos_left = self.home_pos_left + self.left_resume_pos - self.left_detach_pos

                        self.left_detach_pos = left_pos.copy()
                        self.left_detach_quat = left_quaternion.copy()
                        self.detach_time = time.time()  # Record detach time

                        self.left_resume_pos = None
                        self.left_resume_quat = None
                        logger.info(f"Recorded left arm detach pose: pos={self.left_detach_pos}, quat={self.left_detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        detach_dt = time.time() - self.detach_time
                        # Check cooldown time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.left_resume_pos = left_pos.copy()
                            self.left_resume_quat = left_quaternion.copy()
                            logger.info(f"Recorded left arm resume pose: pos={self.left_resume_pos}, quat={self.left_resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Resume cooldown active, {remaining_time:.2f}s remaining")
                    
                    # Use compute_deviation_3 after resume if we have all poses for left arm
                    if (self.left_detach_pos is not None and self.left_resume_pos is not None):
                        home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                        _ , left_quaternion = compute_deviation_3(
                            None, home_rot_left_quat,
                            None, self.left_detach_quat,
                            None, self.left_resume_quat,
                            None, left_quaternion
                        )
                        left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left + self.left_detach_pos - self.left_resume_pos)

                    else:
                        # Fall back to original compute_deviation
                        home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                        _ , left_quaternion = compute_deviation(None, home_rot_left_quat, None, left_quaternion, first_inv=True)
                        left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left)
                else:
                    # Normal computation without button
                    home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                    _ , left_quaternion = compute_deviation(None, home_rot_left_quat, None, left_quaternion, first_inv=True)
                    left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left)
            left_pose = np.concatenate([left_pos, left_quaternion])
            
            # Compute right arm end-effector pose
            right_pos, right_rot = self.kinematics.forward_kinematics_right(right_angles.tolist())
            right_quaternion = R.from_matrix(right_rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                right_quaternion = np.roll(right_quaternion,1)
            if self.ctrl_type == "eef_delta":
                # Handle button state transitions for right arm
                if self.utilize_button:
                    # Check state transitions for right arm
                    if self.last_state and not self.state:  # running -> paused
                        if self.right_detach_pos is not None:
                            # Update home pose using resume and detach poses
                            self.home_rot_right = (R.from_quat(self.right_resume_quat) * R.from_quat(self.right_detach_quat).inv() * R.from_matrix(self.home_rot_right)).as_matrix()
                            self.home_pos_right = self.home_pos_right + self.right_resume_pos - self.right_detach_pos
                            
                        self.right_detach_pos = right_pos.copy()
                        self.right_detach_quat = right_quaternion.copy()
                        self.right_resume_pos = None
                        self.right_resume_quat = None
                        # detach_time already set by left arm processing
                        logger.info(f"Recorded right arm detach pose: pos={self.right_detach_pos}, quat={self.right_detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        # Check cooldown time (same as left arm)
                        detach_dt = time.time() - self.detach_time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.right_resume_pos = right_pos.copy()
                            self.right_resume_quat = right_quaternion.copy()
                            logger.info(f"Recorded right arm resume pose: pos={self.right_resume_pos}, quat={self.right_resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Right arm resume cooldown active, {remaining_time:.2f}s remaining")
                    
                    # Use compute_deviation_3 after resume if we have all poses for right arm
                    if (self.right_detach_pos is not None and
                        self.right_resume_pos is not None):
                        home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                        _ , right_quaternion = compute_deviation_3(
                            None, home_rot_right_quat,
                            None, self.right_detach_quat,
                            None, self.right_resume_quat,
                            None, right_quaternion
                        )
                        right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right + self.right_detach_pos - self.right_resume_pos)

                    else:
                        # Fall back to original compute_deviation
                        home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                        _ , right_quaternion = compute_deviation(None, home_rot_right_quat, None, right_quaternion, first_inv=True)
                        right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right)
                else:
                    # Normal computation without button
                    home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                    _ , right_quaternion = compute_deviation(None, home_rot_right_quat, None, right_quaternion, first_inv=True)
                    right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right)
            right_pose = np.concatenate([right_pos, right_quaternion])
            
            # Combine dual-arm poses
            if not self.utilize_gripper:
                # Combine dual-arm poses [left_pose(7) + right_pose(7)]
                dual_pose = np.concatenate([left_pose, right_pose])
            else:
                # Add gripper data if enabled
                assert hasattr(self, 'left_gripper_data') and hasattr(self, 'right_gripper_data'), "Gripper data not found"
                dual_pose = np.concatenate([left_pose, [self.left_gripper_data], right_pose, [self.right_gripper_data]])
            
            return dual_pose.astype(get_dtype(self.data_dtype))
            
        except Exception as e:
            logger.error(f"Error computing dual end-effector pose: {e}")
            return np.zeros(14, dtype=get_dtype(self.data_dtype))
    
    
    def start_server(self):
        """Start the dual air exoskeleton device server."""
        if self.running:
            logger.info(f"Dual Air Exoskeleton device {self.device_id} is already running")
            return
        
        logger.info(f"Starting Dual Air Exoskeleton device {self.device_id} server...")
        self.running = True
        
        # Start button listener if enabled
        if self.utilize_button:
            self.button.start_listener()
            logger.info(f"Button listener started for key: {self.button.key}")
        
        # Connect to encoders
        left_connected, right_connected = self._connect_encoders()
        if left_connected and right_connected:
            logger.info("Both encoders connected successfully.")
        else:
            if not left_connected and right_connected:
                logger.error("Failed to connect to left encoder.")
            elif not right_connected and left_connected:
                logger.error("Failed to connect to right encoder.")
            else:
                logger.error("Failed to connect to any encoder.")
            raise RuntimeError("Failed to connect to all exoskeleton encoders.")

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        

        
        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        
        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main data reading loop
        while self.running:
            try:
                self.last_state = self.state
                # Check button press and update state
                if self.utilize_button:
                    if self.button.is_pressed():
                        logger.info(f"Button pressed: {'PAUSED' if not self.state else 'RUNNING'}")
                        self.state = not self.state
                    else:
                        self.state = self.last_state
                
                # Read current joint angles
                angle_array = self._read_joint_angles()
                
                timestamp_ns = time.time_ns()
              
                if not self.utilize_button or (self.utilize_button and (self.state or self.last_state)):
                    # Write dual end-effector pose to SHM if EEF control is enabled
                    if self.ctrl_type in ["eef_abs","eef_delta"]:
                        dual_eef_pose = self._compute_dual_end_effector_pose(angle_array)
                        self._write_array_to_shm_with_timestamp(dual_eef_pose, timestamp_ns)
                    else:
                        # Write joint angles to SHM (original behavior)
                        self._write_array_to_shm_with_timestamp(angle_array, timestamp_ns)
                
                # Wait for next iteration using precise timing
                wait_for_next_iteration()
                
            except Exception as e:
                logger.error(f"Error in data reading: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the dual air exoskeleton device server."""
        if not self.running:
            return
        
        logger.info(f"Stopping Dual Air Exoskeleton device {self.device_id} server...")
        self.running = False
        
        # Stop button listener if enabled
        if self.utilize_button:
            self.button.stop_listener()
            logger.info("Button listener stopped")
        
        # Disconnect encoders
        self._disconnect_encoders()
        
        self._cleanup_shared_memory()
        logger.info("Server stopped")
    
    def get_device_info(self) -> Dict[str,Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "left_arm_joints": self.left_arm_joints,
            "right_arm_joints": self.right_arm_joints,
            "total_joints": self.total_joints,
            "left_port": self.left_port,
            "right_port": self.right_port,
            "baudrate": self.baudrate,
            "left_encoder_ids": self.left_encoder_ids,
            "right_encoder_ids": self.right_encoder_ids,
            "left_current_angles": self.left_current_angles.copy(),
            "right_current_angles": self.right_current_angles.copy(),
            "left_encoder_connected": self.left_encoder_connected,
            "right_encoder_connected": self.right_encoder_connected
        })
        return info


def main():
    """Main function to run the joint slider teleoperator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Joint Slider Teleoperator for Robot Control")
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help="Device ID (default: 0)")
    parser.add_argument("--fps", "-f", type=float, default=10.0,
                        help="Update frequency in Hz (default: 10.0)")
    parser.add_argument("--num-joints", "-n", type=int, default=7,
                        help="Number of joints to control (default: 7)")
    parser.add_argument("--dtype", "-t", default="float64",
                        choices=['float32', 'float64'],
                        help="Data type (default: float64)")
    parser.add_argument("--buffer-size", "-b", type=int, default=100,
                        help="Buffer size in frames (default: 100)")
    
    args = parser.parse_args()
    
    # Import common utilities
    from utils.shm_utils import get_dtype
    
    data_dtype = get_dtype(args.dtype)
    
    # Create teleoperator
    teleop = JointSlider(
        device_id=args.device_id,
        fps=args.fps,
        num_joints=args.num_joints,
        data_dtype=data_dtype,
        buffer_size=args.buffer_size
    )
    
    logger.info("Joint Slider Teleoperator")
    logger.info("========================")
    logger.info(f"Device ID: {args.device_id}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Number of joints: {args.num_joints}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info("")
    
    try:
        logger.info("Teleoperator is running. Use sliders to control joint positions...")
        logger.info(f"Device info: {teleop.get_device_info()}")
        teleop.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        teleop.stop_server()


if __name__ == "__main__":
    main() 
