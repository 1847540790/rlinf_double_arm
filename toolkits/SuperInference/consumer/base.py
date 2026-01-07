#!/usr/bin/env python3
"""
Base Consumer - Base class for consuming device manager data.

This module provides the base functionality for reading data from the device manager's
summary shared memory and processing it.

Author: Jun Lv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import signal
import multiprocessing.shared_memory as shm
import struct
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from utils.shm_utils import (
    unpack_manager_header, unpack_device_header, unpack_frame_header, unpack_summary_frame_header,
    get_device_info_offset, get_data_offset, get_summary_frame_offset, get_summary_frame_start_offset,
    MANAGER_HEADER_SIZE, DEVICE_HEADER_SIZE, FRAME_HEADER_SIZE, SUMMARY_FRAME_HEADER_SIZE
)
from utils.shm_utils import get_dtype
from utils.logger_config import logger


class BaseConsumer:
    """
    Base class for consuming device manager data.
    
    This class provides the basic functionality for:
    - Connecting to device manager's summary shared memory
    - Parsing device information
    - Reading device data
    - Managing device state
    """
    
    def __init__(self, summary_shm_name: str = "device_summary_data") -> None:
        """
        Initialize the base consumer.
        
        Args:
            summary_shm_name: Name of the summary shared memory
        """
        self.summary_shm_name = summary_shm_name
        self.summary_shm = None
        self.running = False
        
        # Device data storage
        self.devices: Dict[str, Dict[str, Any]] = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def connect(self) -> bool:
        """Connect to the summary shared memory in READ-ONLY mode."""
        try:
            # Use unified SHM interface for read-only connection
            from utils.shm_utils import connect_to_summary_shm
            
            # Connect in read-only mode since consumers should only READ data
            self.summary_shm = connect_to_summary_shm(read_only=True)
            logger.info(f"Connected to summary SHM: {self.summary_shm_name} (READ-ONLY)")
            
            # Parse initial header
            self._parse_summary_header()
            
            return True
                
        except FileNotFoundError:
            logger.error(f"Summary SHM not found: {self.summary_shm_name}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to summary SHM: {e}")
            return False
    
    def _parse_summary_header(self) -> None:
        """Parse the summary SHM header to get device information."""
        try:
            buf = self.summary_shm.buf
            
            # Parse manager header
            manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
            device_count = manager_header['device_count']
            update_timestamp = manager_header['update_timestamp']
            
            logger.info(f"Summary SHM: {device_count} devices")
            
            # Parse device headers
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset+DEVICE_HEADER_SIZE])
                
                # Create device key
                device_key = f"{device_header['device_type']}_{device_header['device_id']}"
                
                # Store device info
                self.devices[device_key] = {
                    'id': device_header['device_id'],
                    'type': device_header['device_type'],
                    'name': device_key,
                    'fps': device_header['fps'],
                    'data_dtype': device_header['data_dtype'],
                    'shape': device_header['shape'],
                    'frame_size': device_header['frame_size'],
                    'hardware_latency_ms': device_header['hardware_latency_ms'],
                    'last_timestamp': None,
                    'last_update_time': None
                }
                
                logger.info(f"Device {device_key}: fps={device_header['fps']}, shape={device_header['shape']}, hardware_latency={device_header['hardware_latency_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error parsing summary header: {e}")
            raise
    
    def _read_device_data(self, device_key: str, frame_index: Optional[int] = None) -> Optional[tuple]:
        """
        Read data for a specific device from summary SHM buffer.
        
        Args:
            device_key: Device key to read data for
            frame_index: Frame index in buffer:
                        None = latest frame (same as -1)
                        -1 = latest frame
                        -2 = second latest frame
                        0 = oldest frame
                        1 = second oldest frame
                        Absolute value cannot be larger than current_frames_count
        """
        if device_key not in self.devices:
            return None
        
        try:
            device_info = self.devices[device_key]
            buf = self.summary_shm.buf
            
            # Read manager header to get buffer state
            manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
            device_count = manager_header['device_count']
            buffer_size = manager_header['buffer_size']
            current_frames_count = manager_header['current_frames_count']
            write_index = manager_header['write_index']
            
            # If no data available, return None
            if current_frames_count == 0:
                return None
            
            # Determine actual buffer index to read
            if frame_index is None:
                # Default to latest frame (-1)
                actual_frame_index = (write_index - 1) % buffer_size
            else:
                # Validate frame_index bounds
                if abs(frame_index) > current_frames_count:
                    logger.warning(f"Frame index {frame_index} is out of bounds. Available frames: {current_frames_count}")
                    return None
                
                if frame_index < 0:
                    # Negative index: -1 = latest, -2 = second latest, etc.
                    actual_frame_index = (write_index + frame_index) % buffer_size
                else:
                    # Positive index: 0 = oldest, 1 = second oldest, etc.
                    if current_frames_count < buffer_size:
                        # Buffer not full yet, oldest is at index 0
                        actual_frame_index = frame_index
                    else:
                        # Buffer is full, oldest is at write_index
                        actual_frame_index = (write_index + frame_index) % buffer_size
            
            # Get device index
            device_keys = list(self.devices.keys())
            device_index = device_keys.index(device_key)
            frame_sizes = [self.devices[key]['frame_size'] for key in device_keys]
            
            # Calculate frame offset for this device at specified frame
            frame_offset = get_summary_frame_offset(device_index, actual_frame_index, device_count, frame_sizes, buffer_size)
            
            # Read frame header
            frame_header = unpack_frame_header(buf[frame_offset:frame_offset+FRAME_HEADER_SIZE])
            timestamp_ns = frame_header['timestamp_ns']
            
            # If timestamp is 0, no data for this device
            if timestamp_ns == 0:
                return None
            
            # Read data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_size = device_info['frame_size'] - FRAME_HEADER_SIZE
            data_end = data_start + data_size
            
            # Convert data based on dtype
            dtype = get_dtype(device_info['data_dtype'], np.uint8)
            
            # Extract data slice (DataChannel.buf returns raw buffer, so slicing works directly)
            sliced_buf = buf[data_start:data_end]
            data_array = np.frombuffer(sliced_buf, dtype=dtype)
            
            # Reshape data
            shape = device_info['shape']
            if len(shape) > 0:
                data_array = data_array.reshape(shape)
            
            return timestamp_ns, data_array
            
        except Exception as e:
            logger.error(f"Error reading device {device_key} data: {e}")
            return None
    
    def read_all_device_data(self, frame_index: Optional[int] = None, include_frame_timestamp: bool = False) -> Optional[Dict[str, Tuple[int, np.ndarray]]]:
        """
        Read all device data from summary SHM buffer.
        
        Args:
            frame_index: Frame index in buffer:
                        None = latest frame (same as -1)
                        -1 = latest frame
                        -2 = second latest frame
                        0 = oldest frame
                        1 = second oldest frame
                        Absolute value cannot be larger than current_frames_count
            include_frame_timestamp: If True, include '_frame_timestamp' key in returned dict
        """
        try:
            if not self.devices:
                return None
            
            # Read all data from summary SHM buffer once
            buf = self.summary_shm.buf
            
            # Read manager header to get buffer state
            manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
            device_count = manager_header['device_count']
            buffer_size = manager_header['buffer_size']
            current_frames_count = manager_header['current_frames_count']
            write_index = manager_header['write_index']
            
            # If no data available, return None
            if current_frames_count == 0:
                return None
            
            # Determine actual buffer index to read
            if frame_index is None:
                # Default to latest frame (-1)
                target_frame_index = (write_index - 1) % buffer_size
            else:
                # Validate frame_index bounds
                if abs(frame_index) > current_frames_count:
                    logger.warning(f"Frame index {frame_index} is out of bounds. Available frames: {current_frames_count}")
                    return None
                
                if frame_index < 0:
                    # Negative index: -1 = latest, -2 = second latest, etc.
                    target_frame_index = (write_index + frame_index) % buffer_size
                else:
                    # Positive index: 0 = oldest, 1 = second oldest, etc.
                    if current_frames_count < buffer_size:
                        # Buffer not full yet, oldest is at index 0
                        target_frame_index = frame_index
                    else:
                        # Buffer is full, oldest is at write_index
                        target_frame_index = (write_index + frame_index) % buffer_size
            
            # Calculate frame boundaries
            device_keys = list(self.devices.keys())
            frame_sizes = [self.devices[key]['frame_size'] for key in device_keys]
            
            # Get frame start offset using utility function
            frame_start_offset = get_summary_frame_start_offset(target_frame_index, device_count, frame_sizes)
            total_frame_size = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes)
            
            # Read entire frame data at once (including frame header)
            frame_data = buf[frame_start_offset:frame_start_offset + total_frame_size]
            
            # Parse frame header to get frame timestamp
            summary_frame_header = unpack_summary_frame_header(frame_data[:SUMMARY_FRAME_HEADER_SIZE])
            frame_timestamp_ns = summary_frame_header['frame_timestamp_ns']
            
            # Parse frame data for each device
            all_data = {}
            if not hasattr(self, 'obs_devices'):
                self.obs_devices = list(self.devices.keys())
            
            # Start after frame header
            current_offset = SUMMARY_FRAME_HEADER_SIZE
            for device_name in self.obs_devices:
                if device_name in self.devices:
                    device_index = device_keys.index(device_name)
                    device_frame_size = frame_sizes[device_index]
                    
                    # Extract this device's data from frame buffer
                    device_data = frame_data[current_offset:current_offset + device_frame_size]
                    
                    # Parse frame header
                    frame_header = unpack_frame_header(device_data[:FRAME_HEADER_SIZE])
                    timestamp_ns = frame_header['timestamp_ns']
                    
                    # If timestamp is 0, no data for this device
                    if timestamp_ns == 0:
                        current_offset += device_frame_size
                        continue
                    
                    # Parse data - use zero-copy for large data
                    data_bytes = device_data[FRAME_HEADER_SIZE:]
                    dtype = get_dtype(self.devices[device_name]['data_dtype'], np.uint8)
                    
                    if len(data_bytes) > 1024:  # Zero-copy for data > 1KB
                        # Calculate absolute offset in the full buffer
                        abs_offset = frame_start_offset + current_offset + FRAME_HEADER_SIZE
                        np_dtype = np.dtype(dtype)
                        # Zero-copy: buf is now raw buffer from DataChannel.buf
                        data_array = np.frombuffer(
                            buf, 
                            dtype=dtype, 
                            count=len(data_bytes)//np_dtype.itemsize, 
                            offset=abs_offset
                        )
                    else:
                        # Copy for small data: data_bytes is raw buffer slice
                        data_array = np.frombuffer(data_bytes, dtype=dtype)
                    
                    shape = self.devices[device_name]['shape']
                    if len(shape) > 0:
                        data_array = data_array.reshape(shape)
                    
                    all_data[device_name] = (timestamp_ns, data_array)
                    
                    # Move to next device's data
                    current_offset += device_frame_size
            
            # Add frame timestamp to result if requested
            if include_frame_timestamp:
                return frame_timestamp_ns, all_data
            else:
                return all_data
            
        except Exception as e:
            logger.error(f"Error reading all device data: {e}")
            return None
    
    def find_frame_by_timestamp(self, target_timestamp_ns: int, tolerance_ms: float = 50.0) -> Optional[Dict[str, Any]]:
        """
        Find frame by timestamp using binary search.
        
        Args:
            target_timestamp_ns: Target frame timestamp in nanoseconds
            tolerance_ms: Maximum acceptable time difference in milliseconds
            
        Returns:
            Dict containing frame data and metadata, or None if not found
        """
        try:
            if not self.summary_shm:
                return None
            
            buf = self.summary_shm.buf
            
            # Read manager header to get buffer state
            manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
            device_count = manager_header['device_count']
            buffer_size = manager_header['buffer_size']
            current_frames_count = manager_header['current_frames_count']
            write_index = manager_header['write_index']
            
            # If no data available, return None
            if current_frames_count == 0:
                return None
            
            device_keys = list(self.devices.keys())
            frame_sizes = [self.devices[key]['frame_size'] for key in device_keys]
            tolerance_ns = int(tolerance_ms * 1_000_000)  # Convert ms to ns
            
            # Determine search range for circular buffer
            if current_frames_count < buffer_size:
                # Buffer not full yet, search from 0 to current_frames_count-1
                start_idx = 0
                end_idx = current_frames_count - 1
            else:
                # Buffer is full, search in circular order starting from oldest frame
                start_idx = write_index  # Oldest frame position
                end_idx = write_index + buffer_size - 1  # Virtual end index
            
            # Binary search for closest timestamp
            best_frame_index = None
            best_timestamp_diff = float('inf')
            
            left, right = start_idx, end_idx
            
            while left <= right:
                mid = (left + right) // 2
                
                # Convert virtual index to actual buffer index
                actual_frame_idx = mid % buffer_size
                
                # Read frame timestamp at actual index
                frame_start_offset = get_summary_frame_start_offset(actual_frame_idx, device_count, frame_sizes)
                summary_frame_header = unpack_summary_frame_header(buf[frame_start_offset:frame_start_offset+SUMMARY_FRAME_HEADER_SIZE])
                frame_timestamp_ns = summary_frame_header['frame_timestamp_ns']
                
                timestamp_diff = abs(frame_timestamp_ns - target_timestamp_ns)
                
                # Update best match if this is closer
                if timestamp_diff < best_timestamp_diff:
                    best_timestamp_diff = timestamp_diff
                    best_frame_index = actual_frame_idx
                
                # Binary search logic
                if frame_timestamp_ns < target_timestamp_ns:
                    left = mid + 1
                elif frame_timestamp_ns > target_timestamp_ns:
                    right = mid - 1
                else:
                    # Exact match found
                    best_frame_index = actual_frame_idx
                    best_timestamp_diff = 0
                    break
            
            # Check if best match is within tolerance
            if best_frame_index is not None and best_timestamp_diff <= tolerance_ns:
                # Read the frame data with frame timestamp
                _frame_timestamp, frame_data = self.read_all_device_data(frame_index=best_frame_index, include_frame_timestamp=True)
                if frame_data:
                    return {
                        'frame_index': best_frame_index,
                        'frame_timestamp_ns': _frame_timestamp,
                        'timestamp_diff_ns': best_timestamp_diff,
                        'timestamp_diff_ms': best_timestamp_diff / 1_000_000,
                        'data': {k: v for k, v in frame_data.items()}
                    }
            
            logger.debug(f"No frame found within tolerance. Best match: diff={best_timestamp_diff/1_000_000:.2f}ms, tolerance={tolerance_ms}ms")
            return None
            
        except Exception as e:
            logger.error(f"Error finding frame by timestamp: {e}")
            return None

    def disconnect(self) -> None:
        """Disconnect from summary SHM and clean up resources."""
        if self.summary_shm:
            try:
                self.summary_shm.close()
                logger.info("Disconnected from summary SHM")
            except Exception as e:
                logger.error(f"Error disconnecting from summary SHM: {e}")
            finally:
                self.summary_shm = None
    
    def stop(self) -> None:
        """Stop the consumer."""
        if not self.running:
            return
        
        logger.info("Stopping consumer")
        self.running = False
        
        # Disconnect from SHM
        self.disconnect()
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.disconnect() 