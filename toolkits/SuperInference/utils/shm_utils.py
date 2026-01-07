#!/usr/bin/env python3
"""
Shared Memory (SHM) utilities for unified SHM format.

Author: Jun Lv
"""

import struct
import time
import multiprocessing.shared_memory as shm
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from utils.logger_config import logger

# Data type mapping for string to numpy dtype conversion
DTYPE_MAP = {
    'uint8': np.uint8, 'int8': np.int8,
    'uint16': np.uint16, 'int16': np.int16,
    'uint32': np.uint32, 'int32': np.int32,
    'float32': np.float32, 'float64': np.float64
}

def get_dtype(dtype_str: str, default: np.dtype = np.float32) -> np.dtype:
    """Convert string dtype to numpy dtype."""
    return DTYPE_MAP.get(dtype_str, default)

# Constants
MAX_SHAPE_DIMENSIONS = 4

# SHM Names
DEVICE_SHM_PREFIX = "{device_type}_{device_id}_data"
SUMMARY_SHM_NAME = "device_summary_data"

# Header Sizes
DEVICE_HEADER_SIZE = 60 + 4 * MAX_SHAPE_DIMENSIONS  # device_type(20) + device_id(4) + fps(4) + data_dtype(20) + num_dims(4) + shape(16) + frame_size(4) + hardware_latency_ms(4)
BUFFER_HEADER_SIZE = 16  # buffer_size(4) + current_frames_count(4) + write_index(4) + padding(4)
MANAGER_HEADER_SIZE = 24  # device_count(4) + buffer_size(4) + current_frames_count(4) + write_index(4) + update_timestamp(8)
POLICY_HEADER_SIZE = 16   # device_count(4) + update_timestamp(8) + padding(4) - for policy SHM (old format)
FRAME_HEADER_SIZE = 8     # timestamp_ns(8) - for individual device data
SUMMARY_FRAME_HEADER_SIZE = 8  # frame_timestamp_ns(8) - for entire frame in summary SHM

# Format strings
DEVICE_HEADER_FORMAT = '<20sIf20sI' + 'I' * MAX_SHAPE_DIMENSIONS + 'If'  # device_type, device_id, fps, data_dtype, num_dims, shape, frame_size, hardware_latency_ms
BUFFER_HEADER_FORMAT = '<IIII'          # buffer_size, current_frames_count, write_index, padding
MANAGER_HEADER_FORMAT = '<IIIIQ'        # device_count, buffer_size, current_frames_count, write_index, update_timestamp
POLICY_HEADER_FORMAT = '<IQI'           # device_count, update_timestamp, padding - for policy SHM
FRAME_HEADER_FORMAT = '<Q'              # timestamp_ns
SUMMARY_FRAME_HEADER_FORMAT = '<Q'      # frame_timestamp_ns

def pad_shape(shape, max_dims: int = MAX_SHAPE_DIMENSIONS) -> Tuple[int, ...]:
    """Pad shape to max_dims with zeros."""
    # Convert to tuple if it's a list
    if isinstance(shape, list):
        shape = tuple(shape)
    
    if len(shape) > max_dims:
        raise ValueError(f"Shape {shape} has {len(shape)} dimensions, which exceeds the maximum of {max_dims}. "
                        f"You can increase MAX_SHAPE_DIMENSIONS in utils/shm_utils.py to support more dimensions.")
    elif len(shape) < max_dims:
        return shape + (0,) * (max_dims - len(shape))
    return shape

def truncate_shape(shape: Tuple[int, ...], max_dims: int = MAX_SHAPE_DIMENSIONS) -> Tuple[int, ...]:
    """Truncate shape to max_dims."""
    return shape[:max_dims]

def validate_shape(shape, max_dims: int = MAX_SHAPE_DIMENSIONS) -> bool:
    """Validate shape dimensions."""
    # Convert to tuple if it's a list
    if isinstance(shape, list):
        shape = tuple(shape)
    
    return len(shape) <= max_dims and all(dim >= 0 for dim in shape)

def pack_device_header(device_type: str, device_id: int, fps: float, data_dtype: str, 
                      shape: Tuple[int, ...], frame_size: int, hardware_latency_ms: float = 0.0) -> bytes:
    """Pack device header."""
    if not validate_shape(shape):
        raise ValueError(f"Invalid shape: {shape}")
    
    padded_shape = pad_shape(shape)
    device_type_bytes = device_type.encode('utf-8')[:20].ljust(20, b'\x00')
    data_dtype_bytes = data_dtype.encode('utf-8')[:20].ljust(20, b'\x00')
    num_dims = len(shape)
    
    return struct.pack(DEVICE_HEADER_FORMAT, 
                      device_type_bytes, device_id, fps, data_dtype_bytes, 
                      num_dims, *padded_shape, frame_size, hardware_latency_ms)

def unpack_device_header(data: bytes) -> Dict[str, Any]:
    """Unpack device header."""
    if len(data) < DEVICE_HEADER_SIZE:
        raise ValueError(f"Data too short for device header: {len(data)} < {DEVICE_HEADER_SIZE}")
    
    values = struct.unpack(DEVICE_HEADER_FORMAT, data[:DEVICE_HEADER_SIZE])
    
    device_type = values[0].decode('utf-8').rstrip('\x00')
    device_id = values[1]
    fps = values[2]
    data_dtype = values[3].decode('utf-8').rstrip('\x00')
    num_dims = values[4]
    shape = tuple(values[5:5+MAX_SHAPE_DIMENSIONS])
    frame_size = values[5+MAX_SHAPE_DIMENSIONS]
    hardware_latency_ms = values[6+MAX_SHAPE_DIMENSIONS]
    
    # Truncate shape to actual dimensions
    actual_shape = shape[:num_dims] if num_dims > 0 else (1,)
    
    return {
        'device_type': device_type,
        'device_id': device_id,
        'fps': fps,
        'data_dtype': data_dtype,
        'num_dims': num_dims,
        'shape': actual_shape,
        'frame_size': frame_size,
        'hardware_latency_ms': hardware_latency_ms
    }

def pack_buffer_header(buffer_size: int, current_frames_count: int, write_index: int) -> bytes:
    """Pack buffer header."""
    return struct.pack(BUFFER_HEADER_FORMAT, buffer_size, current_frames_count, write_index, 0)

def unpack_buffer_header(data: bytes) -> Dict[str, int]:
    """Unpack buffer header."""
    if len(data) < BUFFER_HEADER_SIZE:
        raise ValueError(f"Data too short for buffer header: {len(data)} < {BUFFER_HEADER_SIZE}")
    
    values = struct.unpack(BUFFER_HEADER_FORMAT, data[:BUFFER_HEADER_SIZE])
    
    return {
        'buffer_size': values[0],
        'current_frames_count': values[1],
        'write_index': values[2],
        'padding': values[3]
    }

def pack_manager_header(device_count: int, buffer_size: int = 0, current_frames_count: int = 0,
                        write_index: int = 0, timestamp: Optional[int] = None) -> bytes:
    """Pack manager header."""
    if timestamp is None:
        timestamp = time.time_ns()
    return struct.pack(MANAGER_HEADER_FORMAT, device_count, buffer_size, current_frames_count, write_index, timestamp)

def pack_policy_header(device_count: int, timestamp: Optional[int] = None) -> bytes:
    """Pack policy header (simple version without buffer support)."""
    if timestamp is None:
        timestamp = time.time_ns()
    return struct.pack(POLICY_HEADER_FORMAT, device_count, timestamp, 0)

def unpack_manager_header(data: bytes) -> Dict[str, Any]:
    """Unpack manager header."""
    if len(data) < MANAGER_HEADER_SIZE:
        raise ValueError(f"Data too short for manager header: {len(data)} < {MANAGER_HEADER_SIZE}")
    
    values = struct.unpack(MANAGER_HEADER_FORMAT, data[:MANAGER_HEADER_SIZE])
    
    return {
        'device_count': values[0],
        'buffer_size': values[1],
        'current_frames_count': values[2],
        'write_index': values[3],
        'update_timestamp': values[4]
    }

def unpack_policy_header(data: bytes) -> Dict[str, Any]:
    """Unpack policy header."""
    if len(data) < POLICY_HEADER_SIZE:
        raise ValueError(f"Data too short for policy header: {len(data)} < {POLICY_HEADER_SIZE}")
    
    values = struct.unpack(POLICY_HEADER_FORMAT, data[:POLICY_HEADER_SIZE])
    
    return {
        'device_count': values[0],
        'update_timestamp': values[1],
        'padding': values[2]
    }

def pack_frame_header(timestamp_ns: int) -> bytes:
    """Pack frame header."""
    return struct.pack(FRAME_HEADER_FORMAT, timestamp_ns)

def unpack_frame_header(data: bytes) -> Dict[str, int]:
    """Unpack frame header."""
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(f"Data too short for frame header: {len(data)} < {FRAME_HEADER_SIZE}")
    
    timestamp_ns = struct.unpack(FRAME_HEADER_FORMAT, data[:FRAME_HEADER_SIZE])[0]
    
    return {
        'timestamp_ns': timestamp_ns
    }

def pack_summary_frame_header(frame_timestamp_ns: int) -> bytes:
    """Pack summary frame header."""
    return struct.pack(SUMMARY_FRAME_HEADER_FORMAT, frame_timestamp_ns)

def unpack_summary_frame_header(data: bytes) -> Dict[str, int]:
    """Unpack summary frame header."""
    if len(data) < SUMMARY_FRAME_HEADER_SIZE:
        raise ValueError(f"Data too short for summary frame header: {len(data)} < {SUMMARY_FRAME_HEADER_SIZE}")
    
    frame_timestamp_ns = struct.unpack(SUMMARY_FRAME_HEADER_FORMAT, data[:SUMMARY_FRAME_HEADER_SIZE])[0]
    
    return {
        'frame_timestamp_ns': frame_timestamp_ns
    }

def calculate_device_shm_size(buffer_size: int, frame_data_size: int) -> int:
    """Calculate total device SHM size.
    
    Args:
        buffer_size: Number of frames in buffer
        frame_data_size: Size of frame data (excluding FRAME_HEADER_SIZE)
    """
    return DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * (FRAME_HEADER_SIZE + frame_data_size)

def calculate_summary_shm_size(device_count: int, frame_sizes: List[int], buffer_size: int) -> int:
    """Calculate total summary SHM size with buffer support."""
    frame_data_size = sum(frame_sizes)  # All device data per frame
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + frame_data_size  # Frame header + device data
    return MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + buffer_size * total_frame_size

def calculate_policy_shm_size(device_count: int, frame_sizes: List[int]) -> int:
    """Calculate total policy SHM size (simple format)."""
    return POLICY_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes)

def get_device_info_offset(device_index: int) -> int:
    """Get device info offset in summary SHM."""
    return MANAGER_HEADER_SIZE + device_index * DEVICE_HEADER_SIZE

def get_data_offset(device_index: int, device_count: int, frame_sizes: List[int]) -> int:
    """Get data offset for device in summary SHM."""
    return MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes[:device_index])

def get_policy_data_offset(device_index: int, device_count: int, frame_sizes: List[int]) -> int:
    """Get data offset for device in policy SHM."""
    return POLICY_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes[:device_index])

def get_frame_offset(device_index: int, buffer_size: int, frame_size: int, frame_index: int) -> int:
    """Get frame offset in device SHM."""
    return DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * (FRAME_HEADER_SIZE + frame_size)

def get_summary_frame_offset(device_index: int, frame_index: int, device_count: int, 
                           frame_sizes: List[int], buffer_size: int) -> int:
    """Get frame offset for device in summary SHM with buffer support."""
    # Buffer layout: Frame0[FrameHeader+Device0+Device1+...] + Frame1[FrameHeader+Device0+Device1+...] + ...
    # Base offset: manager header + all device headers
    base_offset = MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE
    # Offset to specific frame (including frame header)
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes)
    frame_start_offset = base_offset + frame_index * total_frame_size
    # Offset to specific device within that frame (after frame header)
    device_offset_in_frame = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes[:device_index])
    return frame_start_offset + device_offset_in_frame

def get_summary_frame_start_offset(frame_index: int, device_count: int, frame_sizes: List[int]) -> int:
    """Get start offset for entire frame in summary SHM buffer."""
    base_offset = MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes)  # Frame header + device data
    return base_offset + frame_index * total_frame_size


class DataChannel:
    """Data channel for SharedMemory-based inter-process communication.
    
    Provides unified access to SharedMemory with optional read-only protection.
    """
    def __init__(self, shm_obj: shm.SharedMemory, read_only: bool = False):
        self._shm = shm_obj
        self._buffer = shm_obj.buf
        self._name = shm_obj.name
        self._read_only = read_only
        
    def __getitem__(self, key: Union[slice, int]):
        result = self._buffer[key]
        # For slices, return the raw buffer to maintain np.frombuffer compatibility
        # For single items, we can safely return the value
        if isinstance(key, slice):
            return result  # Return raw buffer slice, not wrapped
        else:
            return result  # Single item access
        
    def __setitem__(self, key: Union[slice, int], value: Any):
        self._check_write_permission()
        self._buffer[key] = value
    
    def _check_write_permission(self):
        """Check if write operations are allowed."""
        if self._read_only:
            logger.error(f"Attempted write to read-only data channel: {self._name}")
            raise PermissionError(f"Data channel '{self._name}' is read-only")
        
    def __len__(self):
        return len(self._buffer)
        
    def __bytes__(self):
        return bytes(self._buffer)
    
    def __buffer__(self, flags):
        # Support Python 3.12+ buffer protocol
        if hasattr(self._buffer, '__buffer__'):
            return self._buffer.__buffer__(flags)
        else:
            # Fallback: create memoryview and return its buffer
            return memoryview(self._buffer).__buffer__(flags)
    
    def __getbuffer__(self, view, flags):
        # Support older buffer protocol
        if hasattr(self._buffer, '__getbuffer__'):
            return self._buffer.__getbuffer__(view, flags)
        else:
            # Fallback for objects that don't implement __getbuffer__
            mv = memoryview(self._buffer)
            return mv.__getbuffer__(view, flags)
    
    def __releasebuffer__(self, view):
        if hasattr(self._buffer, '__releasebuffer__'):
            return self._buffer.__releasebuffer__(view)
    
    # Add memoryview support for np.frombuffer compatibility
    def __array_interface__(self):
        """NumPy array interface support."""
        if hasattr(self._buffer, '__array_interface__'):
            return self._buffer.__array_interface__
        return None
    
    # Support memoryview() constructor
    def __memoryview__(self):
        return memoryview(self._buffer)
    
    # SharedMemory compatibility properties and methods
    @property
    def buf(self):
        """Return the buffer for SharedMemory compatibility."""
        if self._read_only:
            # Return readonly memoryview to prevent direct buffer modification
            return self._buffer.toreadonly()
        else:
            return self._buffer
    
    @property
    def name(self) -> str:
        """Get the name of the shared memory."""
        return self._name
    
    @property
    def size(self) -> int:
        """Get the size of the shared memory."""
        return self._shm.size
    
    @property
    def is_read_only(self) -> bool:
        """Check if this channel is read-only."""
        return self._read_only
    
    def close(self):
        """Close the underlying SharedMemory."""
        self._shm.close()
    
    def unlink(self):
        """Unlink the underlying SharedMemory if not read-only."""
        if self._read_only:
            logger.error(f"Attempted unlink on read-only data channel: {self._name}")
            raise PermissionError(f"Cannot unlink read-only data channel: {self._name}")
        else:
            self._shm.unlink()


# Backward compatibility aliases
ReadOnlyBuffer = DataChannel  # Legacy name
ReadOnlySharedMemory = DataChannel  # Legacy name

# Type alias for cleaner code
SharedMemoryChannel = DataChannel

def create_shared_memory(name: str, size: int, readonly_after_creation: bool = False) -> Union[shm.SharedMemory, DataChannel]:
    """
    Create shared memory with unified interface.
    
    Args:
        name: SHM name
        size: SHM size in bytes
        readonly_after_creation: If True, return read-only wrapper after creation
        
    Returns:
        SharedMemory object or ReadOnlyBuffer wrapper
        
    Raises:
        Exception: If creation fails
    """
    # Clean up any existing shared memory with the same name
    try:
        existing_shm = shm.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
    except FileNotFoundError:
        pass  # No existing memory
    except Exception:
        pass  # Ignore cleanup errors
    
    # Create new shared memory
    shm_obj = shm.SharedMemory(create=True, size=size, name=name)
    
    if readonly_after_creation:
        logger.debug(f"Created SHM '{name}' with read-only wrapper")
        return DataChannel(shm_obj, read_only=True)
    else:
        return DataChannel(shm_obj, read_only=False)


def connect_to_shared_memory(name: str, read_only: bool = False) -> DataChannel:
    """
    Connect to existing shared memory with optional read-only protection.
    
    Args:
        name: SHM name
        read_only: If True, return read-only wrapper
        
    Returns:
        SharedMemory object or ReadOnlyBuffer wrapper
        
    Raises:
        FileNotFoundError: If SHM doesn't exist
    """
    try:
        # Prevent resource tracker from auto-cleaning
        import multiprocessing.resource_tracker as tracker
        original_register = tracker.register
        tracker.register = lambda *args, **kwargs: None
        
        try:
            shm_obj = shm.SharedMemory(name=name)
            
            logger.debug(f"Connected to SHM '{name}' in {'read-only' if read_only else 'read-write'} mode")
            return DataChannel(shm_obj, read_only=read_only)
                
        finally:
            tracker.register = original_register
            
    except FileNotFoundError:
        logger.error(f"SHM not found: {name}")
        raise
    except Exception as e:
        logger.error(f"Error connecting to SHM '{name}': {e}")
        raise


# Convenience functions for specific SHM types
def create_device_shm(device_name: str, device_id: int, buffer_size: int, frame_data_size: int, readonly_after_creation: bool = False) -> DataChannel:
    """Create device SHM with standard naming."""
    shm_name = f"{device_name}_{device_id}_data"
    shm_size = calculate_device_shm_size(buffer_size, frame_data_size)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_device_shm(device_name: str, device_id: int, read_only: bool = True) -> DataChannel:
    """Connect to device SHM with read-only default."""
    shm_name = f"{device_name}_{device_id}_data"
    return connect_to_shared_memory(shm_name, read_only=read_only)


def create_control_shm(device_name: str, device_id: int, buffer_size: int, frame_data_size: int, readonly_after_creation: bool = False) -> DataChannel:
    """Create control SHM with standard naming."""
    shm_name = f"{device_name}_{device_id}_control"
    shm_size = calculate_device_shm_size(buffer_size, frame_data_size)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_control_shm(device_name: str, device_id: int, read_only: bool = True) -> DataChannel:
    """Connect to control SHM with read-only default for consumers."""
    shm_name = f"{device_name}_{device_id}_control"
    return connect_to_shared_memory(shm_name, read_only=read_only)


def create_summary_shm(device_count: int, frame_sizes: List[int], buffer_size: int, readonly_after_creation: bool = False) -> DataChannel:
    """Create summary SHM with standard naming."""
    shm_name = "device_summary_data"
    shm_size = calculate_summary_shm_size(device_count, frame_sizes, buffer_size)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_summary_shm(read_only: bool = True) -> DataChannel:
    """Connect to summary SHM with read-only default for consumers."""
    return connect_to_shared_memory("device_summary_data", read_only=read_only)


def create_policy_shm(device_count: int, frame_sizes: List[int], readonly_after_creation: bool = False) -> DataChannel:
    """Create policy SHM with standard naming."""
    shm_name = "policy_data"
    shm_size = calculate_policy_shm_size(device_count, frame_sizes)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_policy_shm(read_only: bool = False) -> DataChannel:
    """Connect to policy SHM with read-write default for policy runners."""
    return connect_to_shared_memory("policy_data", read_only=read_only)


def make_readonly(shm_obj: shm.SharedMemory) -> DataChannel:
    """Convert existing SharedMemory object to read-only wrapper."""
    return DataChannel(shm_obj, read_only=True)