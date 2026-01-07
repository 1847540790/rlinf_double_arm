#!/usr/bin/env python3
"""
Camera device implementations for video capture and streaming.
Includes OpenCV and Hikvision camera support.

Author: Jun Lv
"""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from ctypes import cast, POINTER, c_ubyte
from .base import BaseDevice
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

# Try to import Hikvision camera modules
try:
    from third_party.MvImport.MvCameraControl_class import MvCamera
    from third_party.MvImport.CameraParams_header import (
        MV_CC_DEVICE_INFO, MV_CC_DEVICE_INFO_LIST, 
        MV_FRAME_OUT, MV_TRIGGER_MODE_OFF
    )
    from third_party.MvImport.MvErrorDefine_const import MV_OK
    from third_party.MvImport.CameraParams_const import (
        MV_GIGE_DEVICE, MV_USB_DEVICE, 
        MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE, 
        MV_GENTL_XOF_DEVICE, MV_GENTL_GIGE_DEVICE,
        MV_ACCESS_Exclusive
    )
    from third_party.MvImport.PixelType_header import (
        PixelType_Gvsp_Mono8, PixelType_Gvsp_RGB8_Packed, 
        PixelType_Gvsp_BGR8_Packed, PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerBG8
    )
    HIK_CAMERA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hikvision camera modules not available: {e}")
    HIK_CAMERA_AVAILABLE = False


class OpenCVCameraDevice(BaseDevice):
    """
    OpenCV camera device class that captures video from OpenCV camera and writes to shared memory.
    Supports continuous memory buffer for multiple frames and hardware camera parameters.
    """
    
    def __init__(self, device_id=0, data_shape=(480, 640), fps=30.0, data_dtype=np.uint8, buffer_size=1, hardware_latency_ms=0.0,
                 brightness=None, contrast=None, saturation=None, hue=None, gamma=None, sharpness=None):
        """
        Initialize the camera device.
        
        Args:
            device_id: Camera device ID for OpenCV (usually 0 for default camera)
            data_shape: Shape of camera frames (height, width) for grayscale or (height, width, channels) for color
            fps: Target frames per second for capture
            data_dtype: Data type for camera frames (string or numpy dtype, default: np.uint8)
            buffer_size: Number of frames to store in buffer (default: 1)
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            brightness: Camera brightness setting (if None, not set)
            contrast: Camera contrast setting (if None, not set)
            saturation: Camera saturation setting (if None, not set)
            hue: Camera hue setting (if None, not set)
            gamma: Camera gamma setting (if None, not set)
            sharpness: Camera sharpness setting (if None, not set)
            
        Raises:
            RuntimeError: If camera initialization fails
        """
        super().__init__(device_id=device_id, data_shape=data_shape, fps=fps, data_dtype=data_dtype, buffer_size=buffer_size, hardware_latency_ms=hardware_latency_ms)
        
        # Override device name for OpenCV camera
        self.device_name = "OpenCVCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Camera specific attributes
        self.cap = None
        self.is_color = len(data_shape) == 3 and data_shape[2] == 3
        self._resize_warned = False  # Flag to prevent repeated resize warnings
        
        # Camera hardware parameters (None means not set)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.sharpness = sharpness
        
        # Initialize camera immediately
        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize camera {device_id}")
        
    def _initialize_camera(self) -> None:
        """Initialize OpenCV camera capture and set hardware parameters."""
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.device_id}")
            
            # Set basic camera properties
            if self.is_color:
                height, width, channels = self.data_shape
            else:
                height, width = self.data_shape
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set hardware camera parameters if specified
            self._set_camera_parameters()
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Cannot capture frame from camera")
            
            logger.info(f"Camera {self.device_id} initialized: {frame.shape} -> {self.data_shape}")
            self._log_camera_parameters()
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def _set_camera_parameters(self) -> None:
        """Set hardware camera parameters using OpenCV properties."""
        if self.brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            
        if self.contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            
        if self.saturation is not None:
            self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
            
        if self.hue is not None:
            self.cap.set(cv2.CAP_PROP_HUE, self.hue)
            
        if self.gamma is not None:
            self.cap.set(cv2.CAP_PROP_GAMMA, self.gamma)
            
        if self.sharpness is not None:
            # Note: CAP_PROP_SHARPNESS is not standard in OpenCV
            # Some cameras may support it, but it's not guaranteed
            try:
                self.cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness)
            except:
                logger.warning(f"Camera {self.device_id} does not support sharpness setting")
    
    def _log_camera_parameters(self) -> None:
        """Log the current camera parameters."""
        params = []
        if self.brightness is not None:
            params.append(f"brightness={self.brightness}")
        if self.contrast is not None:
            params.append(f"contrast={self.contrast}")
        if self.saturation is not None:
            params.append(f"saturation={self.saturation}")
        if self.hue is not None:
            params.append(f"hue={self.hue}")
        if self.gamma is not None:
            params.append(f"gamma={self.gamma}")
        if self.sharpness is not None:
            params.append(f"sharpness={self.sharpness}")
        
        if params:
            logger.info(f"Camera {self.device_id} hardware parameters: {', '.join(params)}")
        else:
            logger.info(f"Camera {self.device_id}: No hardware parameters set")
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera and convert to the required format.
        
        Returns:
            tuple: (timestamp_ns, numpy.ndarray) or (None, None) if capture failed
        """
        if not self.cap:
            return None, None
        
        try:
            # Get timestamp immediately when starting to capture
            ret, frame = self.cap.read()
            timestamp_ns = time.time_ns()
            if not ret:
                logger.warning("Failed to capture frame")
                return None, None
            
            # Convert frame to required format
            if self.is_color:
                # For color images, ensure correct channel order (BGR -> RGB if needed)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # OpenCV uses BGR, convert to RGB for consistency
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if necessary
                target_height, target_width, target_channels = self.data_shape
                if frame.shape[:2] != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(f"Resizing frame from {frame.shape[:2]} to ({target_height}, {target_width}). "
                                      f"This adds computational overhead. Consider setting camera resolution to match target shape.")
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # Ensure 3 channels
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:  # RGBA -> RGB
                    frame = frame[:, :, :3]
                    
            else:
                # For grayscale images
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize if necessary
                target_height, target_width = self.data_shape
                if frame.shape != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(f"Resizing frame from {frame.shape} to ({target_height}, {target_width}). "
                                      f"This adds computational overhead. Consider setting camera resolution to match target shape.")
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))
            
            # Convert to required data type
            frame = frame.astype(self.data_dtype)
            
            return timestamp_ns, frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, None
    
    def _cleanup_camera(self) -> None:
        """Clean up camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")
    
    def start_server(self) -> None:
        """Start the camera device server and run frame capture loop."""
        if self.running:
            logger.info(f"Camera {self.device_id} is already running")
            return
        
        logger.info(f"Starting camera server {self.device_id}...")
        self.running = True
        
        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        
        logger.info(f"Camera server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main frame capture loop
        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                
                # Wait for next iteration using precise timing
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the camera device server."""
        if not self.running:
            return
        
        logger.info(f"Stopping camera server {self.device_id}...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Camera server stopped")
    
    def close(self) -> None:
        """Close the camera device and release all resources."""
        self.stop_server()
        self._cleanup_camera()
        logger.info(f"Camera {self.device_id} closed")
    
    def __del__(self) -> None:
        """Destructor to ensure camera and shared memory resources are released."""
        try:
            self._cleanup_camera()
            # Also cleanup shared memory from parent class
            if hasattr(self, 'shared_memory') and self.shared_memory:
                self._cleanup_shared_memory()
        except:
            pass  # Ignore errors during cleanup in destructor
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information."""
        info = super().get_device_info()
        info.update({
            "camera_id": self.device_id,  # camera_id is same as device_id
            "is_color": self.is_color,
            "camera_opened": self.cap is not None and self.cap.isOpened() if self.cap else False,
            "hardware_parameters": {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "hue": self.hue,
                "gamma": self.gamma,
                "sharpness": self.sharpness
            }
        })
        return info


class HikCameraDevice(BaseDevice):
    """
    Hikvision camera device class that captures video from Hikvision camera and writes to shared memory.
    Uses the MvCamera SDK for camera control and image acquisition.
    """
    
    def __init__(self, device_id: int = 0, 
                 data_shape: tuple = (480, 640, 3), fps: float = 30.0, 
                 data_dtype=np.uint8, buffer_size: int = 1, 
                 hardware_latency_ms: float = 0.0,
                 pixel_format: Optional[str] = None,
                 frame_rate_enable: Optional[bool] = None,
                 exposure_time: Optional[float] = None,
                 gain: Optional[float] = None,
                 gamma_enable: Optional[bool] = None,
                 gamma: Optional[float] = None,
                 rotation_angle: int = 0):
        """
        Initialize the Hikvision camera device.
        
        Args:
            device_id: Camera device index (0-based index from enumerated devices)
            data_shape: Shape of camera frames (height, width) for grayscale or (height, width, channels) for color
            fps: Target frames per second for capture
            data_dtype: Data type for camera frames (default: np.uint8)
            buffer_size: Number of frames to store in buffer (default: 1)
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            pixel_format: Pixel format (e.g., "RGB8", "BGR8", "Mono8") (default: None)
            frame_rate_enable: Enable frame rate control (default: None)
            exposure_time: Exposure time in microseconds (default: None)
            gain: Gain in dB (default: None)
            gamma_enable: Enable gamma correction (default: None)
            gamma: Gamma value (default: None)
            rotation_angle: Image rotation angle in degrees (supports 0/90/180/270, clockwise)
            
        Raises:
            RuntimeError: If Hikvision camera modules are not available
            RuntimeError: If camera initialization fails
        """
        if not HIK_CAMERA_AVAILABLE:
            raise RuntimeError("Hikvision camera modules are not available. Please install MvImport.")
        
        super().__init__(device_id=device_id, data_shape=data_shape, fps=fps, 
                        data_dtype=data_dtype, buffer_size=buffer_size, 
                        hardware_latency_ms=hardware_latency_ms)
        
        # Override device name for Hikvision camera
        self.device_name = "HikCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        
        # Camera specific attributes
        self.cam = None
        self.is_color = len(data_shape) == 3 and data_shape[2] == 3
        self._resize_warned = False
        
        # Camera info (will be populated during initialization)
        self.camera_model = "Unknown"
        self.camera_serial = "Unknown"
        self.camera_manufacturer = "Unknown"
        
        # Hikvision camera hardware parameters (None means not set)
        self.pixel_format = pixel_format
        self.frame_rate_enable = frame_rate_enable
        self.exposure_time = exposure_time
        self.gain = gain
        self.gamma_enable = gamma_enable
        self.gamma = gamma
        self.rotation_angle = self._validate_rotation(rotation_angle)
        if self.rotation_angle:
            logger.info(f"Hikvision camera frames will be rotated {self.rotation_angle} degrees clockwise before sharing")
        
        # Initialize camera
        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize Hikvision camera {device_id}")
    
    def _validate_rotation(self, rotation_angle: int) -> int:
        """Validate supported rotation angles."""
        valid_angles = {0, 90, 180, 270}
        if rotation_angle not in valid_angles:
            raise ValueError(f"Unsupported rotation_angle={rotation_angle}. Supported values: {sorted(valid_angles)}")
        return rotation_angle
    
    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Rotate image according to configured rotation_angle."""
        if self.rotation_angle == 0:
            return image
        
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        try:
            return cv2.rotate(image, rotate_map[self.rotation_angle])
        except Exception as exc:
            logger.error(f"Failed to rotate frame by {self.rotation_angle} degrees: {exc}")
            return image
    
    def _set_camera_parameters(self) -> bool:
        """
        Configure Hikvision camera parameters.
        
        Parameters are set in the following order (as required by Hikvision SDK):
        1. Pixel Format (RGB8) - Must be set first
        2. Acquisition Frame Rate Control Enable
        3. Acquisition Frame Rate
        4. Exposure Time
        5. Gain
        6. Gamma Enable
        7. Gamma
        
        Returns:
            bool: True if all parameters set successfully, False otherwise
        """
        if not self.cam:
            return False
        
        logger.info("=== Configuring Hikvision camera parameters ===")
        
        # 1. Set pixel format (must be set first!)
        if self.pixel_format is not None:
            logger.info(f"1. Setting pixel format to {self.pixel_format}...")
            ret = self.cam.MV_CC_SetEnumValueByString("PixelFormat", self.pixel_format)
            if ret != MV_OK:
                logger.warning(f"   Failed to set pixel format! ret[0x{ret:x}]")
                # Try with "Packed" suffix as fallback
                if "Packed" not in self.pixel_format:
                    logger.info(f"   Trying {self.pixel_format}Packed...")
                    ret = self.cam.MV_CC_SetEnumValueByString("PixelFormat", f"{self.pixel_format}Packed")
                    if ret != MV_OK:
                        logger.error(f"   Failed to set pixel format! ret[0x{ret:x}]")
                        return False
            logger.info("   ✓ Pixel format set successfully")
        
        # 2. Enable frame rate control
        if self.frame_rate_enable is not None:
            logger.info(f"2. {'Enabling' if self.frame_rate_enable else 'Disabling'} frame rate control...")
            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", self.frame_rate_enable)
            if ret != MV_OK:
                logger.warning(f"   Failed to set frame rate control! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Frame rate control configured successfully")
        
        # 3. Set frame rate
        if self.fps is not None:
            logger.info(f"3. Setting frame rate to {self.fps}...")
            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.fps)
            if ret != MV_OK:
                logger.warning(f"   Failed to set frame rate! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Frame rate set successfully")
        
        # 4. Set exposure time
        if self.exposure_time is not None:
            logger.info(f"4. Setting exposure time to {self.exposure_time} μs...")
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if ret != MV_OK:
                logger.warning(f"   Failed to set exposure time! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Exposure time set successfully")
        
        # 5. Set gain
        if self.gain is not None:
            logger.info(f"5. Setting gain to {self.gain} dB...")
            ret = self.cam.MV_CC_SetFloatValue("Gain", self.gain)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gain! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gain set successfully")
        
        # 6. Enable gamma correction
        if self.gamma_enable is not None:
            logger.info(f"6. {'Enabling' if self.gamma_enable else 'Disabling'} gamma correction...")
            # Try to set GammaSelector to User (some cameras require this)
            try:
                self.cam.MV_CC_SetEnumValue("GammaSelector", 1)  # 1 = User
            except:
                pass
            
            ret = self.cam.MV_CC_SetBoolValue("GammaEnable", self.gamma_enable)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gamma enable! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gamma correction configured successfully")
        
        # 7. Set gamma value
        if self.gamma is not None:
            logger.info(f"7. Setting gamma value to {self.gamma}...")
            ret = self.cam.MV_CC_SetFloatValue("Gamma", self.gamma)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gamma value! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gamma value set successfully")
        
        logger.info("=== Camera parameters configuration completed ===")
        return True
    
    def _initialize_camera(self) -> bool:
        """Initialize Hikvision camera and configure settings."""
        try:
            # Initialize SDK
            ret = MvCamera.MV_CC_Initialize()
            if ret != MV_OK:
                logger.error(f"Failed to initialize Hikvision SDK, error code: 0x{ret:x}")
                return False
            
            # Enumerate devices
            device_list = MV_CC_DEVICE_INFO_LIST()
            device_list.nDeviceNum = 0
            
            tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                         MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE | MV_GENTL_GIGE_DEVICE)
            
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
            if ret != MV_OK:
                logger.error(f"Failed to enumerate devices, error code: 0x{ret:x}")
                MvCamera.MV_CC_Finalize()
                return False
            
            if device_list.nDeviceNum == 0:
                logger.error("No Hikvision cameras found!")
                MvCamera.MV_CC_Finalize()
                return False
            
            logger.info(f"Found {device_list.nDeviceNum} Hikvision camera(s)")
            
            # Use device_id as index to select device
            if self.device_id >= device_list.nDeviceNum:
                logger.error(f"Device index {self.device_id} out of range (found {device_list.nDeviceNum} cameras)")
                MvCamera.MV_CC_Finalize()
                return False
            
            # Get device info
            stDeviceList = cast(device_list.pDeviceInfo[self.device_id], POINTER(MV_CC_DEVICE_INFO)).contents
            
            # Extract camera information
            if stDeviceList.nTLayerType == MV_USB_DEVICE:
                usb_info = stDeviceList.SpecialInfo.stUsb3VInfo
                self.camera_model = ''.join([chr(c) for c in usb_info.chModelName if c != 0])
                self.camera_serial = ''.join([chr(c) for c in usb_info.chSerialNumber if c != 0])
                self.camera_manufacturer = ''.join([chr(c) for c in usb_info.chManufacturerName if c != 0])
            else:
                gige_info = stDeviceList.SpecialInfo.stGigEInfo
                self.camera_model = ''.join([chr(c) for c in gige_info.chModelName if c != 0])
                self.camera_serial = ''.join([chr(c) for c in gige_info.chSerialNumber if c != 0])
                self.camera_manufacturer = ''.join([chr(c) for c in gige_info.chManufacturerName if c != 0])
            
            logger.info(f"Selected camera: {self.camera_manufacturer} {self.camera_model} (SN: {self.camera_serial})")
            
            # Create camera instance
            self.cam = MvCamera()
            
            # Create handle
            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != MV_OK:
                logger.error(f"Failed to create handle, error code: 0x{ret:x}")
                MvCamera.MV_CC_Finalize()
                return False
            
            # Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != MV_OK:
                logger.error(f"Failed to open device, error code: 0x{ret:x}")
                self.cam.MV_CC_DestroyHandle()
                MvCamera.MV_CC_Finalize()
                return False
            
            # Configure camera parameters (must be done after opening device, before starting grabbing)
            if not self._set_camera_parameters():
                logger.warning("Failed to configure camera parameters, continuing with defaults...")
            
            # Set trigger mode to off (free run mode)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != MV_OK:
                logger.warning(f"Failed to set trigger mode, error code: 0x{ret:x}")
            
            # Increase SDK buffer size for async mode to prevent frame drops
            # Default is 1, increase to 20 to buffer more frames in async mode
            try:
                ret = self.cam.MV_CC_SetImageNodeNum(1000)
                if ret == MV_OK:
                    logger.info(f"Set SDK image buffer count to 1000 for async mode")
                else:
                    logger.warning(f"Failed to set SDK buffer count, error code: 0x{ret:x}")
            except Exception as e:
                logger.warning(f"Failed to set SDK buffer count: {e}")
            
            logger.info(f"Hikvision camera {self.camera_serial} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.cam:
                try:
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
                except:
                    pass
            try:
                MvCamera.MV_CC_Finalize()
            except:
                pass
            return False
    
    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Capture a frame from the Hikvision camera.
        
        Returns:
            tuple: (timestamp_ns, numpy.ndarray) or (None, None) if capture failed
        """
        if not self.cam:
            return None, None
        
        try:
            # Get one frame with timeout (increase to 2000ms for async mode)
            stFrameInfo = MV_FRAME_OUT()
            ret = self.cam.MV_CC_GetImageBuffer(stFrameInfo, 2000)  # 2000ms timeout
            
            if ret != MV_OK:
                if ret != 0x80000007:  # MV_E_NODATA
                    logger.warning(f"Failed to get image buffer, error code: 0x{ret:x}")
                return None, None
            
            # Get timestamp
            timestamp_ns = time.time_ns()
            
            try:
                # Validate frame length before processing
                frame_len = stFrameInfo.stFrameInfo.nFrameLen
                height = stFrameInfo.stFrameInfo.nHeight
                width = stFrameInfo.stFrameInfo.nWidth
                pixel_type = stFrameInfo.stFrameInfo.enPixelType
                
                # Validate frame size is reasonable
                if frame_len == 0:
                    logger.warning(f"Received frame with zero length, skipping")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None
                
                # Calculate expected frame size based on pixel format
                expected_size = 0
                if pixel_type == PixelType_Gvsp_Mono8:
                    expected_size = height * width
                elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                    expected_size = height * width * 3
                elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                                   PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                    expected_size = height * width
                else:
                    logger.warning(f"Unknown pixel format: 0x{pixel_type:x}, using frame_len={frame_len}")
                    expected_size = frame_len
                
                # Validate frame_len matches expected size
                if expected_size > 0 and frame_len != expected_size:
                    logger.error(f"Frame size mismatch: frame_len={frame_len}, expected={expected_size} "
                               f"for {height}x{width} with pixel_type=0x{pixel_type:x}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None
                
                # Convert image data to numpy array
                image_data = cast(stFrameInfo.pBufAddr, POINTER(c_ubyte * frame_len)).contents
                image = np.frombuffer(image_data, dtype=np.uint8)
                
                # Validate image buffer size matches frame_len
                if len(image) != frame_len:
                    logger.error(f"Image buffer size mismatch: expected {frame_len}, got {len(image)}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None
                
                # Process based on pixel format (already extracted above)
                if pixel_type == PixelType_Gvsp_Mono8:
                    # Mono8 format: 1 byte per pixel
                    # Double-check size before reshape (already validated above, but be extra safe)
                    if len(image) != height * width:
                        logger.error(f"Mono8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width))
                    if self.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        
                elif pixel_type == PixelType_Gvsp_RGB8_Packed:
                    # RGB8 format: 3 bytes per pixel
                    # Double-check size before reshape
                    if len(image) != height * width * 3:
                        logger.error(f"RGB8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width * 3}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width, 3))
                    # Already in RGB format
                    
                elif pixel_type == PixelType_Gvsp_BGR8_Packed:
                    # BGR8 format: 3 bytes per pixel
                    # Double-check size before reshape
                    if len(image) != height * width * 3:
                        logger.error(f"BGR8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width * 3}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width, 3))
                    if self.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                                   PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                    # Bayer format: 1 byte per pixel (needs demosaicing)
                    # Double-check size before reshape
                    if len(image) != height * width:
                        logger.error(f"Bayer frame size mismatch before reshape: got {len(image)} bytes, expected {height * width}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width))
                    if pixel_type == PixelType_Gvsp_BayerGR8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerGR2RGB if self.is_color else cv2.COLOR_BayerGR2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerRG8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB if self.is_color else cv2.COLOR_BayerRG2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerGB8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerGB2RGB if self.is_color else cv2.COLOR_BayerGB2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerBG8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB if self.is_color else cv2.COLOR_BayerBG2GRAY)
                else:
                    logger.warning(f"Unsupported pixel format: 0x{pixel_type:x}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None
                
                # Apply configured rotation before resizing
                image = self._apply_rotation(image)
                
                # Resize if necessary
                target_shape = self.data_shape
                if self.is_color:
                    target_height, target_width, _ = target_shape
                    if image.shape[:2] != (target_height, target_width):
                        if not self._resize_warned:
                            logger.warning(f"Resizing frame from {image.shape[:2]} to ({target_height}, {target_width})")
                            self._resize_warned = True
                        image = cv2.resize(image, (target_width, target_height))
                else:
                    target_height, target_width = target_shape
                    if image.shape[:2] != (target_height, target_width):
                        if not self._resize_warned:
                            logger.warning(f"Resizing frame from {image.shape[:2]} to ({target_height}, {target_width})")
                            self._resize_warned = True
                        image = cv2.resize(image, (target_width, target_height))
                
                # Convert to required data type
                image = image.astype(self.data_dtype)
                
                # Free image buffer
                self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                
                return timestamp_ns, image
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                logger.error(f"  Frame info: frame_len={frame_len}, height={height}, width={width}, pixel_type=0x{pixel_type:x}")
                logger.error(f"  Image buffer: len={len(image) if 'image' in locals() else 'N/A'}, expected={expected_size if 'expected_size' in locals() else 'N/A'}")
                self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                return None, None
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
    
    def _cleanup_camera(self) -> None:
        """Clean up camera resources."""
        if self.cam:
            try:
                # Stop grabbing if it's running
                self.cam.MV_CC_StopGrabbing()
            except:
                pass
            
            try:
                # Close device
                self.cam.MV_CC_CloseDevice()
            except:
                pass
            
            try:
                # Destroy handle
                self.cam.MV_CC_DestroyHandle()
            except:
                pass
            
            self.cam = None
            
            try:
                # Finalize SDK
                MvCamera.MV_CC_Finalize()
            except:
                pass
            
            logger.info("Hikvision camera released")
    
    def start_server(self) -> None:
        """Start the camera device server and run frame capture loop."""
        if self.running:
            logger.info(f"Hikvision camera {self.camera_serial} is already running")
            return
        
        logger.info(f"Starting Hikvision camera server {self.camera_serial}...")
        self.running = True
        
        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        
        # Start grabbing
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            logger.error(f"Failed to start grabbing, error code: 0x{ret:x}")
            self.running = False
            self._cleanup_shared_memory()
            raise RuntimeError("Failed to start grabbing")
        
        logger.info(f"Hikvision camera server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        
        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        
        # Main frame capture loop
        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                
                # Wait for next iteration using precise timing
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                break
    
    def stop_server(self) -> None:
        """Stop the camera device server."""
        if not self.running:
            return
        
        logger.info(f"Stopping Hikvision camera server {self.camera_serial}...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Hikvision camera server stopped")
    
    def close(self) -> None:
        """Close the camera device and release all resources."""
        self.stop_server()
        self._cleanup_camera()
        logger.info(f"Hikvision camera {self.camera_serial} closed")
    
    def __del__(self) -> None:
        """Destructor to ensure camera and shared memory resources are released."""
        try:
            self._cleanup_camera()
            if hasattr(self, 'shared_memory') and self.shared_memory:
                self._cleanup_shared_memory()
        except:
            pass
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information."""
        info = super().get_device_info()
        info.update({
            "camera_type": "Hikvision",
            "camera_model": self.camera_model,
            "camera_serial": self.camera_serial,
            "camera_manufacturer": self.camera_manufacturer,
            "is_color": self.is_color,
            "camera_opened": self.cam is not None,
            "hardware_parameters": {
                "pixel_format": self.pixel_format,
                "frame_rate_enable": self.frame_rate_enable,
                "exposure_time": self.exposure_time,
                "gain": self.gain,
                "gamma_enable": self.gamma_enable,
                "gamma": self.gamma,
                "rotation_angle": self.rotation_angle
            }
        })
        return info


def main() -> None:
    """Main function to run the OpenCV camera device server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenCV Camera Device Server")
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--shape", "-s", type=str, default="240,320",
                        help="Frame shape as comma-separated values (default: 240,320)")
    parser.add_argument("--fps", "-f", type=float, default=15.0,
                        help="Frames per second (default: 15.0)")
    parser.add_argument("--dtype", "-t", default="uint8",
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 
                                'float32', 'float64'],
                        help="Data type (default: uint8)")
    parser.add_argument("--buffer-size", "-b", type=int, default=1,
                        help="Buffer size in frames (default: 1)")
    parser.add_argument("--color", "-c", action="store_true",
                        help="Enable color capture (3 channels)")
    parser.add_argument("--brightness", "-B", type=float, default=None,
                        help="Camera brightness setting (if not specified, not set)")
    parser.add_argument("--contrast", "-C", type=float, default=None,
                        help="Camera contrast setting (if not specified, not set)")
    parser.add_argument("--saturation", "-S", type=float, default=None,
                        help="Camera saturation setting (if not specified, not set)")
    parser.add_argument("--hue", "-H", type=float, default=None,
                        help="Camera hue setting (if not specified, not set)")
    parser.add_argument("--gamma", "-g", type=float, default=None,
                        help="Camera gamma setting (if not specified, not set)")
    parser.add_argument("--sharpness", "-X", type=float, default=None,
                        help="Camera sharpness setting (if not specified, not set)")
    
    args = parser.parse_args()
    
    # Parse shape string to tuple
    try:
        data_shape = tuple(int(x.strip()) for x in args.shape.split(','))
        if args.color and len(data_shape) == 2:
            data_shape = data_shape + (3,)  # Add color channel
        elif not args.color and len(data_shape) == 3:
            data_shape = data_shape[:2]  # Remove color channel
    except ValueError:
        logger.error("Invalid shape format. Use comma-separated integers (e.g., '240,320' or '240,320,3')")
        return
    
    # Import common utilities
    from utils.shm_utils import get_dtype
    
    data_dtype = get_dtype(args.dtype)
    
    # Create camera device
    camera = OpenCVCameraDevice(
        device_id=args.device_id,
        data_shape=data_shape,
        fps=args.fps,
        data_dtype=data_dtype,
        buffer_size=args.buffer_size,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        gamma=args.gamma,
        sharpness=args.sharpness
    )
    
    logger.info("OpenCV Camera Device Server")
    logger.info("===========================")
    logger.info(f"Camera ID: {args.device_id}")
    logger.info(f"Frame shape: {data_shape}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info(f"Color: {args.color}")
    logger.info("Camera Hardware Parameters:")
    logger.info(f"  Brightness: {args.brightness if args.brightness is not None else 'Not set'}")
    logger.info(f"  Contrast: {args.contrast if args.contrast is not None else 'Not set'}")
    logger.info(f"  Saturation: {args.saturation if args.saturation is not None else 'Not set'}")
    logger.info(f"  Hue: {args.hue if args.hue is not None else 'Not set'}")
    logger.info(f"  Gamma: {args.gamma if args.gamma is not None else 'Not set'}")
    logger.info(f"  Sharpness: {args.sharpness if args.sharpness is not None else 'Not set'}")
    logger.info("")
    
    try:
        logger.info("Camera server is running. Press Ctrl+C to stop...")
        logger.info(f"Camera info: {camera.get_device_info()}")
        camera.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        camera.close()


if __name__ == "__main__":
    main() 