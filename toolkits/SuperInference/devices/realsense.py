#!/usr/bin/env python3
"""
Intel RealSense camera device for video capture and streaming.

Author: Letian
"""

import time
from typing import Optional, Dict, Any

import numpy as np
import pyrealsense2 as rs

try:
    import cv2
except ImportError:  # pragma: no cover - debug visualization is optional
    cv2 = None

from .base import BaseDevice
from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class RealSenseCameraDevice(BaseDevice):
    """
    RealSense camera device class that captures video using Intel RealSense cameras
    and writes frames to shared memory.
    """

    def __init__(
        self,
        device_id: int = 0,
        data_shape=(480, 640),
        fps: float = 30.0,
        data_dtype=np.uint8,
        buffer_size: int = 1,
        hardware_latency_ms: float = 0.0,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
        gamma: Optional[float] = None,
        sharpness: Optional[float] = None,
        debug: bool = False,
        realsense_serial: Optional[str] = None,
    ) -> None:
        """
        Initialize the RealSense camera device.

        Args:
            device_id: Logical identifier for this camera (used for shared memory naming)
            data_shape: Expected frame shape (H, W) or (H, W, C)
            fps: Target frames per second
            data_dtype: Data type for frames
            buffer_size: Number of frames in the shared memory buffer
            hardware_latency_ms: Reported hardware latency in milliseconds
            brightness/contrast/...: Optional hardware parameters to apply when supported
        """
        super().__init__(
            device_id=device_id,
            data_shape=data_shape,
            fps=fps,
            data_dtype=data_dtype,
            buffer_size=buffer_size,
            hardware_latency_ms=hardware_latency_ms,
        )

        self.device_name = "RealSenseCamera"
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"

        # Camera specific attributes
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.pipeline_profile: Optional[rs.pipeline_profile] = None
        self.color_sensor: Optional[rs.sensor] = None
        self.device_serial: Optional[str] = None
        self.device_product_line: Optional[str] = None
        self.device_friendly_name: Optional[str] = None
        self.is_color = len(self.data_shape) == 3 and self.data_shape[2] == 3
        self._resize_warned = False
        self.requested_serial_number = realsense_serial

        # Store requested hardware parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.sharpness = sharpness
        self.debug = debug
        self._debug_window_name = f"RealSenseCameraDevice_{self.device_id}"
        if self.debug and cv2 is None:
            raise ImportError("OpenCV (cv2) is required for debug visualization. Install opencv-python to use --debug.")

        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize RealSense camera {device_id}")

    def _initialize_camera(self) -> bool:
        """Initialize the RealSense pipeline and configure the color stream."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            if self.requested_serial_number:
                try:
                    self.config.enable_device(self.requested_serial_number)
                except Exception as exc:
                    logger.error(
                        "Failed to select RealSense device with serial %s: %s",
                        self.requested_serial_number,
                        exc,
                    )
                    return False

            fps = max(1, int(round(self.fps)))
            if self.is_color:
                height, width, _ = self.data_shape
                stream_format = rs.format.rgb8
            else:
                height, width = self.data_shape
                # Use RGB stream and convert to grayscale later for compatibility
                stream_format = rs.format.rgb8

            self.config.enable_stream(rs.stream.color, width, height, stream_format, fps)

            self.pipeline_profile = self.pipeline.start(self.config)
            device = self.pipeline_profile.get_device()

            self.device_serial = self._safe_get_info(device, rs.camera_info.serial_number)
            self.device_product_line = self._safe_get_info(device, rs.camera_info.product_line)
            self.device_friendly_name = self._safe_get_info(device, rs.camera_info.name)
            if (
                self.requested_serial_number
                and self.device_serial
                and self.requested_serial_number != self.device_serial
            ):
                logger.warning(
                    "Requested RealSense serial %s but connected device reports serial %s",
                    self.requested_serial_number,
                    self.device_serial,
                )

            self.color_sensor = self._get_color_sensor(device)
            if self.color_sensor:
                self._set_camera_parameters()
            else:
                logger.warning("RealSense color sensor not found; hardware parameters not applied")

            # Warm up by grabbing a few frames
            for _ in range(5):
                self.pipeline.wait_for_frames()

            logger.info(
                "RealSense camera initialized: %s (%s) serial=%s",
                self.device_friendly_name or "Unknown device",
                self.device_product_line or "Unknown line",
                self.device_serial or "N/A",
            )
            self._log_camera_parameters()
            return True
        except Exception as exc:
            logger.error(f"RealSense camera initialization failed: {exc}")
            self._cleanup_camera()
            return False
    # Get information of device
    @staticmethod
    def _safe_get_info(device: rs.device, info_key: rs.camera_info) -> Optional[str]:
        try:
            return device.get_info(info_key)
        except Exception:
            return None
    # Get color sensor
    @staticmethod
    def _get_color_sensor(device: rs.device) -> Optional[rs.sensor]:
        try:
            sensors = list(device.query_sensors())
        except Exception:
            sensors = []

        for sensor in sensors:
            try:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    return sensor
            except Exception:
                continue

        return sensors[0] if sensors else None

    def _set_camera_parameters(self) -> None:
        """Apply requested hardware parameters when the sensor supports them."""
        if not self.color_sensor:
            return

        option_map = {
            "brightness": rs.option.brightness,
            "contrast": rs.option.contrast,
            "saturation": rs.option.saturation,
            "hue": rs.option.hue,
            "gamma": rs.option.gamma,
            "sharpness": rs.option.sharpness,
        }
        # Set param according to config
        for attr, option in option_map.items():
            value = getattr(self, attr)
            if value is None:
                continue

            try:
                if not self.color_sensor.supports(option):
                    logger.warning(f"Sensor does not support option {option.name.lower()}")
                    continue
                self.color_sensor.set_option(option, float(value))
            except Exception as exc:
                logger.warning(f"Failed to set {attr} to {value}: {exc}")

    def _log_camera_parameters(self) -> None:
        """Log current hardware parameter values."""
        if not self.color_sensor:
            logger.info("RealSense camera: No hardware parameters available")
            return

        option_map = {
            "brightness": rs.option.brightness,
            "contrast": rs.option.contrast,
            "saturation": rs.option.saturation,
            "hue": rs.option.hue,
            "gamma": rs.option.gamma,
            "sharpness": rs.option.sharpness,
        }

        params = []
        for name, option in option_map.items():
            if self.color_sensor.supports(option):
                try:
                    params.append(f"{name}={self.color_sensor.get_option(option):.2f}")
                except Exception:
                    params.append(f"{name}=unavailable")
            else:
                params.append(f"{name}=unsupported")

        logger.info("RealSense hardware parameters: %s", ", ".join(params))

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the RealSense camera."""
        if not self.pipeline:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.warning("RealSense color frame not available")
                return None, None

            timestamp_ns = time.time_ns()
            try:
                rs_timestamp = color_frame.get_timestamp()
                timestamp_ns = int(rs_timestamp * 1_000_000)
            except Exception:
                pass

            frame_array = np.asanyarray(color_frame.get_data())

            if not self.is_color:
                # Convert RGB data to grayscale using luminance weights
                frame_array = np.dot(frame_array[..., :3], [0.2989, 0.5870, 0.1140])
                frame_array = frame_array.astype(np.float32)

            target_height, target_width = self.data_shape[:2]
            if frame_array.shape[0] != target_height or frame_array.shape[1] != target_width:
                if not self._resize_warned:
                    logger.warning(
                        "RealSense frame size %s does not match target shape (%d, %d); "
                        "update configuration for optimal performance",
                        frame_array.shape[:2],
                        target_height,
                        target_width,
                    )
                    self._resize_warned = True
                frame_array = frame_array[:target_height, :target_width]

            if self.is_color and frame_array.shape[-1] == 4:
                frame_array = frame_array[..., :3]

            frame_array = frame_array.astype(self.data_dtype, copy=False)

            return timestamp_ns, frame_array

        except Exception as exc:
            logger.error(f"Error capturing RealSense frame: {exc}")
            return None, None

    def _cleanup_camera(self) -> None:
        """Stop the RealSense pipeline and release resources."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as exc:
                logger.warning(f"Error stopping RealSense pipeline: {exc}")
            finally:
                self.pipeline = None
                self.pipeline_profile = None
                self.color_sensor = None

    def _display_debug_frame(self, frame: np.ndarray) -> None:
        """Display the latest frame when debug mode is enabled."""
        if not self.debug or cv2 is None:
            return

        if self.is_color and frame.ndim == 3 and frame.shape[2] == 3:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display = frame

        cv2.imshow(self._debug_window_name, display)
        cv2.waitKey(1)

    def _close_debug_window(self) -> None:
        """Close the debug visualization window if it was opened."""
        if self.debug and cv2 is not None:
            try:
                cv2.destroyWindow(self._debug_window_name)
            except Exception:
                pass

    def start_server(self) -> None:
        """Start capturing frames and writing to shared memory."""
        if self.running:
            logger.info(f"RealSense camera {self.device_id} is already running")
            return

        logger.info(f"Starting RealSense camera server {self.device_id}...")
        self.running = True

        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"RealSense server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                    self._display_debug_frame(frame_array)
                wait_for_next_iteration()
            except Exception as exc:
                logger.error(f"Error in RealSense capture loop: {exc}")
                break

    def stop_server(self) -> None:
        """Stop the camera server."""
        if not self.running:
            return

        logger.info(f"Stopping RealSense camera server {self.device_id}...")
        self.running = False
        self._cleanup_shared_memory()
        self._cleanup_camera()
        self._close_debug_window()
        logger.info("RealSense camera server stopped")

    def close(self) -> None:
        """Release all resources."""
        self.stop_server()
        self._cleanup_camera()
        self._close_debug_window()
        logger.info(f"RealSense camera {self.device_id} closed")

    def __del__(self) -> None:
        try:
            self._cleanup_camera()
            if hasattr(self, "shared_memory") and self.shared_memory:
                self._cleanup_shared_memory()
            self._close_debug_window()
        except Exception:
            pass

    def get_device_info(self) -> Dict[str, Any]:
        """Get RealSense-specific device information."""
        info = super().get_device_info()
        info.update(
            {
                "camera_id": self.device_id,
                "is_color": self.is_color,
                "camera_opened": self.pipeline is not None,
                "requested_serial_number": self.requested_serial_number,
                "serial_number": self.device_serial,
                "product_line": self.device_product_line,
                "device_name": self.device_friendly_name,
                "hardware_parameters": {
                    "brightness": self.brightness,
                    "contrast": self.contrast,
                    "saturation": self.saturation,
                    "hue": self.hue,
                    "gamma": self.gamma,
                    "sharpness": self.sharpness,
                },
                "debug": self.debug,
            }
        )
        return info


def main() -> None:
    """Run the RealSense camera device server."""
    import argparse

    parser = argparse.ArgumentParser(description="Intel RealSense Camera Device Server")
    parser.add_argument("--device-id", "-d", type=int, default=0, 
                        help="Logical camera ID (default: 0)")
    parser.add_argument("--serial-number", "-n", type=str, default=None,
                        help="Specific RealSense serial number to open")
    parser.add_argument("--shape", "-s", type=str, default="480,640", 
                        help="Frame shape as comma-separated values")
    parser.add_argument("--fps", "-f", type=float, default=30.0, 
                        help="Frames per second (default: 30.0)")
    parser.add_argument("--dtype","-t",default="uint8",
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 
                                'float32', 'float64'],
                        help="Data type (default: uint8)",)
    parser.add_argument("--buffer-size", "-b", type=int, default=1, 
                        help="Buffer size in frames (default: 1)")
    parser.add_argument("--color", "-c", action="store_true", 
                        help="Enable color capture (3 channels)")
    parser.add_argument("--brightness", "-B", type=float, default=None, 
                        help="Camera brightness setting")
    parser.add_argument("--contrast", "-C", type=float, default=None, 
                        help="Camera contrast setting")
    parser.add_argument("--saturation", "-S", type=float, default=None, 
                        help="Camera saturation setting")
    parser.add_argument("--hue", "-H", type=float, default=None, 
                        help="Camera hue setting")
    parser.add_argument("--gamma", "-g", type=float, default=None, 
                        help="Camera gamma setting")
    parser.add_argument("--sharpness", "-x", type=float, default=None, 
                        help="Camera sharpness setting")
    parser.add_argument("--debug", action="store_true", 
                        help="Display captured frames in a debug window")

    args = parser.parse_args()

    try:
        data_shape = tuple(int(x.strip()) for x in args.shape.split(","))
        if args.color and len(data_shape) == 2:
            data_shape = data_shape + (3,)
        elif not args.color and len(data_shape) == 3:
            data_shape = data_shape[:2]
    except ValueError:
        logger.error(
            "Invalid shape format. Use comma-separated integers (e.g., '480,640' or '480,640,3')"
        )
        return

    from utils.shm_utils import get_dtype

    data_dtype = get_dtype(args.dtype)

    camera = RealSenseCameraDevice(
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
        sharpness=args.sharpness,
        debug=args.debug,
        realsense_serial=args.serial_number,
    )

    logger.info("RealSense Camera Device Server")
    logger.info("==============================")
    logger.info(f"Logical Camera ID: {args.device_id}")
    logger.info(f"Requested serial number: {args.serial_number or 'Any available'}")
    logger.info(f"Frame shape: {data_shape}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info(f"Color: {args.color}")
    logger.info(f"Debug visualization: {args.debug}")
    logger.info("Camera Hardware Parameters:")
    logger.info(f"  Brightness: {args.brightness if args.brightness is not None else 'Not set'}")
    logger.info(f"  Contrast: {args.contrast if args.contrast is not None else 'Not set'}")
    logger.info(f"  Saturation: {args.saturation if args.saturation is not None else 'Not set'}")
    logger.info(f"  Hue: {args.hue if args.hue is not None else 'Not set'}")
    logger.info(f"  Gamma: {args.gamma if args.gamma is not None else 'Not set'}")
    logger.info(f"  Sharpness: {args.sharpness if args.sharpness is not None else 'Not set'}")
    logger.info("")

    try:
        logger.info("RealSense camera server is running. Press Ctrl+C to stop...")
        logger.info(f"Camera info: {camera.get_device_info()}")
        camera.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
