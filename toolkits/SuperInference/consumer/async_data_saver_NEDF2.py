#!/usr/bin/env python3
"""
Asynchronous Data Saver Consumer - Save device manager data to NEDF2 format.

This module provides data saving functionality by consuming device manager data
and storing it to NEDF2 format using NEDF SDK.

Author:Dong Jiulong
"""

import sys
import time
import signal
import argparse
import os
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import queue
import threading

# NEDF SDK imports
from nmx_nedf_api import (
    NEDFFactory,
    NEDFWriterConfig,
    NEDFMetadata,
    NEDFTopicsConfig,
    NEDFType
)
from nmx_msg.Lowdim_pb2 import LowdimData
from nmx_msg.Image_pb2 import Image

# Ensure project root is on PYTHONPATH when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger
from utils.time_control import precise_loop_timing

try:
    from .data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from .data_saver import wait_enter_blocking, enter_pressed_nonblocking
except ImportError:
    from data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from data_saver import wait_enter_blocking, enter_pressed_nonblocking


class DataSaverConsumer(SyncDataSaverConsumer):
    """
    Data saver consumer for storing device manager data to NEDF2 format.

    This consumer saves data using NEDF SDK:
    - Camera devices: Saved as JPG compressed images in MCAP format
    - Robot devices: Saved as lowdim data (TCP, Joint, etc.)
    - Other devices: Saved as custom lowdim topics
    - All data stored in NEDF2 format (MCAP-based) with metadata

    Features:
    - Async writing for high-frequency data collection
    - Automatic camera parameter writing
    - Support for image resizing
    - NEDF2 metadata generation
    """

    def __init__(self, summary_shm_name: str = "device_summary_data",
                 output_dir: str = "saved_data", save_interval: float = 0.04,
                 start_idx: int = 0, image_size: Optional[Tuple[int, int]] = None,
                 async_write: bool = True, queue_size: int = 5000,
                 robot_type: str = "Rizon", task_name: str = "data_collection") -> None:
        """
        Initialize the data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            start_idx: Starting episode index
            image_size: Target image size (width, height) for resizing
            async_write: Whether to use async writing thread (default: True)
            queue_size: Size of async write queue (default: 1000)
            robot_type: Robot type for NEDF metadata
            task_name: Task name for NEDF metadata
        """
        super().__init__(summary_shm_name, output_dir, save_interval, start_idx, image_size)

        # Async writing infrastructure
        self.async_write = async_write
        self.write_queue = queue.Queue(maxsize=queue_size)
        self.write_thread = None
        self.write_thread_running = False

        # NEDF2 writer management
        self.nedf_writer = None
        self.nedf_writer_context = None  # Store the context manager
        self.current_episode_dir = None
        self.episode_active = False
        self.device_data_counts: Dict[str, int] = {}

        # NEDF metadata configuration
        self.robot_type = robot_type
        self.task_name = task_name

        # Camera ID mapping (device_key -> camera_id)
        self.camera_id_mapping: Dict[str, str] = {}

        # Track if camera intrinsics/extrinsics have been written
        self.camera_params_written: Dict[str, bool] = {}

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def open_episode_streams(self, episode_dir: str) -> None:
        """
        Open NEDF2 writer for the current episode.

        Args:
            episode_dir: Directory path for this episode
        """
        try:
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_dir = episode_dir

            # Build camera_info dictionary from devices
            camera_info = {}
            for device_key in self.devices:
                device_info = self.devices[device_key]
                device_type = device_info['type']

                if device_type in ['OpenCVCameraDevice', 'HikCameraDevice']:
                    # Use device_key as camera_id
                    camera_id = device_key
                    camera_info[camera_id] = device_info['name']
                    self.camera_id_mapping[device_key] = camera_id
                    self.camera_params_written[camera_id] = False
                    logger.debug(f"Registered camera: {camera_id} -> {device_info['name']}")

            # Set main camera (first camera in the list)
            main_camera = list(camera_info.keys())[0] if camera_info else "CAMID_1"

            # Create NEDF metadata
            metadata = NEDFMetadata(
                task_name=self.task_name,
                robot_vendor_model=f"Flexiv {self.robot_type}",
                data_format=NEDFType.NEDF2,
                mcap_file_name="data.mcap",
                mcap_file_type="CHUNK_FILE",
                end_effector="Gripper",
                robot_type=self.robot_type,
                arm_base_to_world_transform=[1.0, 0.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0, 0.0,
                                             0.0, 0.0, 1.0, 0.0,
                                             0.0, 0.0, 0.0, 1.0],
                camera_info=camera_info,
                main_camera=main_camera,
                extra_keys=[]
            )

            # Create NEDF writer configuration
            metadata_path = os.path.join(episode_dir, "metadata.json")
            writer_config = NEDFWriterConfig(
                metadata_file_path=metadata_path,
                metadata=metadata,
                topics_config=NEDFTopicsConfig(),
                max_file_size_mb=100,
                mcap_chunk_size_kb=256
            )

            # Create NEDF2 writer (context manager)
            self.nedf_writer_context = NEDFFactory.get_writer(NEDFType.NEDF2, writer_config)
            # Manually enter the context manager and save the actual writer object
            self.nedf_writer = self.nedf_writer_context.__enter__()

            # Initialize data counters
            for device_key in self.devices:
                self.device_data_counts[device_key] = 0

            self.episode_active = True
            logger.info(f"Opened NEDF2 writer in {episode_dir}")
            logger.info(f"Cameras: {list(camera_info.keys())}")

        except Exception as e:
            logger.error(f"Error opening NEDF2 writer: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close_episode_streams(self) -> None:
        """Close NEDF2 writer."""
        try:
            if self.nedf_writer_context is not None:
                # Exit context manager to finalize NEDF2 file
                self.nedf_writer_context.__exit__(None, None, None)
                self.nedf_writer = None
                self.nedf_writer_context = None

                # Log statistics
                for device_key, count in self.device_data_counts.items():
                    logger.debug(f"Wrote {count} frames for {device_key}")

                logger.info("Closed NEDF2 writer")

            self.device_data_counts.clear()
            self.camera_id_mapping.clear()
            self.camera_params_written.clear()
            self.episode_active = False

        except Exception as e:
            logger.error(f"Error closing NEDF2 writer: {e}")
            import traceback
            traceback.print_exc()

    def _async_write_worker(self) -> None:
        """
        Background worker thread that continuously consumes data from write queue.

        This thread runs independently, writing data to NEDF2 as it becomes available.
        """
        logger.info("Async write worker thread started")

        while self.write_thread_running:
            try:
                # Block for up to 0.1s waiting for data
                write_task = self.write_queue.get(timeout=0.1)

                # Extract task data
                device_key = write_task['device_key']
                device_type = write_task['device_type']
                device_name = write_task['device_name']
                timestamp_ns = write_task['timestamp_ns']
                data_array = write_task['data_array']
                device_info = write_task['device_info']

                # Check if episode is active and writer is open
                if not self.episode_active or self.nedf_writer is None:
                    logger.warning(f"Episode not active or writer not open for {device_key}, dropping data")
                    self.write_queue.task_done()
                    continue

                # Process based on device type
                if device_type in ['OpenCVCameraDevice', 'HikCameraDevice']:
                    # Handle camera data
                    camera_id = self.camera_id_mapping.get(device_key)
                    if camera_id is None:
                        logger.warning(f"Camera ID not found for {device_key}")
                        self.write_queue.task_done()
                        continue

                    # Write camera intrinsics/extrinsics once per episode
                    if not self.camera_params_written.get(camera_id, False):
                        self._write_camera_params(camera_id, timestamp_ns)
                        self.camera_params_written[camera_id] = True

                    # Resize image if needed
                    if self.image_size is not None:
                        target_width, target_height = self.image_size
                        original_shape = data_array.shape
                        data_array = cv2.resize(data_array, (target_width, target_height),
                                                interpolation=cv2.INTER_LINEAR)
                        logger.debug(f"Resized image from {original_shape[:2]} to {data_array.shape[:2]}")

                    # Convert RGB to BGR for OpenCV encoding
                    # Images from devices are stored in RGB format, but cv2.imencode expects BGR
                    if len(data_array.shape) == 3 and data_array.shape[2] == 3:
                        data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2BGR)

                    # Encode image as JPG
                    _, jpg_bytes = cv2.imencode('.jpg', data_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Create Image protobuf message
                    jpg_image = Image()
                    jpg_image.data = jpg_bytes.tobytes()

                    # Write to NEDF2
                    self.nedf_writer.write_jpg(
                        camera_id=camera_id,
                        jpg=jpg_image,
                        log_timestamp_ns=timestamp_ns,
                        publish_timestamp_ns=timestamp_ns
                    )

                elif device_type in ['FlexivRobotDevice', 'SimulatedRobotDevice']:
                    # Handle robot data - write as TCP or Joint data
                    # Convert numpy array to list
                    if isinstance(data_array, np.ndarray):
                        data_list = data_array.flatten().tolist()
                    else:
                        data_list = list(data_array)

                    # Determine if this is TCP or Joint data based on device_key
                    if 'tcp' in device_key.lower() or 'eef' in device_key.lower():
                        # TCP/EEF data handling with split-and-save logic
                        data_length = len(data_list)

                        if data_length == 8:
                            # Split: Pose (7D: x,y,z,qw,qx,qy,qz) + Gripper (1D: width)
                            # Action A: Save Pose data using write_tcp
                            pose_data = LowdimData()
                            pose_data.data.extend(data_list[:7])  # [x, y, z, qw, qx, qy, qz]
                            self.nedf_writer.write_tcp(pose_data, timestamp_ns, timestamp_ns)

                            # Action B: Save Gripper data using write_ee_state
                            gripper_data = LowdimData()
                            gripper_data.data.extend([data_list[7]])  # Gripper width
                            self.nedf_writer.write_ee_state(gripper_data, timestamp_ns, timestamp_ns)

                            logger.debug(f"Split TCP data for {device_key}: pose(7D) + gripper(1D)")

                        elif data_length == 7:
                            # Pure Pose data (no gripper)
                            pose_data = LowdimData()
                            pose_data.data.extend(data_list)  # [x, y, z, qw, qx, qy, qz]
                            self.nedf_writer.write_tcp(pose_data, timestamp_ns, timestamp_ns)

                        else:
                            # Unexpected length - fallback to generic lowdim
                            logger.warning(f"Unexpected TCP/EEF data length {data_length} for {device_key}, "
                                           f"using generic lowdim write")
                            lowdim_data = LowdimData()
                            lowdim_data.data.extend(data_list)
                            topic = f"/lowdim/{device_key}"
                            self.nedf_writer.write_lowdim(topic, lowdim_data, timestamp_ns, timestamp_ns)

                    elif 'joint' in device_key.lower():
                        # Joint data handling
                        joint_data = LowdimData()
                        joint_data.data.extend(data_list)
                        self.nedf_writer.write_joint(joint_data, timestamp_ns, timestamp_ns)

                    else:
                        # Default: write as custom lowdim data
                        lowdim_data = LowdimData()
                        lowdim_data.data.extend(data_list)
                        topic = f"/lowdim/{device_key}"
                        self.nedf_writer.write_lowdim(topic, lowdim_data, timestamp_ns, timestamp_ns)

                elif device_type in ['ViveTrackerDevice', 'TeleoperatorDevice', 'RotaryEncoderDevice']:
                    # Handle other devices as lowdim data
                    lowdim_data = LowdimData()

                    if isinstance(data_array, np.ndarray):
                        data_list = data_array.flatten().tolist()
                    else:
                        data_list = list(data_array)

                    lowdim_data.data.extend(data_list)

                    # Write as custom lowdim topic
                    topic = f"/lowdim/{device_key}"
                    self.nedf_writer.write_lowdim(topic, lowdim_data, timestamp_ns, timestamp_ns)

                else:
                    logger.warning(f"Unknown device type: {device_type} for {device_key}")
                    # Still mark task as done to avoid queue blocking
                    self.write_queue.task_done()
                    continue

                # Update counter
                self.device_data_counts[device_key] += 1

                # Log progress periodically
                if self.device_data_counts[device_key] % 100 == 0:
                    logger.debug(f"{device_key}: wrote {self.device_data_counts[device_key]} frames")

                self.write_queue.task_done()

            except queue.Empty:
                # No data available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in async write worker: {e}")
                import traceback
                traceback.print_exc()

        logger.info("Async write worker thread stopped")

    def _write_camera_params(self, camera_id: str, timestamp_ns: int) -> None:
        """
        Write camera intrinsics and extrinsics to NEDF2.

        Args:
            camera_id: Camera identifier
            timestamp_ns: Timestamp in nanoseconds
        """
        try:
            # Write camera intrinsics (default 3x3 identity-like matrix)
            # TODO: Load actual camera intrinsics from calibration files
            intrinsic = LowdimData()
            # Default intrinsics: fx=800, fy=800, cx=320, cy=240
            intrinsic.data.extend([800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0])

            self.nedf_writer.write_camera_intrinsic(
                camera_id=camera_id,
                camera_intrinsic=intrinsic,
                log_timestamp_ns=timestamp_ns,
                publish_timestamp_ns=timestamp_ns
            )

            # Write camera extrinsics (default identity transform)
            # TODO: Load actual camera extrinsics from calibration files
            from nmx_msg.Image_pb2 import CameraExtrinsic as ProtoCameraExtrinsic

            extrinsic = ProtoCameraExtrinsic()
            extrinsic.is_global = True
            extrinsic.parent_link_name = "base_link"
            extrinsic.pose_in_link.extend([1.0, 0.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0, 0.0,
                                           0.0, 0.0, 1.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0])
            extrinsic.error = 0.0

            self.nedf_writer.write_camera_extrinsic(
                camera_id=camera_id,
                camera_extrinsic=extrinsic,
                log_timestamp_ns=timestamp_ns,
                publish_timestamp_ns=timestamp_ns
            )

            logger.debug(f"Wrote camera parameters for {camera_id}")

        except Exception as e:
            logger.error(f"Error writing camera parameters for {camera_id}: {e}")
            import traceback
            traceback.print_exc()

    def start_async_writer(self) -> None:
        """Start the background async writer thread."""
        if self.async_write and not self.write_thread_running:
            self.write_thread_running = True
            self.write_thread = threading.Thread(target=self._async_write_worker, daemon=True)
            self.write_thread.start()
            logger.info("Started async write thread")

    def stop_async_writer(self) -> None:
        """Stop the background async writer thread and close file streams."""
        if self.write_thread_running:
            logger.info("Stopping async write thread...")
            self.write_thread_running = False

            if self.write_thread:
                self.write_thread.join(timeout=5.0)
                if self.write_thread.is_alive():
                    logger.warning("Async write thread did not stop gracefully")
                else:
                    logger.info("Async write thread stopped")

            # Process any remaining items in queue
            remaining = 0
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    remaining += 1
                except queue.Empty:
                    break

            if remaining > 0:
                logger.warning(f"Discarded {remaining} items from write queue")

            # Close any open file streams
            if self.episode_active:
                self.close_episode_streams()

    def collect_data(self) -> None:
        """
        Collect data from all devices in a non-blocking manner.

        Each device is polled independently:
        - High-frequency devices (e.g., cameras) update frequently
        - Low-frequency devices (e.g., robot states) update less often
        - Only new data (based on timestamp changes) is collected
        - Data is either queued for async writing or buffered for batch saving
        """
        current_time = time.time()

        # Read each device independently - non-blocking
        # This allows high-freq devices to update without waiting for low-freq ones
        for device_key in self.devices:
            # Read latest data for this device from shared memory
            # _read_device_data() is non-blocking - just reads from SHM buffer
            data = self._read_device_data(device_key)

            if data is not None:
                timestamp_ns, data_array = data

                # Only process if this is new data (timestamp changed)
                if timestamp_ns != self.devices[device_key]['last_timestamp']:

                    # Option 1: Async writing - queue data for background thread
                    if self.async_write and self.write_thread_running:
                        try:
                            # Try non-blocking put first
                            device_info = self.devices[device_key]
                            write_task = {
                                'device_key': device_key,
                                'device_type': device_info['type'],
                                'device_name': device_info['name'],
                                'timestamp_ns': timestamp_ns,
                                'data_array': data_array.copy(),  # Copy to avoid race conditions
                                'device_info': device_info,
                                'collect_time': current_time
                            }
                            self.write_queue.put_nowait(write_task)
                        except queue.Full:
                            # Queue is full, try blocking put with timeout (10ms)
                            # This gives the writer thread a chance to process some data
                            try:
                                self.write_queue.put(write_task, timeout=0.01)
                            except queue.Full:
                                # Still full after timeout, drop this data point
                                logger.warning(f"Write queue full after timeout, dropping data for {device_key} "
                                               f"(queue size: {self.write_queue.qsize()}/{self.write_queue.maxsize})")

                    # Option 2: Buffer data for batch saving (original behavior)
                    else:
                        self.data_buffer[device_key].append((timestamp_ns, data_array))

                    # Update tracking
                    self.devices[device_key]['last_timestamp'] = timestamp_ns
                    self.devices[device_key]['last_update_time'] = current_time

        # Note: We do NOT use read_all_device_data() here because:
        # 1. It reads ALL devices from the SAME frame timestamp
        # 2. This would force all devices to be synchronized
        # 3. We want each device to update independently at its own frequency

    def _finalize_nedf2_episode(self, episode_dir: str) -> None:
        """
        Finalize NEDF2 episode by renaming directory.
        If target directory exists, it will be overwritten.

        Args:
            episode_dir: Directory containing NEDF2 files
        """
        try:
            import shutil

            # Rename episode directory from temp to final name
            final_episode_dir = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}")

            if os.path.exists(episode_dir):
                # Remove existing final directory if it exists
                if os.path.exists(final_episode_dir):
                    logger.warning(f"Overwriting existing episode: {final_episode_dir}")
                    shutil.rmtree(final_episode_dir)

                os.rename(episode_dir, final_episode_dir)
                logger.info(f"Finalized NEDF2 episode: {final_episode_dir}")

            self.episode_counts += 1

        except Exception as e:
            logger.error(f"Error finalizing NEDF2 episode: {e}")
            import traceback
            traceback.print_exc()

    def start_saving(self) -> None:
        """Start the data saving process."""
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            return

        self.running = True

        # Start async writer if enabled
        if self.async_write:
            self.start_async_writer()
            logger.info("Async write mode enabled - data will be written continuously to disk")
        else:
            logger.info("Batch mode enabled - data will be saved at intervals")

        logger.info("Start data acquisition. Ctrl+C to exit.")
        logger.info("+" * 40)

        try:
            while self.running:
                logger.info("Press Enter to start data acquisition.")
                wait_enter_blocking()
                data_folders = []
                logger.info("Data acquisition started. Press Enter to stop.")
                time_episode_start = time.time()

                # Async mode: Open file streams for this episode
                if self.async_write:
                    episode_dir = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}_temp")
                    self.open_episode_streams(episode_dir)

                # Create precise timing function for collection loop
                # In async mode, this is just the polling rate
                # In batch mode, this is the save interval
                collection_interval = 0.001 if self.async_write else self.save_interval
                wait_for_next_collect = precise_loop_timing(collection_interval)

                while True:
                    # Collect data from devices - non-blocking, independent per device
                    self.collect_data()

                    # Batch mode: save data at intervals
                    if not self.async_write:
                        if (self.frame_timestamp is not None and
                                self.last_saving_timestamp != self.frame_timestamp and
                                any(self.data_buffer.values())):

                            saved_data_folder = self.save_data()
                            if saved_data_folder is not None:
                                data_folders.append(saved_data_folder)
                                self.last_saving_timestamp = self.frame_timestamp

                            if len(data_folders) != len(set(data_folders)):
                                logger.error("Data folders have redundancy")

                    # Wait for next collection interval
                    wait_for_next_collect()

                    if enter_pressed_nonblocking():
                        logger.info("Data acquisition stopped by user.")
                        break

                # Batch mode: save any remaining data
                if not self.async_write:
                    if all(self.data_buffer.values()) and self.last_saving_timestamp != self.frame_timestamp:
                        saved_data_folder = self.save_data()
                        if saved_data_folder is not None and saved_data_folder not in data_folders:
                            data_folders.append(saved_data_folder)

                time_episode_end = time.time()

                # Async mode: Close writer and finalize NEDF2
                if self.async_write:
                    logger.info("Waiting for async writer to finish...")
                    self.write_queue.join()  # Wait for all queued data to be written
                    logger.info("All data written to NEDF2")

                    # Close NEDF2 writer
                    self.close_episode_streams()

                    # Finalize NEDF2 episode
                    logger.info("Finalizing NEDF2 episode...")
                    self._finalize_nedf2_episode(episode_dir)
                else:
                    # Batch mode not supported for NEDF2
                    logger.warning("Batch mode is not supported for NEDF2 format")

                logger.info(f"Episode completed in {time.time() - time_episode_start:.2f}s")

        except KeyboardInterrupt:
            logger.info("Data saving interrupted by user")

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the data saver consumer and async writer."""
        if not self.running:
            return

        logger.info("Stopping data saver consumer...")

        # Stop async writer first
        self.stop_async_writer()

        # Call parent stop method
        super().stop()

        logger.info("Data saver consumer stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of the data saver consumer."""
        status = {
            'connected': self.summary_shm is not None,
            'running': self.running,
            'devices': len(self.devices),
            'output_dir': self.output_dir,
            'save_interval': self.save_interval,
            'frame_timestamp': self.frame_timestamp,
            'async_write': self.async_write,
            'write_thread_running': self.write_thread_running,
            'queue_size': self.write_queue.qsize() if self.async_write else 0
        }

        # Add buffer information
        buffer_info = {}
        for device_key, buffer_data in self.data_buffer.items():
            buffer_info[device_key] = len(buffer_data)
        status['buffer_samples'] = buffer_info

        return status


def main() -> None:
    """Main function to run the data saver consumer."""
    parser = argparse.ArgumentParser(description="Data Saver Consumer - Save Device Data to NEDF2 Format")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--output-dir", "-o", default="saved_data",
                        help="Output directory for saved data (default: saved_data)")
    parser.add_argument("--interval", "-i", type=float, default=0.001,
                        help="Save interval in seconds (default: 0.001)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size for camera data as 'width,height' (e.g., '400,300'). "
                             "If not specified, save at original size.")
    parser.add_argument("--async-write", "-a", action="store_true", default=True,
                        help="Use async write mode (default: True)")
    parser.add_argument("--batch-mode", "-b", action="store_true",
                        help="Use batch mode instead of async write (NOT SUPPORTED for NEDF2)")
    parser.add_argument("--queue-size", "-q", type=int, default=5000,
                        help="Async write queue size (default: 5000)")
    parser.add_argument("--robot-type", "-r", type=str, default="Rizon",
                        help="Robot type for NEDF metadata (default: Rizon)")
    parser.add_argument("--task-name", "-t", type=str, default="data_collection",
                        help="Task name for NEDF metadata (default: data_collection)")
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")

    args = parser.parse_args()

    # Determine write mode (NEDF2 requires async mode)
    async_write = not args.batch_mode if args.batch_mode else args.async_write

    if not async_write:
        logger.warning("NEDF2 format requires async write mode. Enabling async mode.")
        async_write = True

    # Parse image size
    image_size = None
    if args.image_size:
        try:
            width, height = map(int, args.image_size.split(','))
            image_size = (width, height)
            logger.info(f"Target image size: {image_size} (width, height)")
        except ValueError:
            logger.error(f"Invalid image size format: {args.image_size}. Expected 'width,height' (e.g., '400,300')")
            return

    # Create data saver consumer
    consumer = DataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=args.start_idx,
        image_size=image_size,
        async_write=async_write,
        queue_size=args.queue_size,
        robot_type=args.robot_type,
        task_name=args.task_name
    )

    if args.status:
        # Show status
        status = consumer.get_status()
        logger.info("Data Saver Consumer Status:")
        logger.info(f"Connected: {status['connected']}")
        logger.info(f"Running: {status['running']}")
        logger.info(f"Devices: {status['devices']}")
        logger.info(f"Output directory: {status['output_dir']}")
        logger.info(f"Save interval: {status['save_interval']}s")
        logger.info(f"Async write mode: {status['async_write']}")
        logger.info(f"Write thread running: {status['write_thread_running']}")
        logger.info(f"Write queue size: {status['queue_size']}")
        logger.info(f"Frame timestamp: {status['frame_timestamp']}")

        if status['buffer_samples']:
            logger.info("\nBuffer samples:")
        for device_key, samples in status['buffer_samples'].items():
            logger.info(f"  {device_key}: {samples} samples")
        return

    logger.info("Data Saver Consumer - Save Device Data to NEDF2 Format")
    logger.info("========================================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save interval: {args.interval}s")
    logger.info(f"Robot type: {args.robot_type}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Write mode: Async (NEDF2 streaming)")
    logger.info(f"Queue size: {args.queue_size}")
    if image_size:
        logger.info(f"Image resize: {image_size[0]}x{image_size[1]}")
    logger.info("")

    # Try to connect
    if not consumer.connect():
        logger.error("Failed to connect to summary SHM. Make sure the Device Manager is running.")
        return

    try:
        # consumer.start_saving()
        consumer.start_saving()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()