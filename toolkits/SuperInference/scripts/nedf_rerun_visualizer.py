"""
NEDF Episode Visualizer (Final Optimized Version)

This module provides a rerun-based visualizer for NEDF2 episode data.
It reads data using nmx_nedf_api and visualizes it using the Rerun SDK,
with a GUI interface for easy episode selection and visualization.

Improvements:
    - Fixes memory leak by forcing Garbage Collection (GC) after each episode.
    - Fixes Rerun lag (Issue #11585) by using unique recording_ids for every session.
    - Limits Rerun Viewer memory usage to 4GB using official API.

changed:
    - rotate the arm poses around the y-axis by 150 degrees
    - set the view coordinates to RIGHT_HAND_Y_UP

Usage:
    python scripts/nedf_rerun_visualizer.py

Author: Junjie Xu
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import cv2
import rerun as rr
import sys
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import gc        # [Fix] For memory cleanup
import uuid      # [Fix] For unique recording IDs

from rerun.datatypes import Angle, RotationAxisAngle
# Handle imports for project structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger
from utils.rerun_visualization import visualize_trajectory_with_colors

# NEDF SDK imports
try:
    from nmx_nedf_api import NEDFFactory, NEDFReaderConfig
except ImportError:
    logger.error("Could not import nmx_nedf_api. Please ensure NEDF SDK is installed/in path.")
    sys.exit(1)

# ============================================================================
# NEDF Data Loading Logic
# ============================================================================

def load_nedf_episode(episode_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load an NEDF2 episode using nmx_nedf_api.
    """
    if not episode_path.exists():
        logger.error(f"Episode path does not exist: {episode_path}")
        return {}

    metadata_path = episode_path / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"metadata.json not found in {episode_path}")
        return {}

    devices = {}
    nedf_reader = None

    try:
        # Create NEDF reader
        reader_config = NEDFReaderConfig(metadata_file_path=str(metadata_path))
        nedf_reader = NEDFFactory.get_reader(reader_config)

        # --- 1. Timestamp Synchronization ---
        all_timestamps_ns = []
        
        # Get camera timestamps
        camera_info = nedf_reader.metadata.camera_info
        for camera_id in camera_info.keys():
            ts = nedf_reader.get_raw_image_timestamps(camera_id)
            if len(ts) > 0:
                all_timestamps_ns.extend(ts)
        
        # Get lowdim timestamps
        all_topics_timestamps = nedf_reader.get_raw_topics_timestamps()
        for timestamps_ns in all_topics_timestamps.values():
            if len(timestamps_ns) > 0:
                all_timestamps_ns.extend(timestamps_ns)
        
        min_timestamp_ns = min(all_timestamps_ns) if all_timestamps_ns else 0
        logger.info(f"Global Start Time: {min_timestamp_ns} ns")

        # --- 2. Load Cameras ---
        for camera_id, camera_name in camera_info.items():
            img_list = []
            img_timestamps_list = []
            
            # Iterate and decode images
            for msg in nedf_reader.iter_color(camera_id):
                log_ts_ns = msg.message.log_time
                jpg_image = msg.decoded_message
                
                # Decode JPG
                jpg_bytes = np.frombuffer(jpg_image.data, dtype=np.uint8)
                img = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Convert BGR (OpenCV) to RGB (Rerun)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_list.append(img)
                    img_timestamps_list.append(log_ts_ns)
            
            if img_list:
                timestamps_s = (np.array(img_timestamps_list) - min_timestamp_ns) / 1e9
                
                devices[camera_id] = {
                    'name': camera_id,
                    'type': 'camera',
                    'timestamps': timestamps_s,
                    'data': np.array(img_list), # [T, H, W, C]
                }
                logger.info(f"Loaded Camera {camera_id}: {len(img_list)} frames")

        # --- 3. Load Lowdim Topics ---
        for topic, timestamps_ns in all_topics_timestamps.items():
            if '/lowdim/' not in topic:
                continue

            device_key = topic.replace('/lowdim/', '')
            
            # Read data
            data_list = []
            timestamps_list = []
            
            try:
                for msg in nedf_reader.get_lowdim(topic):
                    log_ts_ns = msg.message.log_time
                    lowdim_data = msg.decoded_message
                    data_array = np.array(lowdim_data.data, dtype=np.float32)
                    
                    data_list.append(data_array)
                    timestamps_list.append(log_ts_ns)
            except Exception as e:
                logger.error(f"Error reading topic {topic}: {e}")
                continue

            if not data_list:
                continue

            # Process Data
            data_array = np.stack(data_list)
            timestamps_s = (np.array(timestamps_list) - min_timestamp_ns) / 1e9
            
            # --- Logic for Device Type Classification & Splitting ---
            
            # Case A: Vive Tracker (14D -> Split Left/Right)
            if 'vivetrackerdevice' in device_key.lower() and (
                (data_array.ndim == 2 and data_array.shape[1] == 14) or 
                (data_array.ndim == 2 and data_array.shape[1] == 1 and data_array.size/len(data_array) >= 14)
            ):
                if data_array.shape[1] != 14:
                    try:
                        data_array = data_array.reshape(len(data_array), 14)
                    except:
                        logger.warning(f"Could not reshape Vive data {device_key}")
                        continue

                # Left (First 7)
                devices[f"{device_key}_left"] = {
                    'name': f"{device_key}_left",
                    'type': 'vive_pose',
                    'timestamps': timestamps_s,
                    'data': data_array[:, :7], 
                }
                
                # Right (Last 7)
                devices[f"{device_key}_right"] = {
                    'name': f"{device_key}_right",
                    'type': 'vive_pose',
                    'timestamps': timestamps_s,
                    'data': data_array[:, 7:14],
                }

            # Case B: Standard Pose (7D)
            elif data_array.ndim == 2 and data_array.shape[1] == 7:
                devices[device_key] = {
                    'name': device_key,
                    'type': 'vive_pose',
                    'timestamps': timestamps_s,
                    'data': data_array,
                }

            # Case C: Rotary Encoder / Gripper (1D)
            elif 'rotaryencoderdevice' in device_key.lower() or data_array.ndim == 1 or (data_array.ndim == 2 and data_array.shape[1] == 1):
                if data_array.ndim == 1:
                    data_array = data_array.reshape(-1, 1)
                
                devices[device_key] = {
                    'name': device_key,
                    'type': 'timeseries',
                    'timestamps': timestamps_s,
                    'data': data_array,
                }
            
            else:
                # Unknown/Generic
                devices[device_key] = {
                    'name': device_key,
                    'type': 'unknown',
                    'timestamps': timestamps_s,
                    'data': data_array
                }

    except Exception as e:
        logger.error(f"Critical error loading NEDF episode: {e}")
        import traceback
        traceback.print_exc()
    
    return devices


# ============================================================================
# Visualizer Class (UI Logic)
# ============================================================================

class NEDFEpisodeVisualizer:
    """
    Visualizer for parsed NEDF data using Rerun.
    """
    
    def __init__(self, devices: Dict[str, Dict[str, Any]], episode_name: str):
        self.devices = devices
        self.episode_name = episode_name
        self._parse_data()
        
    def _parse_data(self):
        self.pose_timestamps = None 
        self.arm_poses = {} 
        self.camera_data = {} 
        self.gripper_data = {} 
        
        cameras = [d for k, d in self.devices.items() if d['type'] == 'camera']
        cameras.sort(key=lambda x: x['name'])
        
        for cam in cameras:
            self.camera_data[cam['name']] = {
                'data': cam['data'],
                'timestamps': cam['timestamps']
            }

        poses = [d for k, d in self.devices.items() if d['type'] == 'vive_pose']
        for pose in poses:
            self.arm_poses[pose['name']] = {
                'data': pose['data'],
                'timestamps': pose['timestamps']
            }
            if self.pose_timestamps is None:
                self.pose_timestamps = pose['timestamps']

        grippers = [d for k, d in self.devices.items() if d['type'] == 'timeseries']
        for gripper in grippers:
            self.gripper_data[gripper['name']] = {
                'data': gripper['data'],
                'timestamps': gripper['timestamps']
            }

        if self.pose_timestamps is None:
            for dev in self.devices.values():
                if len(dev['timestamps']) > 0:
                    self.pose_timestamps = dev['timestamps']
                    break

    def log_images(self, frame_idx_map: Dict[str, int], timestamp: float):
        for cam_name, cam_info in self.camera_data.items():
            idx = frame_idx_map.get(cam_name, -1)
            if idx >= 0 and idx < len(cam_info['data']):
                rr.log(f"/cameras/{cam_name}", rr.Image(cam_info['data'][idx]))
    
    def log_arm_poses(self, frame_idx_map: Dict[str, int], timestamp: float):
        pose_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        for arm_name, arm_info in self.arm_poses.items():
            idx = frame_idx_map.get(arm_name, -1)
            if idx >= 0 and idx < len(arm_info['data']):
                pose = arm_info['data'][idx]
                for i, label in enumerate(pose_labels):
                    rr.log(f"/arms/{arm_name}/{label}", rr.Scalars(np.array([float(pose[i])])))
    
    def log_grippers(self, frame_idx_map: Dict[str, int], timestamp: float):
        for grip_name, grip_info in self.gripper_data.items():
            idx = frame_idx_map.get(grip_name, -1)
            if idx >= 0 and idx < len(grip_info['data']):
                val = grip_info['data'][idx]
                rr.log(f"/grippers/{grip_name}", rr.Scalars(np.array([float(val[0] if val.ndim>0 else val)])))
    
    def visualize_episode(self, fps: float = 30.0):
        if self.pose_timestamps is None or len(self.pose_timestamps) == 0:
            logger.warning("No data found to visualize.")
            return
            
        logger.info(f"Visualizing episode: {self.episode_name} ({len(self.pose_timestamps)} frames)")
        
        # Calculate trajectory center
        trajectory_center = np.zeros(3)
        count = 0
        for arm_name, arm_info in self.arm_poses.items():
            trajectory_center += np.mean(arm_info['data'][:, :3], axis=0)
            count += 1
        if count > 0:
            trajectory_center /= count
        
        # Setup coordinates
        rr.set_time_sequence("static_init", 0)
        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        rr.log("/world/origin/axes", rr.Arrows3D(
            vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
            origins=[[0,0,0], [0,0,0], [0,0,0]],
            colors=[[255,0,0], [0,255,0], [0,0,255]],
            radii=[0.003]*3
        ), static=True)
        
        arm_colors = [
            [100, 150, 255], [255, 100, 100], [100, 255, 100], 
            [255, 200, 50], [200, 100, 255], [255, 150, 150]
        ]
        
        # Main Loop
        for frame_idx in range(len(self.pose_timestamps)):
            timestamp_sec = self.pose_timestamps[frame_idx]
            rr.set_time_seconds("timestamp", timestamp_sec)
            
            frame_idx_map = {}
            
            for arm_name, arm_info in self.arm_poses.items():
                if frame_idx < len(arm_info['data']):
                    frame_idx_map[arm_name] = frame_idx
            
            for grip_name, grip_info in self.gripper_data.items():
                g_idx = np.searchsorted(grip_info['timestamps'], timestamp_sec)
                g_idx = min(g_idx, len(grip_info['data']) - 1)
                frame_idx_map[grip_name] = g_idx
            
            # Camera deduplication
            prev_cam_indices = {}
            if frame_idx > 0:
                prev_timestamp = self.pose_timestamps[frame_idx - 1]
                for cam_name, cam_info in self.camera_data.items():
                    prev_idx = np.searchsorted(cam_info['timestamps'], prev_timestamp)
                    prev_idx = min(prev_idx, len(cam_info['data']) - 1)
                    prev_cam_indices[cam_name] = prev_idx
            
            for cam_name, cam_info in self.camera_data.items():
                img_idx = np.searchsorted(cam_info['timestamps'], timestamp_sec)
                img_idx = min(img_idx, len(cam_info['data']) - 1)
                if cam_name not in prev_cam_indices or prev_cam_indices[cam_name] != img_idx:
                    frame_idx_map[cam_name] = img_idx
            
            self.log_arm_poses(frame_idx_map, timestamp_sec)
            self.log_grippers(frame_idx_map, timestamp_sec)
            self.log_images(frame_idx_map, timestamp_sec)
            
            for arm_idx, (arm_name, arm_info) in enumerate(self.arm_poses.items()):
                if frame_idx < len(arm_info['data']):
                    # rotate the arm poses around the y-axis by 150 degrees
                    positions = arm_info['data'][:frame_idx+1, :3] - trajectory_center
                    theta = np.radians(150)                    
                    c, s = np.cos(theta), np.sin(theta)
                    R_y = np.array([
                        [c,  0, s],
                        [0,  1, 0],
                        [-s, 0, c]
                    ])
                    positions = positions @ R_y.T
                    color = arm_colors[arm_idx % len(arm_colors)]
                    if len(positions) > 1:
                        visualize_trajectory_with_colors(f"/world/arms/{arm_name}/trajectory", positions, [color] * len(positions), radii=0.002)
                    rr.log(f"/world/arms/{arm_name}/current", rr.Points3D([positions[-1]], colors=[color], radii=[0.01]))

    def create_blueprint(self):
        from rerun.blueprint import (
            Blueprint, Horizontal, Vertical, Spatial3DView,
            TimeSeriesView, Tabs, SelectionPanel, TimePanel,
        )
        pose_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        
        arm_tabs = []
        for arm_name in sorted(self.arm_poses.keys()):
            arm_views = [TimeSeriesView(name=f"{arm_name}_{label}", origin=f"/arms/{arm_name}/{label}") 
                        for label in pose_labels]
            arm_tabs.append(Vertical(*arm_views, name=arm_name))
        
        camera_views = []
        for cam_name in sorted(self.camera_data.keys()):
            camera_views.append(rr.blueprint.Spatial2DView(name=cam_name, origin=f"/cameras/{cam_name}"))
        
        gripper_views = []
        for grip_name in sorted(self.gripper_data.keys()):
            gripper_views.append(TimeSeriesView(name=grip_name, origin=f"/grippers/{grip_name}"))
        
        all_tabs = []
        if arm_tabs:
            all_tabs.extend(arm_tabs)
        if gripper_views:
            all_tabs.append(Vertical(*gripper_views, name="grippers"))
        
        main_vertical_children = [
            Spatial3DView(name="3D Trajectory View", origin="/world", contents=["/world/**"])
        ]
        if camera_views:
            main_vertical_children.append(Horizontal(*camera_views))
        
        main_vertical = Vertical(*main_vertical_children, row_shares=[2, 1] if camera_views else [1])
        
        right_vertical = None
        if all_tabs:
            right_vertical = Vertical(Tabs(*all_tabs, active_tab=0))
        
        return Blueprint(
            Horizontal(
                main_vertical,
                right_vertical if right_vertical else Vertical(),
                column_shares=[3, 2],
            ),
            SelectionPanel(expanded=False),
            TimePanel(expanded=True),
        )

# ============================================================================
# GUI for Episode Selection
# ============================================================================

class EpisodeSelectorGUI:
    """GUI for selecting and visualizing NEDF episodes."""
    
    def __init__(self, root: tk.Tk, default_dir: Optional[Path] = None):
        self.root = root
        self.root.title("NEDF Episode Visualizer")
        self.root.geometry("800x600")
        
        self.data_dir = tk.StringVar()
        self.episode_dirs: List[Path] = []
        self.selected_episode = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self.is_visualizing = False
        self.default_dir = default_dir
        self.auto_visualize_enabled = True
        self.is_programmatic_selection = False
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        dir_frame = ttk.Frame(main_frame)
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_episodes).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(main_frame, text="Episode:").grid(row=2, column=0, sticky=tk.W, pady=5)
        episode_combo = ttk.Combobox(main_frame, textvariable=self.selected_episode, width=50, state="readonly")
        episode_combo.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        episode_combo.bind("<<ComboboxSelected>>", self.on_episode_selected)
        self.episode_combo = episode_combo
        
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        status_label = ttk.Label(status_frame, textvariable=self.status_text, foreground="blue")
        status_label.pack(side=tk.LEFT, padx=5)
        self.status_label = status_label
        
        # Device statistics display area
        ttk.Label(main_frame, text="Device Statistics:").grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create scrollable text widget for statistics
        stats_text = tk.Text(stats_frame, height=12, width=70, wrap=tk.WORD, font=("Courier", 9))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=stats_text.yview)
        stats_text.configure(yscrollcommand=stats_scrollbar.set)
        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text = stats_text
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        self.visualize_button = ttk.Button(button_frame, text="Visualize", command=self.start_visualization)
        self.visualize_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(button_frame, text="Next", command=self.next_episode_and_visualize)
        self.next_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        if self.default_dir is not None:
            default_path = Path(self.default_dir)
            if default_path.exists():
                self.data_dir.set(str(default_path))
                self.refresh_episodes()
            else:
                logger.warning(f"Default directory does not exist: {default_path}")
        else:
            script_dir = Path(__file__).parent.parent
            fallback_dir = script_dir / "data"
            if fallback_dir.exists():
                self.data_dir.set(str(fallback_dir))
                self.refresh_episodes()
    
    def browse_directory(self):
        directory = filedialog.askdirectory(
            title="Select Data Directory",
            initialdir=self.data_dir.get() if self.data_dir.get() else str(Path.home())
        )
        if directory:
            self.data_dir.set(directory)
            self.refresh_episodes()
    
    def refresh_episodes(self):
        data_dir_path = Path(self.data_dir.get())
        if not data_dir_path.exists():
            messagebox.showerror("Error", f"Directory does not exist: {data_dir_path}")
            return False
        
        episode_dirs = sorted(glob.glob(str(data_dir_path / "episode_*")))
        self.episode_dirs = [Path(d) for d in episode_dirs if Path(d).is_dir()]
        
        if not self.episode_dirs:
            messagebox.showwarning("No Episodes", f"No episode directories found in: {data_dir_path}")
            self.episode_combo['values'] = []
            return False
        
        episode_names = [ep.name for ep in self.episode_dirs]
        self.episode_combo['values'] = episode_names
        
        if episode_names:
            self.is_programmatic_selection = True
            self.selected_episode.set(episode_names[0])
            self.is_programmatic_selection = False
        
        logger.info(f"Found {len(self.episode_dirs)} episode(s) in {data_dir_path}")
        return True
    
    def on_episode_selected(self, event=None):
        if self.auto_visualize_enabled and not self.is_visualizing and not self.is_programmatic_selection:
            self.root.after(100, self.start_visualization)
    
    def next_episode_and_visualize(self):
        if self.is_visualizing:
            messagebox.showwarning("Warning", "Visualization is already running. Please wait for it to complete.")
            return
        
        current_selected = self.selected_episode.get()
        if not self.refresh_episodes():
            return
        
        episode_names = [ep.name for ep in self.episode_dirs]
        current_idx = -1
        if current_selected in episode_names:
            current_idx = episode_names.index(current_selected)
        
        next_idx = (current_idx + 1) % len(episode_names) if episode_names else -1
        
        if next_idx >= 0:
            old_auto = self.auto_visualize_enabled
            self.auto_visualize_enabled = False
            self.selected_episode.set(episode_names[next_idx])
            self.auto_visualize_enabled = old_auto
            self.start_visualization()
        else:
            messagebox.showwarning("Warning", "No episodes available to visualize.")
    
    def start_visualization(self):
        if self.is_visualizing:
            messagebox.showwarning("Warning", "Visualization is already running. Please wait for it to complete.")
            return
        
        if not self.selected_episode.get():
            messagebox.showerror("Error", "Please select an episode")
            return
        
        selected_name = self.selected_episode.get()
        episode_dir = None
        for ep_dir in self.episode_dirs:
            if ep_dir.name == selected_name:
                episode_dir = ep_dir
                break
        
        if not episode_dir or not episode_dir.exists():
            messagebox.showerror("Error", f"Episode directory not found: {selected_name}")
            return
        
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            messagebox.showerror("Error", f"metadata.json not found in: {episode_dir}")
            return
        
        self.is_visualizing = True
        self.visualize_button.config(state="disabled")
        self.next_button.config(state="disabled")
        self.status_text.set(f"Visualizing: {selected_name}...")
        self.status_label.config(foreground="green")
        
        thread = threading.Thread(
            target=self._run_visualization,
            args=(episode_dir,),
            daemon=True
        )
        thread.start()
    
    def _run_visualization(self, episode_dir: Path):
        """Run visualization in a separate thread with Memory Cleanup and Unique Session IDs."""
        devices = None
        vis = None
        
        try:
            # Load Data
            logger.info(f"Loading episode from: {episode_dir}")
            devices = load_nedf_episode(episode_dir)
            
            if not devices:
                logger.error("No valid devices loaded.")
                self.root.after(0, self._visualization_finished, "Error: No valid devices loaded")
                return
            
            # Update device statistics in GUI
            self.root.after(0, self._update_device_statistics, devices)
            
            # --- CRITICAL FIX for #11585 ---
            # 1. Use the episode name as the application_id (friendly name)
            # 2. Generate a random UUID for recording_id (technical uniqueness)
            # This ensures Rerun treats this as a brand new recording and doesn't try to merge indices.
            app_id = f"NEDF_{episode_dir.name}"
            rec_id = str(uuid.uuid4())
            logger.info(f"Init Rerun - App: {app_id}, RecID: {rec_id}")
            
            # [Fix] Use official memory limit API
            # Initialize without spawning first
            rr.init(app_id, recording_id=rec_id, spawn=False)
            # Spawn with explicit memory limit
            rr.spawn(memory_limit="4GB")
            
            # Visualize
            vis = NEDFEpisodeVisualizer(devices, episode_dir.name)
            rr.send_blueprint(vis.create_blueprint())
            vis.visualize_episode(fps=30.0)
            
            logger.info("Visualization completed.")
            self.root.after(0, self._visualization_finished, f"Completed: {episode_dir.name}")
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._visualization_finished, f"Error: {str(e)}")
            
        finally:
            # --- CRITICAL FIX for Memory Leak ---
            # Explicitly break references and force garbage collection
            logger.info("Cleaning up memory for next episode...")
            if vis:
                del vis
            if devices:
                devices.clear()
                del devices
            
            # Force Python Garbage Collection
            gc.collect()

    def _format_device_statistics(self, devices: Dict[str, Dict[str, Any]]) -> str:
        """
        Format device statistics including timestamp counts, duration, and frame rates.
        
        Args:
            devices: Dictionary of device data loaded from episode
            
        Returns:
            Formatted string with statistics
        """
        if not devices:
            return "No devices loaded."
        
        lines = []
        lines.append("=" * 80)
        lines.append("Device Statistics")
        lines.append("=" * 80)
        lines.append("")
        
        # Group devices by type
        cameras = []
        poses = []
        timeseries = []
        unknown = []
        
        for device_key, device_data in devices.items():
            dev_type = device_data.get('type', 'unknown')
            if dev_type == 'camera':
                cameras.append((device_key, device_data))
            elif dev_type == 'vive_pose':
                poses.append((device_key, device_data))
            elif dev_type == 'timeseries':
                timeseries.append((device_key, device_data))
            else:
                unknown.append((device_key, device_data))
        
        # Calculate statistics for each device
        def calc_stats(device_data: Dict[str, Any]) -> Tuple[int, float, float]:
            """Calculate count, duration, and frame rate."""
            timestamps = device_data.get('timestamps', [])
            if len(timestamps) == 0:
                return 0, 0.0, 0.0
            
            count = len(timestamps)
            duration = float(timestamps[-1] - timestamps[0]) if count > 1 else 0.0
            fps = count / duration if duration > 0 else 0.0
            
            return count, duration, fps
        
        # Print cameras
        if cameras:
            lines.append("Cameras:")
            lines.append("-" * 80)
            for device_key, device_data in sorted(cameras, key=lambda x: x[0]):
                count, duration, fps = calc_stats(device_data)
                lines.append(f"  {device_key:40s} | Count: {count:6d} | Duration: {duration:8.2f}s | FPS: {fps:7.2f}")
            lines.append("")
        
        # Print poses
        if poses:
            lines.append("Poses (Vive Tracker / Arm):")
            lines.append("-" * 80)
            for device_key, device_data in sorted(poses, key=lambda x: x[0]):
                count, duration, fps = calc_stats(device_data)
                lines.append(f"  {device_key:40s} | Count: {count:6d} | Duration: {duration:8.2f}s | FPS: {fps:7.2f}")
            lines.append("")
        
        # Print timeseries (grippers, encoders, etc.)
        if timeseries:
            lines.append("Time Series (Grippers / Encoders):")
            lines.append("-" * 80)
            for device_key, device_data in sorted(timeseries, key=lambda x: x[0]):
                count, duration, fps = calc_stats(device_data)
                lines.append(f"  {device_key:40s} | Count: {count:6d} | Duration: {duration:8.2f}s | FPS: {fps:7.2f}")
            lines.append("")
        
        # Print unknown
        if unknown:
            lines.append("Unknown Devices:")
            lines.append("-" * 80)
            for device_key, device_data in sorted(unknown, key=lambda x: x[0]):
                count, duration, fps = calc_stats(device_data)
                lines.append(f"  {device_key:40s} | Count: {count:6d} | Duration: {duration:8.2f}s | FPS: {fps:7.2f}")
            lines.append("")
        
        # Summary
        total_devices = len(devices)
        lines.append("=" * 80)
        lines.append(f"Total Devices: {total_devices}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _update_device_statistics(self, devices: Dict[str, Dict[str, Any]]):
        """Update the statistics text widget with device information."""
        stats_text = self._format_device_statistics(devices)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.see(1.0)  # Scroll to top
    
    def _visualization_finished(self, message: str):
        self.is_visualizing = False
        self.visualize_button.config(state="normal")
        self.next_button.config(state="normal")
        self.status_text.set(message)
        self.status_label.config(foreground="blue")


# ============================================================================
# Visualization Function (CLI Mode)
# ============================================================================

def visualize_episode(episode_dir: Path):
    """Visualize a single episode."""
    if not episode_dir.exists():
        logger.error("Episode directory not found")
        return

    # Load Data
    logger.info(f"Loading episode from: {episode_dir}")
    devices = load_nedf_episode(episode_dir)
    
    if not devices:
        logger.error("No valid devices loaded. Exiting.")
        return

    # Init Rerun with unique ID
    app_id = f"NEDF_{episode_dir.name}"
    rec_id = str(uuid.uuid4())
    
    # [Fix] Use official memory limit API
    # Initialize without spawning first
    rr.init(app_id, recording_id=rec_id, spawn=False)
    # Spawn with explicit memory limit
    rr.spawn(memory_limit="4GB")
    
    # Visualize
    vis = NEDFEpisodeVisualizer(devices, episode_dir.name)
    rr.send_blueprint(vis.create_blueprint())
    vis.visualize_episode(fps=30.0)
    
    logger.info("Done.")


# ============================================================================
# Main
# ============================================================================

def main():
    # [Removed] os.environ["RERUN_MEMORY_LIMIT"] = "4GB" replaced by rr.spawn(memory_limit="4GB")
    
    parser = argparse.ArgumentParser(description="Visualize NEDF Episode with Rerun")
    parser.add_argument('--episode_dir', type=Path, default=None, help="Path to episode directory (containing metadata.json)")
    parser.add_argument('--gui', action='store_true', help="Launch GUI for episode selection")
    parser.add_argument('--default_dir', type=Path, default=None, help="Default data directory for GUI (containing episode folders)")
    args = parser.parse_args()
    
    # Launch GUI if requested or if no episode_dir provided
    if args.gui or args.episode_dir is None:
        root = tk.Tk()
        app = EpisodeSelectorGUI(root, default_dir=args.default_dir)
        root.mainloop()
    else:
        # Command line mode
        visualize_episode(episode_dir=args.episode_dir)

if __name__ == "__main__":
    main()