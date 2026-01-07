#!/usr/bin/env python3
"""
Async Episode Visualizer for NEDF2 Format - Visualize episodes saved in NEDF2 format.

- Supports NEDF2 format (MCAP-based) with metadata.json and data.mcap files
- Reads data using NEDF SDK
- Aligns different device streams by a common time slider using nearest-sample lookup
- Visualizes:
  - Up to two camera streams as images (JPG compressed)
  - Vive tracker trajectories (3D + time series)
  - Rotary encoder time series

Author: Dong Jiulong
"""
import sys
import argparse
import os
import glob
from typing import Dict, Any, Optional, Tuple, List

import base64
import io

import dash
from dash import dcc, html, Input, Output
import dash.dependencies
import plotly.graph_objects as go
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# NEDF SDK imports
from nmx_nedf_api import NEDFFactory, NEDFReaderConfig, NEDFType
from nmx_msg.Lowdim_pb2 import LowdimData
from nmx_msg.Image_pb2 import Image as ProtoImage

# Ensure project root is importable when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger


class AsyncEpisode:
    def __init__(self, path: str):
        self.path = path
        self.episode_dirs: List[str] = []
        self.current_episode_idx: int = 0
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.global_time_range: Tuple[float, float] = (0.0, 0.0)
        self.nedf_reader = None
        self.load_index()
        self.load_episode(0)

    def load_index(self) -> None:
        """
        Load index of NEDF2 episode directories.
        Each episode is a directory containing metadata.json and data.mcap files.
        """
        if os.path.isdir(self.path):
            # Find all episode_XXXX directories
            episode_dirs = sorted(glob.glob(os.path.join(self.path, 'episode_*')))
            # Filter to only keep directories (not files)
            self.episode_dirs = [d for d in episode_dirs if os.path.isdir(d)]
            
            if not self.episode_dirs:
                raise ValueError(f"No episode directories found in: {self.path}")
        else:
            raise ValueError(f"Path must be a valid directory containing episode folders: {self.path}")
        
        logger.info(f"Found {len(self.episode_dirs)} episode directory(ies)")

    def close_reader(self):
        """Close NEDF reader if open."""
        if self.nedf_reader:
            self.nedf_reader = None

    def load_episode(self, episode_idx: int) -> None:
        """
        Load NEDF2 episode data using NEDF SDK.
        
        Args:
            episode_idx: Index of episode to load
        """
        if episode_idx < 0 or episode_idx >= len(self.episode_dirs):
            logger.warning(f"Episode index {episode_idx} out of range")
            return
        
        self.close_reader()
        
        episode_dir = self.episode_dirs[episode_idx]
        logger.info(f"Loading NEDF2 episode {episode_idx}: {episode_dir}")
        self.current_episode_idx = episode_idx
        self.devices = {}
        
        # Get metadata file path
        metadata_path = os.path.join(episode_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.error(f"metadata.json not found in {episode_dir}")
            return
        
        # Create NEDF reader
        try:
            reader_config = NEDFReaderConfig(metadata_file_path=metadata_path)
            self.nedf_reader = NEDFFactory.get_reader(reader_config)
            
            # Get all available topics and their timestamps
            all_topics_timestamps = self.nedf_reader.get_raw_topics_timestamps()
            
            logger.info(f"Found {len(all_topics_timestamps)} topics in NEDF2 file")
            
            # Collect all timestamps first to find minimum for relative time conversion
            all_timestamps_ns = []
            
            # Get camera timestamps
            camera_info = self.nedf_reader.metadata.camera_info
            for camera_id in camera_info.keys():
                camera_timestamps_ns = self.nedf_reader.get_raw_image_timestamps(camera_id)
                if len(camera_timestamps_ns) > 0:
                    all_timestamps_ns.extend(camera_timestamps_ns)
            
            # Get lowdim timestamps
            all_topics_timestamps = self.nedf_reader.get_raw_topics_timestamps()
            for timestamps_ns in all_topics_timestamps.values():
                if len(timestamps_ns) > 0:
                    all_timestamps_ns.extend(timestamps_ns)
            
            # Calculate minimum timestamp for relative time conversion
            min_timestamp_ns = min(all_timestamps_ns) if all_timestamps_ns else 0
            logger.info(f"Minimum timestamp across all devices: {min_timestamp_ns} ns ({min_timestamp_ns / 1e9:.3f}s)")
            
            # Process camera devices
            for camera_id, camera_name in camera_info.items():
                # Load all images with timestamps
                img_list = []
                img_timestamps_list = []
                
                for msg in self.nedf_reader.iter_color(camera_id):
                    # Get timestamp in nanoseconds
                    log_ts_ns = msg.message.log_time
                    jpg_image = msg.decoded_message
                    
                    # Decode JPG image
                    jpg_bytes = np.frombuffer(jpg_image.data, dtype=np.uint8)
                    img = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_list.append(img)
                        img_timestamps_list.append(log_ts_ns)
                
                if img_list:
                    # Convert timestamps to relative time in seconds
                    camera_timestamps_ns = np.array(img_timestamps_list)
                    camera_timestamps_s = (camera_timestamps_ns - min_timestamp_ns) / 1e9
                    
                    device = {
                        'name': camera_id,
                        'type': 'camera',
                        'timestamps': camera_timestamps_s,
                        'data': np.array(img_list),
                        'data_preview_shape': (len(img_list),) + img_list[0].shape,
                        'dtype': 'uint8'
                    }
                    self.devices[camera_id] = device
                    logger.info(f"Loaded camera {camera_id}: {len(img_list)} frames, "
                               f"time range=[{camera_timestamps_s[0]:.3f}s, {camera_timestamps_s[-1]:.3f}s]")
            
            # Process custom lowdim topics (ViveTracker, RotaryEncoder, etc.)
            for topic, timestamps_ns in all_topics_timestamps.items():
                if '/lowdim/' not in topic:
                    continue
                
                # Extract device key from topic (lowercase in NEDF2)
                device_key = topic.replace('/lowdim/', '')
                
                # Skip if already processed
                if device_key in self.devices:
                    continue
                
                # Read lowdim data using get_lowdim (as per async_data_saver_NEDF2.py)
                data_list = []
                timestamps_list = []
                
                logger.debug(f"Reading lowdim topic: {topic}")
                
                try:
                    sample_count = 0
                    for msg in self.nedf_reader.get_lowdim(topic):
                        # msg.message.log_time is in nanoseconds
                        log_ts_ns = msg.message.log_time
                        lowdim_data = msg.decoded_message
                        
                        # Extract data array
                        data_array = np.array(lowdim_data.data, dtype=np.float32)
                        data_list.append(data_array)
                        timestamps_list.append(log_ts_ns)
                        
                        # Debug: print first few samples
                        if sample_count < 3:
                            logger.debug(f"  Sample {sample_count}: timestamp={log_ts_ns} ns, data_len={len(data_array)}, "
                                       f"data_preview={data_array[:min(5, len(data_array))]}")
                        sample_count += 1
                        
                except Exception as e:
                    logger.error(f"Error reading lowdim topic {topic}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                if not data_list:
                    logger.warning(f"No data found for topic {topic}")
                    continue
                
                logger.info(f"Read {len(data_list)} samples from topic {topic}")
                
                # Convert timestamps to relative time in seconds
                timestamps_ns_array = np.array(timestamps_list)
                timestamps_s = (timestamps_ns_array - min_timestamp_ns) / 1e9
                
                # Stack data into array
                try:
                    data_array = np.stack(data_list)
                    logger.info(f"Stacked data shape: {data_array.shape}")
                except Exception as e:
                    logger.error(f"Error stacking data for topic {topic}: {e}")
                    logger.error(f"First data sample shape: {data_list[0].shape if data_list else 'N/A'}")
                    continue
                
                logger.info(f"Processing lowdim topic {topic}: device_key={device_key}, shape={data_array.shape}, "
                           f"time range=[{timestamps_s[0]:.3f}s, {timestamps_s[-1]:.3f}s], "
                           f"num_samples={len(data_list)}")
                
                # Classify device type based on device name and data shape
                # ViveTrackerDevice: 14D data [left_pose(7), right_pose(7)]
                # Check if this is ViveTracker data (14 dimensions)
                is_vive_tracker = False
                if 'vivetrackerdevice' in device_key.lower():
                    if len(data_array.shape) == 2 and data_array.shape[1] == 14:
                        is_vive_tracker = True
                    elif len(data_array.shape) == 2 and data_array.shape[1] == 1 and len(data_array.flatten()) >= 14:
                        # Data might be stored as (N, 1) with 14 elements - need to check first element
                        logger.warning(f"Unexpected shape for ViveTracker {device_key}: {data_array.shape}")
                
                if is_vive_tracker:
                    logger.info(f"Found ViveTrackerDevice format for {device_key} (shape: {data_array.shape}). Splitting into two virtual devices.")
                    
                    # Ensure data is in correct shape (N, 14)
                    if data_array.shape[1] != 14:
                        logger.error(f"ViveTracker data has unexpected shape: {data_array.shape}, expected (N, 14)")
                        continue
                    
                    # Split into left and right
                    left_data = data_array[:, :7]  # [x, y, z, qx, qy, qz, qw]
                    right_data = data_array[:, 7:14]  # [x, y, z, qx, qy, qz, qw]
                    
                    logger.info(f"Split ViveTracker data: left={left_data.shape}, right={right_data.shape}")
                    
                    # Left tracker
                    left_device = {
                        'name': f"{device_key}_left",
                        'type': 'vive_pose',
                        'timestamps': timestamps_s,
                        'data': left_data,
                        'data_preview_shape': left_data.shape,
                        'dtype': 'float32'
                    }
                    self.devices[f"{device_key}_left"] = left_device
                    
                    # Right tracker
                    right_device = {
                        'name': f"{device_key}_right",
                        'type': 'vive_pose',
                        'timestamps': timestamps_s,
                        'data': right_data,
                        'data_preview_shape': right_data.shape,
                        'dtype': 'float32'
                    }
                    self.devices[f"{device_key}_right"] = right_device
                    
                    logger.info(f"Loaded ViveTrackerDevice {device_key}: {len(data_list)} frames (split into left/right)")
                    
                # RotaryEncoderDevice: 1D gripper data
                elif 'rotaryencoderdevice' in device_key.lower():
                    # Data might be 1D or 2D with shape (T, 1)
                    if len(data_array.shape) == 2 and data_array.shape[1] == 1:
                        # Already in correct shape
                        pass
                    elif len(data_array.shape) == 1:
                        # Reshape to (T, 1)
                        data_array = data_array.reshape(-1, 1)
                    
                    device = {
                        'name': device_key,
                        'type': 'timeseries',
                        'timestamps': timestamps_s,
                        'data': data_array,
                        'data_preview_shape': data_array.shape,
                        'dtype': 'float32'
                    }
                    self.devices[device_key] = device
                    logger.info(f"Loaded RotaryEncoderDevice {device_key}: {len(data_list)} frames")
                    
                # Generic ViveTracker pose: [T, 7]
                elif len(data_array.shape) == 2 and data_array.shape[1] == 7:
                    device = {
                        'name': device_key,
                        'type': 'vive_pose',
                        'timestamps': timestamps_s,
                        'data': data_array,
                        'data_preview_shape': data_array.shape,
                        'dtype': 'float32'
                    }
                    self.devices[device_key] = device
                    logger.info(f"Loaded pose device {device_key}: {len(data_list)} frames")
                    
                # Generic timeseries: [T, 1] or [T,]
                elif (len(data_array.shape) == 2 and data_array.shape[1] == 1) or len(data_array.shape) == 1:
                    # Ensure 2D shape (T, 1)
                    if len(data_array.shape) == 1:
                        data_array = data_array.reshape(-1, 1)
                    
                    device = {
                        'name': device_key,
                        'type': 'timeseries',
                        'timestamps': timestamps_s,
                        'data': data_array,
                        'data_preview_shape': data_array.shape,
                        'dtype': 'float32'
                    }
                    self.devices[device_key] = device
                    logger.info(f"Loaded timeseries device {device_key}: {len(data_list)} frames")
                    
                else:
                    # Unknown format - still load it
                    device = {
                        'name': device_key,
                        'type': 'unknown',
                        'timestamps': timestamps_s,
                        'data': data_array,
                        'data_preview_shape': data_array.shape,
                        'dtype': 'float32'
                    }
                    self.devices[device_key] = device
                    logger.warning(f"Loaded unknown device {device_key}: shape {data_array.shape}")
            
            # Compute global time range
            if self.devices:
                tmins = [np.min(dev['timestamps']) for dev in self.devices.values() if len(dev['timestamps']) > 0]
                tmaxs = [np.max(dev['timestamps']) for dev in self.devices.values() if len(dev['timestamps']) > 0]
                if tmins and tmaxs:
                    self.global_time_range = (float(min(tmins)), float(max(tmaxs)))
                else:
                    self.global_time_range = (0.0, 0.0)
            else:
                self.global_time_range = (0.0, 0.0)
            
            logger.info(f"Global time range: {self.global_time_range[0]:.3f}s to {self.global_time_range[1]:.3f}s")
            
            # Summary of loaded devices
            logger.info(f"\n{'='*60}")
            logger.info("Loaded Devices Summary:")
            logger.info(f"{'='*60}")
            for device_key, device in self.devices.items():
                logger.info(f"  {device_key}:")
                logger.info(f"    Type: {device['type']}")
                logger.info(f"    Shape: {device['data_preview_shape']}")
                logger.info(f"    Samples: {len(device['timestamps'])}")
                logger.info(f"    Time range: [{device['timestamps'][0]:.3f}s, {device['timestamps'][-1]:.3f}s]")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error loading NEDF2 episode: {e}")
            import traceback
            traceback.print_exc()

    def get_time_aligned_indices(self, t: float) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for key, dev in self.devices.items():
            ts = dev['timestamps']
            if ts is None or len(ts) == 0:
                indices[key] = -1
                continue
            # nearest index
            idx = int(np.clip(np.searchsorted(ts, t, side='left'), 0, len(ts) - 1))
            # choose closer between idx and idx-1
            if idx > 0 and abs(ts[idx] - t) > abs(ts[idx - 1] - t):
                idx = idx - 1
            indices[key] = idx
        return indices

    @staticmethod
    def image_to_base64(img_array: np.ndarray) -> str:
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{encoded}"


class AsyncEpisodeVisualizer:
    def __init__(self, path: str):
        self.episode = AsyncEpisode(path)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self) -> None:
        t0, t1 = self.episode.global_time_range
        camera_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'camera'][:2]
        vive_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'vive_pose']
        timeseries_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'timeseries']

        self.app.layout = html.Div([
            html.H1("Async Episode Visualizer", style={'textAlign': 'center', 'margin-top': '100px'}),

            # control panel
            html.Div([
                html.Div([
                    html.Label("Episode:", style={'margin-right': '10px', 'color': 'white'}),
                    dcc.Dropdown(
                        id='episode-dropdown',
                        options=[{'label': f"Episode {i}: {os.path.basename(p)}", 'value': i} for i, p in enumerate(self.episode.episode_dirs)],
                        value=self.episode.current_episode_idx,
                        style={'width': '360px'}
                    )
                ], style={'display': 'inline-block', 'margin-right': '30px'}),
                html.Div([
                    html.Button('Previous', id='prev-button', n_clicks=0, style={'margin-right': '10px', 'padding': '10px 20px'}),
                    html.Button('Next', id='next-button', n_clicks=0, style={'padding': '10px 20px'}),
                ], style={'display': 'inline-block'}),
            ], style={'position': 'fixed','top': '0','left': '0','right': '0','background-color': 'rgba(52,58,64,0.95)','padding': '15px 20px','z-index': '1000','box-shadow': '0 2px 4px rgba(0,0,0,0.1)','backdrop-filter': 'blur(5px)'}),

            # time slider
            html.Div([
                html.Label("Time (s):", style={'color': 'white', 'margin-bottom': '10px', 'display': 'block'}),
                dcc.Slider(id='time-slider', min=t0, max=max(t1, t0 + 1e-3), step=0.001, value=t0,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'margin': '100px 20px 20px 20px'}),

            # images
            html.Div([
                html.H3("Camera Images"),
                html.Div([
                    html.Div([
                        html.H4(camera_keys[i] if i < len(camera_keys) else ''),
                        html.Img(id=f"image-{i}", style={'width': '100%', 'max-width': '640px'})
                    ], style={'display': 'inline-block', 'width': '48%', 'margin': '1%'})
                    for i in range(min(2, len(camera_keys)))
                ])
            ]),

            # vive plots
            html.Div([
                html.H3("Vive Tracker 3D Trajectories"),
                dcc.Graph(id='vive-3d-plot', style={'height': '600px'})
            ]),
            html.Div([
                html.H3("Vive Tracker Position & Rotation"),
                html.Div([
                    html.Div([dcc.Graph(id='vive-pos-plot')], style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='vive-rot-plot')], style={'width': '48%', 'display': 'inline-block', 'margin-left': '2%'})
                ])
            ]),

            # encoder
            html.Div([
                html.H3("Timeseries (e.g., Encoders)"),
                dcc.Graph(id='timeseries-plot')
            ]),

            html.Div([html.Div(id='data-info')], style={'margin': '20px'})
        ])

        # store keys for callbacks
        self.camera_keys = camera_keys
        self.vive_keys = vive_keys
        self.timeseries_keys = timeseries_keys
        
        # Debug: log the keys
        logger.info(f"Setup layout - camera_keys: {camera_keys}")
        logger.info(f"Setup layout - vive_keys: {vive_keys}")
        logger.info(f"Setup layout - timeseries_keys: {timeseries_keys}")

    def setup_callbacks(self) -> None:
        # episode navigation updates time range and keys
        @self.app.callback(
            [Output('time-slider', 'min'), Output('time-slider', 'max'), Output('time-slider', 'value'), Output('episode-dropdown', 'value')],
            [Input('prev-button', 'n_clicks'), Input('next-button', 'n_clicks'), Input('episode-dropdown', 'value')],
            [dash.dependencies.State('time-slider', 'value')],
            prevent_initial_call=False
        )
        def update_episode(prev_clicks, next_clicks, episode_value, current_time):
            ctx = dash.callback_context
            new_idx = self.episode.current_episode_idx
            if ctx.triggered:
                bid = ctx.triggered[0]['prop_id'].split('.')[0]
                if bid == 'prev-button' and self.episode.current_episode_idx > 0:
                    new_idx = self.episode.current_episode_idx - 1
                elif bid == 'next-button' and self.episode.current_episode_idx < len(self.episode.episode_dirs) - 1:
                    new_idx = self.episode.current_episode_idx + 1
                elif bid == 'episode-dropdown' and episode_value is not None:
                    new_idx = int(episode_value)
            if new_idx != self.episode.current_episode_idx:
                self.episode.load_episode(new_idx)
                # refresh device keys
                self.camera_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'camera'][:2]
                self.vive_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'vive_pose']
                self.timeseries_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'timeseries']
                logger.info(f"Episode changed - camera_keys: {self.camera_keys}")
                logger.info(f"Episode changed - vive_keys: {self.vive_keys}")
                logger.info(f"Episode changed - timeseries_keys: {self.timeseries_keys}")
            t0, t1 = self.episode.global_time_range
            return t0, max(t1, t0 + 1e-3), t0, self.episode.current_episode_idx

        # camera updates
        for i, cam_key in enumerate(self.camera_keys):
            @self.app.callback(
                Output(f'image-{i}', 'src'),
                [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
                prevent_initial_call=False
            )
            def update_image(t, episode_idx, cam_idx=i, key=cam_key):
                if key not in self.episode.devices:
                    return ''
                dev = self.episode.devices[key]
                idx_map = self.episode.get_time_aligned_indices(t)
                idx = idx_map.get(key, -1)
                if idx < 0 or dev['data'] is None or idx >= dev['data'].shape[0]:
                    return ''
                img = dev['data'][idx]
                return AsyncEpisode.image_to_base64(img)

        # vive plots
        @self.app.callback(
            [Output('vive-3d-plot', 'figure'), Output('vive-pos-plot', 'figure'), Output('vive-rot-plot', 'figure')],
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_vive_plots(t, episode_idx):
            logger.debug(f"update_vive_plots called: t={t:.3f}s, vive_keys={self.vive_keys}")
            colors = ['blue', 'red', 'green', 'orange']
            # 3D
            fig3d = go.Figure()
            pos_fig = go.Figure()
            rot_fig = go.Figure()
            for i, key in enumerate(self.vive_keys):
                logger.debug(f"Processing vive key: {key}, exists={key in self.episode.devices}")
                if key not in self.episode.devices:
                    logger.warning(f"Vive key {key} not found in devices")
                    continue
                dev = self.episode.devices[key]
                logger.debug(f"Device {key}: type={dev['type']}, shape={dev['data_preview_shape']}")
                
                # Get data from device
                data = dev['data']
                if data is None or len(data.shape) != 2 or data.shape[1] != 7:
                    continue
                
                ts = dev['timestamps']
                # draw trajectory up to t
                mask = ts <= t
                if not np.any(mask):
                    continue

                pos = data[mask, :3]
                quat = data[mask, 3:]

                color = colors[i % len(colors)]
                # 3D trajectory
                fig3d.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='lines+markers', name=key,
                                             line=dict(color=color, width=4), marker=dict(size=3)))
                # current point
                idx_map = self.episode.get_time_aligned_indices(t)
                idx = idx_map.get(key, -1)
                if 0 <= idx < data.shape[0]:
                    fig3d.add_trace(go.Scatter3d(x=[data[idx, 0]], y=[data[idx, 1]], z=[data[idx, 2]], mode='markers',
                                                 name=f'{key} current', marker=dict(size=8, color=color, symbol='diamond')))
                # position series
                frames = ts[mask]
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 0], name=f'{key} X', line=dict(color=color)))
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 1], name=f'{key} Y', line=dict(color=color, dash='dash')))
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 2], name=f'{key} Z', line=dict(color=color, dash='dot')))
                # quaternion series
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 0], name=f'{key} qx', line=dict(color=color)))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 1], name=f'{key} qy', line=dict(color=color, dash='dash')))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 2], name=f'{key} qz', line=dict(color=color, dash='dot')))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 3], name=f'{key} qw', line=dict(color=color, dash='dashdot')))
            fig3d.update_layout(title=f"Vive 3D Trajectories (t={t:.3f}s)", scene=dict(aspectmode='cube'))
            pos_fig.update_layout(title="Vive Positions", xaxis_title="Time (s)", yaxis_title="Position (m)")
            rot_fig.update_layout(title="Vive Quaternions", xaxis_title="Time (s)", yaxis_title="Quaternion")
            return fig3d, pos_fig, rot_fig

        # timeseries plot
        @self.app.callback(
            Output('timeseries-plot', 'figure'),
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_timeseries(t, episode_idx):
            logger.debug(f"update_timeseries called: t={t:.3f}s, timeseries_keys={self.timeseries_keys}")
            fig = go.Figure()
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, key in enumerate(self.timeseries_keys):
                logger.debug(f"Processing timeseries key: {key}, exists={key in self.episode.devices}")
                if key not in self.episode.devices:
                    logger.warning(f"Timeseries key {key} not found in devices")
                    continue
                dev = self.episode.devices[key]
                logger.debug(f"Device {key}: type={dev['type']}, shape={dev['data_preview_shape']}")
                data = dev['data']
                ts = dev['timestamps']
                if data is None:
                    continue
                # squeeze to 1D if needed
                arr = data
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                elif arr.ndim > 1:
                    continue  # skip unsupported shapes here
                mask = ts <= t
                if np.any(mask):
                    fig.add_trace(go.Scatter(x=ts[mask], y=arr[mask], name=key, line=dict(color=colors[i % len(colors)])))
            fig.update_layout(title=f"Timeseries (t={t:.3f}s)", xaxis_title="Time (s)", yaxis_title="Value")
            return fig

        # info
        @self.app.callback(
            Output('data-info', 'children'),
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_info(t, episode_idx):
            indices = self.episode.get_time_aligned_indices(t)
            info_lines = [html.P([html.Strong(f"t={t:.3f}s")])]
            for key, idx in indices.items():
                info_lines.append(html.P(f"{key}: idx={idx}"))
            return info_lines

    def run(self, host: str = '127.0.0.1', port: int = 8051, debug: bool = True) -> None:
        logger.info(f"Starting async visualizer at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Async episode visualizer for NEDF2 format")
    parser.add_argument("path", nargs='?', default="/home/noematrix/Desktop/jiulong/nedf_data", 
                       help="Path to directory containing NEDF2 episode folders (default: /home/noematrix/Desktop/jiulong/nedf_data)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8051, help="Port to run server on")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    args = parser.parse_args()

    try:
        vis = AsyncEpisodeVisualizer(args.path)
        vis.run(host=args.host, port=args.port, debug=not args.no_debug)
    except Exception as e:
        logger.error(f"Failed to start async visualizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
