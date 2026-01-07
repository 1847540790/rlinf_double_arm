#!/usr/bin/env python3
"""
Camera Visualizer - Real-time visualization for OpenCV camera streams.

This visualizer provides real-time camera stream display with floating
status information overlay.

Author: Jun Lv
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
import sys
import os
from typing import List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .base_visualizer import BaseVisualizer
except ImportError:
    from visualizers.base_visualizer import BaseVisualizer

from utils.logger_config import logger


class CameraVisualizer(BaseVisualizer):
    """
    Camera visualizer to display real-time camera frames from OpenCVCameraDevice shared memory.
    Supports both grayscale and color camera streams with status overlay.
    """
    
    def __init__(self, shared_memory_name: str = "opencv_camera_0_data", data_dtype: Any = np.uint8, smoothing_window: int = 10) -> None:
        """
        Initialize the camera visualizer.
        
        Args:
            shared_memory_name: Name of the shared memory to read from
            data_dtype: Expected data type of the numpy array
            smoothing_window: Number of samples for FPS and latency smoothing
        """
        # Initialize base visualizer
        super().__init__(shared_memory_name, data_dtype)
        
        # Override matplotlib setup for camera visualization
        plt.close(self.fig)  # Close the default status monitoring figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_title("Camera Stream")
        self.im = None
        self.info_text = None
        self.current_shape: Optional[Tuple[int, int]] = None

    def _ensure_figure_matches_resolution(self, frame_shape: Tuple[int, int]) -> None:
        """Resize figure window to match incoming frame resolution to avoid distortion."""
        if self.current_shape == frame_shape:
            return

        self.current_shape = frame_shape
        frame_height, frame_width = frame_shape
        dpi = self.fig.get_dpi() or plt.rcParams.get("figure.dpi", 100)

        min_size_inches = 2.0
        fig_width = max(frame_width / dpi, min_size_inches)
        fig_height = max(frame_height / dpi, min_size_inches)

        self.fig.set_size_inches(fig_width, fig_height, forward=True)
        self.ax.set_aspect('equal', adjustable='box')
    
    def update_plot(self, frame: Any) -> List:
        """Update the matplotlib plot with new data."""
        timestamp_ns, data_array = self.read_latest_frame()
        
        if timestamp_ns is None or data_array is None:
            return []
        
        # Calculate FPS
        self.calculate_fps(timestamp_ns)
        
        # Convert timestamp to readable format
        timestamp_s = timestamp_ns / 1e9
        dt = datetime.fromtimestamp(timestamp_s)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # millisecond precision
        
        # Calculate latency (difference between current time and frame timestamp)
        latency_ms = self.calculate_latency(timestamp_ns)
        
        # Handle color vs grayscale images
        if len(data_array.shape) == 3 and data_array.shape[2] == 3:
            # Color image
            cmap = None
            display_data = data_array
            shape_str = f"{data_array.shape[1]}x{data_array.shape[0]}x{data_array.shape[2]}"
        else:
            # Grayscale image
            cmap = 'gray'
            display_data = data_array
            shape_str = f"{data_array.shape[1]}x{data_array.shape[0]}"
        
        self._ensure_figure_matches_resolution(data_array.shape[:2])

        if self.im is None:
            # First time: create image plot
            self.ax.clear()
            self.im = self.ax.imshow(display_data, cmap=cmap, aspect='equal')
            self.ax.set_title("Camera Stream")
            self.ax.axis('off')  # Hide axes for cleaner image view
        else:
            # Update existing image
            self.im.set_array(display_data)
            if cmap == 'gray':  # Only set clim for grayscale
                self.im.set_clim(vmin=np.min(display_data), vmax=np.max(display_data))
        
        # Remove old info text if exists
        if self.info_text is not None:
            self.info_text.remove()
        
        # Create floating info box text
        info_text = f"Size: {shape_str}\nFPS: {self.fps_estimate:.1f}\nTime: {time_str}\nLatency: {latency_ms:.1f}ms"
        
        # Add floating info box in top-left corner
        self.info_text = self.ax.text(0.02, 0.98, info_text, 
                                     transform=self.ax.transAxes,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     fontsize=10,
                                     fontfamily='monospace',
                                     bbox=dict(boxstyle="round,pad=0.3", 
                                              facecolor='black', 
                                              alpha=0.7,
                                              edgecolor='white'),
                                     color='white')
        
        return []
    
    def start_visualization(self, update_interval: int = 50) -> None:
        """
        Start real-time visualization.
        
        Args:
            update_interval: Update interval in milliseconds
        """
        if not self.connect():
            return
        
        logger.info("Starting camera visualization. Close the plot window to stop.")
        logger.info(f"Monitoring: {self.shared_memory_name}")
        logger.info("-" * 50)
        
        # Set up the animation
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=update_interval, 
                                    blit=False, cache_frame_data=False)
        
        # Show the plot
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.show()
        
        # Clean up
        self.shared_memory.close()


def main() -> None:
    """Main function to run the camera visualizer."""
    parser = argparse.ArgumentParser(description="Camera visualization client")
    parser.add_argument("--shared-memory", default="opencv_camera_0_data", 
                        help="Shared memory name (default: opencv_camera_0_data)")
    parser.add_argument("--dtype", "-d", default="uint8", 
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 
                                'float32', 'float64'],
                        help="Data type (default: uint8)")
    parser.add_argument("--interval", "-i", type=int, default=50,
                        help="Update interval in ms (default: 50)")
    parser.add_argument("--smoothing", "-s", type=int, default=5,
                        help="Smoothing window size for FPS and latency (default: 5)")
    
    args = parser.parse_args()
    
    # Import common utilities
    from utils.shm_utils import get_dtype
    
    logger.info(f"Starting Camera visualization...")
    logger.info(f"Shared memory: {args.shared_memory}")
    logger.info(f"Data type: {args.dtype}")
    
    visualizer = CameraVisualizer(
        shared_memory_name=args.shared_memory, 
        data_dtype=get_dtype(args.dtype),
        smoothing_window=args.smoothing
    )
    
    visualizer.start_visualization(update_interval=args.interval)


if __name__ == "__main__":
    main() 