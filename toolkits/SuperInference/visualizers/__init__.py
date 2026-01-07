"""
Visualizers package for device data visualization.

Author: Jun Lv
"""

from .base_visualizer import BaseVisualizer
from .camera_visualizer import CameraVisualizer
from .action_visualizer import ActionVisualizer

# Visualizer class mapping - add new visualizer classes here
VISUALIZER_CLASSES = {
    'BaseDevice': BaseVisualizer,
    'BaseRobot': BaseVisualizer,
    'SimRobot': BaseVisualizer,
    'OpenCVCameraDevice': CameraVisualizer,
    'HikCameraDevice': CameraVisualizer,  # Hikvision camera uses the same visualizer as OpenCV camera
    'ViveTrackerDevice': BaseVisualizer,
}

__all__ = ['BaseVisualizer', 'CameraVisualizer', 'ActionVisualizer', 'VISUALIZER_CLASSES']