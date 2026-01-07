"""
Devices package for various device implementations.

Author: Jun Lv, Han Xue
"""

from .base import BaseDevice
from .realsense import RealSenseCameraDevice
from .camera import OpenCVCameraDevice, HikCameraDevice
from .iphone import IPhoneCameraDevice
from .robot import BaseRobot, SimRobot, DualArmSimRobot, RizonRobot
from .vive_tracker import ViveTrackerDevice 
from .teleoperator import JointSlider, DualArmJointSlider, Airexoskeleton, DualAirexoskeleton
from .rotary_encoder import RotaryEncoderDevice

DEVICE_CLASSES = {
    'BaseDevice': BaseDevice,
    'OpenCVCameraDevice': OpenCVCameraDevice,
    'IPhoneCameraDevice': IPhoneCameraDevice,
    'RealSense':RealSenseCameraDevice,
    'HikCameraDevice': HikCameraDevice,
    'BaseRobot': BaseRobot,
    'SimRobot': SimRobot,
    'DualArmSimRobot': DualArmSimRobot,
    'JointSlider': JointSlider,
    'DualArmJointSlider': DualArmJointSlider,
    'Airexoskeleton': Airexoskeleton,
    'DualAirexoskeleton': DualAirexoskeleton,
    'ViveTrackerDevice': ViveTrackerDevice,
    'RizonRobot': RizonRobot,
    'RotaryEncoderDevice': RotaryEncoderDevice,
}
__all__ = ['BaseDevice', 'ViveTrackerDevice', 'OpenCVCameraDevice', 'IPhoneCameraDevice', 'HikCameraDevice', 'RealSense', 'BaseRobot', 'SimRobot', 'DualArmSimRobot', 'JointSlider', 'DualArmJointSlider', 'Airexoskeleton', 'DualAirexoskeleton', 'RizonRobot','RotaryEncoderDevice', 'DEVICE_CLASSES'] 
