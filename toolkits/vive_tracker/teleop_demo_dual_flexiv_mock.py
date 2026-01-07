"""Mock robot and gripper for testing vive trackers on a simulated Flexiv dual-arm platform.

This script mocks the FlexivArm and Robotiq2F85Gripper classes to allow testing
the Vive tracker without real hardware.

Author: Chenxi Wang (original), Mock version adapted for testing
"""

import time
import numpy as np
from pynput import keyboard

from vive_tracker import HTCViveTracker


class MockFlexivArm:
    """Mock implementation of FlexivArm for testing without hardware."""

    def __init__(self, serial: str):
        self.serial = serial
        self.current_pose = np.array([0.5, 0.0, 0.3, 0, 0, 1, 0])
        self.cartesian_impedance = [10000, 10000, 10000, 500, 500, 500]
        print(f"[MockFlexivArm] Initialized robot with serial: {serial}")

    def send_tcp_pose(
        self,
        pose: np.ndarray,
        max_vel: float = 0.5,
        max_acc: float = 1.0,
        max_angular_vel: float = 1.0,
        max_angular_acc: float = 2.0,
    ):
        """Mock sending TCP pose command."""
        self.current_pose = pose.copy()
        pos = pose[:3]
        quat = pose[3:7]
        print(
            f"[MockFlexivArm {self.serial}] TCP pose -> "
            f"pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "
            f"quat: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
        )

    def set_cartesian_impedance(self, impedance: list):
        """Mock setting cartesian impedance."""
        self.cartesian_impedance = impedance
        print(f"[MockFlexivArm {self.serial}] Cartesian impedance set to: {impedance}")

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose."""
        return self.current_pose.copy()


class MockRobotiq2F85Gripper:
    """Mock implementation of Robotiq2F85Gripper for testing without hardware."""

    def __init__(self, port: str, streaming_freq: int = 50):
        self.port = port
        self.streaming_freq = streaming_freq
        self.is_open = False
        self.position = 0  # 0 = closed, 255 = fully open
        print(f"[MockRobotiq2F85Gripper] Initialized gripper on port: {port}")

    def open_gripper(self):
        """Mock opening the gripper."""
        self.is_open = True
        self.position = 255
        print(f"[MockRobotiq2F85Gripper {self.port}] Gripper OPENED")

    def close_gripper(self):
        """Mock closing the gripper."""
        self.is_open = False
        self.position = 0
        print(f"[MockRobotiq2F85Gripper {self.port}] Gripper CLOSED")

    def set_position(self, position: int):
        """Mock setting gripper position (0-255)."""
        self.position = max(0, min(255, position))
        self.is_open = self.position > 127
        print(f"[MockRobotiq2F85Gripper {self.port}] Position set to: {self.position}")

    def get_position(self) -> int:
        """Return current gripper position."""
        return self.position


INIT_ROBOT_POSE_LEFT = np.array([0.7, 0.1, 0, 0, 0, 1, 0])
INIT_ROBOT_POSE_RIGHT = np.array([0.7, -0.1, -0.02, 0, 0, 1, 0])
CALIB_PATH = "calib.json"
PAUSE = True
STOP = False
PREV_GRIPPER_OPEN_LEFT = False
PREV_GRIPPER_OPEN_RIGHT = False
GRIPPER_OPEN_LEFT = False
GRIPPER_OPEN_RIGHT = False


def _on_press(key):
    global PAUSE, STOP, GRIPPER_OPEN_LEFT, GRIPPER_OPEN_RIGHT
    try:
        if key.char == "t":
            PAUSE = not PAUSE
            print(f"Pause set to {PAUSE}")
        if key.char == "q":
            STOP = True
            print(f"Stop teleoperation")
        if key.char == "0" and not PAUSE:
            GRIPPER_OPEN_LEFT = not PREV_GRIPPER_OPEN_LEFT
            print(f"Left gripper open set to {GRIPPER_OPEN_LEFT}")
        if key.char == "1" and not PAUSE:
            GRIPPER_OPEN_RIGHT = not PREV_GRIPPER_OPEN_RIGHT
            print(f"Right gripper open set to {GRIPPER_OPEN_RIGHT}")
    except AttributeError:
        pass


def _on_release(key):
    pass


def init_robot():
    """Initialize mock robots."""
    robot_left = MockFlexivArm("Rizon4-062077-MOCK")
    robot_right = MockFlexivArm("Rizon4R-062016-MOCK")
    robot_left.send_tcp_pose(INIT_ROBOT_POSE_LEFT)
    robot_right.send_tcp_pose(INIT_ROBOT_POSE_RIGHT)
    robot_right.set_cartesian_impedance([10000, 10000, 10000, 500, 500, 500])
    return robot_left, robot_right


def init_gripper():
    """Initialize mock grippers."""
    gripper_left = MockRobotiq2F85Gripper("/dev/ttyUSB2-MOCK", streaming_freq=50)
    gripper_right = MockRobotiq2F85Gripper("/dev/ttyUSB3-MOCK", streaming_freq=50)
    return gripper_left, gripper_right


if __name__ == "__main__":
    print("=" * 60)
    print("MOCK TELEOPERATION DEMO - Testing Vive Tracker")
    print("=" * 60)
    print("Controls:")
    print("  t - Toggle pause/resume teleoperation")
    print("  0 - Toggle left gripper open/close")
    print("  1 - Toggle right gripper open/close")
    print("  q - Quit")
    print("=" * 60)

    robot_left, robot_right = init_robot()
    gripper_left, gripper_right = init_gripper()

    # Initialize tracker with specified serials
    tracker = HTCViveTracker(
        serials=["LHR-EFDCEC3F", "LHR-146E8F33"], auto_calib_tracker=True
    )

    # Load calibration from file
    # Uncomment the following line to calibrate for the first time:
    # tracker.calibrate_robot_tracker_transform([INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT], CALIB_PATH)
    tracker.set_calibration_from_file(
        [INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT], CALIB_PATH
    )

    listener = keyboard.Listener(
        on_press=_on_press, on_release=_on_release, daemon=True
    )
    listener.start()

    print("\n[INFO] Mock teleoperation started. Press 't' to start tracking.\n")

    while True:
        time.sleep(0.05)
        if STOP:
            break
        if PAUSE:
            tracker.pause()
            continue
        else:
            tracker.start()

        tcp_left, tcp_right = tracker.get_aligned_robot_pose()
        print(f"[Tracker] Left: {tcp_left[:3]}, Right: {tcp_right[:3]}")
        robot_left.send_tcp_pose(
            tcp_left, max_vel=1, max_acc=3, max_angular_vel=2, max_angular_acc=8
        )
        robot_right.send_tcp_pose(
            tcp_right, max_vel=1, max_acc=3, max_angular_vel=2, max_angular_acc=8
        )

        if GRIPPER_OPEN_LEFT != PREV_GRIPPER_OPEN_LEFT:
            if GRIPPER_OPEN_LEFT:
                gripper_left.open_gripper()
            else:
                gripper_left.close_gripper()
            PREV_GRIPPER_OPEN_LEFT = GRIPPER_OPEN_LEFT

        if GRIPPER_OPEN_RIGHT != PREV_GRIPPER_OPEN_RIGHT:
            if GRIPPER_OPEN_RIGHT:
                gripper_right.open_gripper()
            else:
                gripper_right.close_gripper()
            PREV_GRIPPER_OPEN_RIGHT = GRIPPER_OPEN_RIGHT

    tracker.stop()
    print("\n[INFO] Mock teleoperation stopped.")

