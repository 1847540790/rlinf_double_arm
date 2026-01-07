"""Example to use vive trackers on a Flexiv dual-arm platform.

Author: Chenxi Wang
"""

import time
import numpy as np
from easyrobot.arm.flexiv import FlexivArm
from easyrobot.gripper.robotiq import Robotiq2F85Gripper
from pynput import keyboard

from vive_tracker import HTCViveTracker

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
    global PAUSE, STOP,GRIPPER_OPEN_LEFT, GRIPPER_OPEN_RIGHT
    try:
        if key.char == 't':
            PAUSE = not PAUSE
            print(f"Pause set to {PAUSE}")
        if key.char == 'q':
            STOP = True
            print(f"Stop teleoperation")
        if key.char == '0' and not PAUSE:
            GRIPPER_OPEN_LEFT = not PREV_GRIPPER_OPEN_LEFT
            print(f"Left gripper open set to {GRIPPER_OPEN_LEFT}")
        if key.char == '1' and not PAUSE:
            GRIPPER_OPEN_RIGHT = not PREV_GRIPPER_OPEN_RIGHT
            print(f"Right gripper open set to {GRIPPER_OPEN_RIGHT}")
    except AttributeError:
        pass

def _on_release(key):
    pass

def init_robot():
    robot_left = FlexivArm("Rizon4-062077")
    robot_right = FlexivArm("Rizon4R-062016")
    robot_left.send_tcp_pose(INIT_ROBOT_POSE_LEFT)
    robot_right.send_tcp_pose(INIT_ROBOT_POSE_RIGHT)
    robot_right.set_cartesian_impedance([10000, 10000, 10000, 500, 500, 500])
    return robot_left, robot_right

def init_gripper():
    gripper_left = Robotiq2F85Gripper("/dev/ttyUSB2", streaming_freq = 50)
    gripper_right = Robotiq2F85Gripper("/dev/ttyUSB3", streaming_freq = 50)
    return gripper_left, gripper_right

if __name__ == "__main__":
    robot_left, robot_right = init_robot()
    gripper_left, gripper_right = init_gripper()
    # tracker = HTCViveTracker(2, auto_calib_tracker=True)
    tracker = HTCViveTracker(serials=["LHR-37267EB0", "LHR-ED74041F"], auto_calib_tracker=True)
    # # uncomment the following line for the first time
    # tracker.calibrate_robot_tracker_transform([INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT], CALIB_PATH)
    tracker.set_calibration_from_file([INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT], CALIB_PATH)
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, daemon=True)
    listener.start()
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
        print(tcp_left, tcp_right)
        robot_left.send_tcp_pose(tcp_left, max_vel = 1, max_acc = 3, max_angular_vel = 2, max_angular_acc = 8)
        robot_right.send_tcp_pose(tcp_right, max_vel = 1, max_acc = 3, max_angular_vel = 2, max_angular_acc = 8)
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