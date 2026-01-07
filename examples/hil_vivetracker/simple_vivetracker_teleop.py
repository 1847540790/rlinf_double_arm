#!/usr/bin/env python3
"""Simple ViveTracker Teleoperation for Dual Flexiv Arms.

This is a minimal example showing how to use ViveTrackers for teleoperating
dual Flexiv robot arms using the RLinf input device wrapper.

On first run:
    - Creates default config file (vive_robot_config.yaml) if not present
    - Prompts for tracker serial numbers and robot poses
    - Performs calibration and saves to calibration file

Usage:
    # First run - interactive setup
    python simple_vivetracker_teleop.py
    
    # With custom config path
    python simple_vivetracker_teleop.py --config my_config.yaml
    
    # Recalibrate
    python simple_vivetracker_teleop.py --calibrate
    
    # Non-interactive (use defaults)
    python simple_vivetracker_teleop.py --no-interactive

Controls:
    - Move ViveTracker to move robot arm
    - Press keyboard 'P' to pause/resume
    - Press keyboard 'G' to toggle left gripper, 'H' for right
    - Press keyboard 'ESC' to stop
"""

import argparse
import os
import sys
import time
import threading

import numpy as np

# Add rlinf to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlinf.envs.hil.input_devices import (
    ViveTrackerInputDevice,
    DualViveTrackerInputDevice,
    ViveTrackerConfig,
    InputDeviceConfig,
)


# ============================================
# Default Configuration
# ============================================

DEFAULT_CONFIG_PATH = "vive_robot_config.yaml"
DEFAULT_CALIB_PATH = "vive_calib.json"

# Motion limits for Flexiv arms
MAX_VEL = 1.0  # m/s
MAX_ACC = 3.0  # m/s^2
MAX_ANGULAR_VEL = 2.0  # rad/s
MAX_ANGULAR_ACC = 8.0  # rad/s^2

# Control frequency
CONTROL_FREQ = 20.0  # Hz


# ============================================
# Global stop/pause flags
# ============================================
STOP = False
PAUSE = False


# ============================================
# Mock Robot for Testing
# ============================================

class MockFlexivRobot:
    """Mock Flexiv robot for testing without hardware."""
    
    def __init__(self, name: str):
        self.name = name
        self._tcp_pose = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._gripper_open = True
        
    def send_tcp_pose(self, pose, max_vel=1, max_acc=3, max_angular_vel=2, max_angular_acc=8):
        """Send TCP pose command (mock)."""
        self._tcp_pose = pose[:6] if len(pose) >= 6 else pose
        
    def get_tcp_pose(self):
        """Get current TCP pose (mock)."""
        return self._tcp_pose.copy()
        
    def open_gripper(self):
        """Open gripper (mock)."""
        self._gripper_open = True
        
    def close_gripper(self):
        """Close gripper (mock)."""
        self._gripper_open = False
        
    def get_gripper_state(self):
        """Get gripper state (mock)."""
        return 1.0 if self._gripper_open else -1.0


def init_robot(use_mock: bool = True):
    """Initialize dual Flexiv robot arms.
    
    Replace the mock implementation with your actual robot SDK.
    
    Args:
        use_mock: If True, use mock robots for testing
        
    Returns:
        Tuple of (robot_left, robot_right)
    """
    if use_mock:
        print("[Robot] Using mock robots for testing.")
        return MockFlexivRobot("left"), MockFlexivRobot("right")
    
    # TODO: Replace with actual robot initialization
    # Example using flexiv_sdk:
    # from flexiv_sdk import FlexivRobot
    # robot_left = FlexivRobot("192.168.2.100")
    # robot_right = FlexivRobot("192.168.2.101")
    # robot_left.enable()
    # robot_right.enable()
    # return robot_left, robot_right
    
    print("[Robot] Real robot SDK not implemented. Using mock.")
    return MockFlexivRobot("left"), MockFlexivRobot("right")


# ============================================
# Keyboard Controller
# ============================================

class KeyboardController:
    """Keyboard controller for pause/stop and gripper controls."""
    
    def __init__(self, tracker_device: ViveTrackerInputDevice):
        self.tracker_device = tracker_device
        self._stop = False
        self._pause = True  # Start paused
        self._listener = None
        self._lock = threading.Lock()
        
    def start(self):
        """Start keyboard listener."""
        try:
            from pynput import keyboard
            
            def on_press(key):
                with self._lock:
                    try:
                        if hasattr(key, 'char') and key.char:
                            k = key.char.lower()
                            if k == 'p':
                                self._pause = not self._pause
                                if self._pause:
                                    self.tracker_device.pause()
                                    print("[Keyboard] PAUSED")
                                else:
                                    self.tracker_device.resume()
                                    print("[Keyboard] RESUMED")
                            elif k == 'g':
                                self.tracker_device.toggle_gripper(0)
                                print("[Keyboard] Left gripper toggled")
                            elif k == 'h':
                                self.tracker_device.toggle_gripper(1)
                                print("[Keyboard] Right gripper toggled")
                    except AttributeError:
                        pass
                        
                    if key == keyboard.Key.esc:
                        self._stop = True
                        print("[Keyboard] Stopping...")
                        return False
                        
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            
            print("[KeyboardController] Started")
            print("  'P' - Pause/Resume tracking")
            print("  'G' - Toggle left gripper")
            print("  'H' - Toggle right gripper")
            print("  'ESC' - Stop")
            
        except ImportError:
            print("[KeyboardController] pynput not installed. Run: pip install pynput")
            
    def stop(self):
        """Stop keyboard listener."""
        if self._listener:
            self._listener.stop()
            
    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._stop
            
    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._pause


# ============================================
# Main Teleoperation Loop
# ============================================

def run_teleop(
    tracker: DualViveTrackerInputDevice,
    robot_left,
    robot_right,
    control_freq: float = 20.0,
):
    """Run the teleoperation loop.
    
    Args:
        tracker: ViveTracker input device
        robot_left: Left robot arm
        robot_right: Right robot arm
        control_freq: Control loop frequency in Hz
    """
    print("\n" + "=" * 60)
    print("TELEOPERATION ACTIVE")
    print("=" * 60)
    print("Move trackers to control robot arms.")
    print("Press 'P' to start, 'ESC' to stop.\n")
    
    # Setup keyboard controller
    keyboard_ctrl = KeyboardController(tracker)
    keyboard_ctrl.start()
    
    # Start paused
    tracker.pause()
    
    dt = 1.0 / control_freq
    
    try:
        while not keyboard_ctrl.should_stop:
            time.sleep(dt)
            
            # Get action from tracker
            action, is_intervening = tracker.get_action()
            
            if not is_intervening:
                continue
                
            # Split action for left and right arms
            action_left = action[:7]   # dx, dy, dz, drx, dry, drz, gripper
            action_right = action[7:14] if len(action) >= 14 else np.zeros(7)
            
            # Execute on left robot
            current_left = robot_left.get_tcp_pose()
            target_left = np.zeros(6)
            target_left[:3] = current_left[:3] + action_left[:3]
            target_left[3:6] = current_left[3:6] + action_left[3:6]
            robot_left.send_tcp_pose(
                target_left,
                max_vel=MAX_VEL,
                max_acc=MAX_ACC,
                max_angular_vel=MAX_ANGULAR_VEL,
                max_angular_acc=MAX_ANGULAR_ACC,
            )
            
            # Handle left gripper
            if action_left[6] > 0.5:
                robot_left.open_gripper()
            elif action_left[6] < -0.5:
                robot_left.close_gripper()
            
            # Execute on right robot
            current_right = robot_right.get_tcp_pose()
            target_right = np.zeros(6)
            target_right[:3] = current_right[:3] + action_right[:3]
            target_right[3:6] = current_right[3:6] + action_right[3:6]
            robot_right.send_tcp_pose(
                target_right,
                max_vel=MAX_VEL,
                max_acc=MAX_ACC,
                max_angular_vel=MAX_ANGULAR_VEL,
                max_angular_acc=MAX_ANGULAR_ACC,
            )
            
            # Handle right gripper
            if action_right[6] > 0.5:
                robot_right.open_gripper()
            elif action_right[6] < -0.5:
                robot_right.close_gripper()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        keyboard_ctrl.stop()
        print("Teleoperation stopped.")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="ViveTracker Teleoperation for Dual Flexiv Arms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First run (interactive setup):
    python simple_vivetracker_teleop.py
    
  With custom config:
    python simple_vivetracker_teleop.py --config my_config.yaml
    
  Recalibrate:
    python simple_vivetracker_teleop.py --calibrate
    
  Non-interactive (use defaults):
    python simple_vivetracker_teleop.py --no-interactive --mock
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Force recalibration even if calibration file exists",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Non-interactive mode (use defaults, no prompts)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock robots (no real hardware)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default config file and exit",
    )
    args = parser.parse_args()
    
    # Handle create-config
    if args.create_config:
        path = ViveTrackerConfig.create_default_config_file(args.config)
        print(f"Created default config file: {path}")
        print("Edit this file and run again.")
        return
    
    print("\n" + "=" * 60)
    print("ViveTracker Teleoperation for Dual Flexiv Arms")
    print("=" * 60)
    
    # Create tracker device from config file
    # This will:
    #   - Create default config if not present
    #   - Prompt for tracker serials and robot poses (if interactive)
    #   - Load existing calibration or prompt for new calibration
    print("\n[1/3] Loading configuration...")
    
    try:
        tracker = DualViveTrackerInputDevice.from_config_file(
            config_path=args.config,
            create_if_missing=True,
            interactive=not args.no_interactive,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run with --create-config to create a default config file.")
        return
    except ImportError as e:
        print(f"Error: {e}")
        return
        
    # Initialize robots
    print("\n[2/3] Initializing robots...")
    robot_left, robot_right = init_robot(use_mock=args.mock)
    
    # Start tracker
    print("\n[3/3] Starting tracker...")
    tracker.start()
    
    # Handle forced recalibration
    if args.calibrate:
        print("\nForcing recalibration...")
        tracker.calibrate(save=True)
    
    try:
        # Run teleoperation
        run_teleop(
            tracker=tracker,
            robot_left=robot_left,
            robot_right=robot_right,
            control_freq=tracker.tracker_config.control_freq,
        )
    finally:
        # Cleanup
        print("\nCleaning up...")
        tracker.stop()
        print("Done.")


if __name__ == "__main__":
    main()
