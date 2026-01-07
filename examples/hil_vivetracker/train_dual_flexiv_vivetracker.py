#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for Dual Flexiv Arm HIL with ViveTracker Teleoperation.

This script demonstrates how to use HTC Vive Trackers for human-in-the-loop
teleoperation of dual Flexiv robot arms during RL training.

Features:
    - Dual arm control with two ViveTrackers
    - Automatic coordinate calibration between trackers and robot frames
    - Keyboard controls for pause/resume and gripper
    - HIL-augmented PPO training

Usage:
    # First time: Calibrate trackers
    python train_dual_flexiv_vivetracker.py --calibrate
    
    # Training with HIL
    python train_dual_flexiv_vivetracker.py --config config/dual_flexiv_hil.yaml
    
    # Demo collection (no policy, pure teleoperation)
    python train_dual_flexiv_vivetracker.py --demo_only --save_demos demos.pkl

Controls:
    - Move ViveTracker to control robot arm
    - Press 'P' to pause/resume tracking
    - Press 'G' to toggle left gripper, 'H' to toggle right gripper
    - Press 'ESC' to stop
"""

import argparse
import os
import sys
import time
import threading
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal

# Add rlinf to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlinf.runners.embodied_hil_runner import EmbodiedHILRunner, EmbodiedHILConfig
from rlinf.envs.hil.input_devices import (
    ViveTrackerInputDevice,
    DualViveTrackerInputDevice,
    ViveTrackerConfig,
    InputDeviceConfig,
)


# ============================================
# Configuration
# ============================================

# Vive Tracker serial numbers - replace with your tracker serials
TRACKER_SERIAL_LEFT = "LHR-37267EB0"
TRACKER_SERIAL_RIGHT = "LHR-ED74041F"

# Initial robot TCP poses [x, y, z, qx, qy, qz, qw]
INIT_ROBOT_POSE_LEFT = np.array([0.7, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0])
INIT_ROBOT_POSE_RIGHT = np.array([0.7, -0.1, -0.02, 0.0, 0.0, 1.0, 0.0])

# Calibration file path
CALIB_PATH = "vive_calib.json"


# ============================================
# Dual Flexiv Environment
# ============================================

class DualFlexivEnv:
    """Dual Flexiv robot arm environment.
    
    This is a wrapper that interfaces with two Flexiv robot arms.
    Replace this with your actual robot interface.
    """
    
    def __init__(
        self,
        left_robot_ip: str = "192.168.2.100",
        right_robot_ip: str = "192.168.2.101",
        control_freq: float = 20.0,
        use_mock: bool = False,
    ):
        self.left_robot_ip = left_robot_ip
        self.right_robot_ip = right_robot_ip
        self.control_freq = control_freq
        self.use_mock = use_mock
        
        self.robot_left = None
        self.robot_right = None
        
        # Action/observation dimensions
        self.action_dim = 14  # 7 per arm (dx, dy, dz, drx, dry, drz, gripper)
        self.obs_dim = 28     # State from both arms
        
        # Motion limits
        self.max_vel = 1.0
        self.max_acc = 3.0
        self.max_angular_vel = 2.0
        self.max_angular_acc = 8.0
        
        if not use_mock:
            self._init_robots()
            
    def _init_robots(self):
        """Initialize connection to robots."""
        try:
            # Import your robot SDK here
            # from flexiv_sdk import FlexivRobot
            # self.robot_left = FlexivRobot(self.left_robot_ip)
            # self.robot_right = FlexivRobot(self.right_robot_ip)
            print(f"[DualFlexivEnv] Initializing robots at {self.left_robot_ip}, {self.right_robot_ip}")
            print("[DualFlexivEnv] Robot SDK not imported - using mock mode")
            self.use_mock = True
        except ImportError:
            print("[DualFlexivEnv] Robot SDK not available. Using mock mode.")
            self.use_mock = True
            
    def reset(self):
        """Reset robots to initial poses."""
        print("[DualFlexivEnv] Resetting to initial poses...")
        if not self.use_mock and self.robot_left and self.robot_right:
            # Move to initial poses
            self.robot_left.move_to_pose(INIT_ROBOT_POSE_LEFT)
            self.robot_right.move_to_pose(INIT_ROBOT_POSE_RIGHT)
        return self.get_observation()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action on both arms.
        
        Args:
            action: [14] = [dx_L, dy_L, dz_L, drx_L, dry_L, drz_L, grip_L,
                          dx_R, dy_R, dz_R, drx_R, dry_R, drz_R, grip_R]
                          
        Returns:
            observation, reward, done, info
        """
        action_left = action[:7]
        action_right = action[7:14]
        
        if not self.use_mock and self.robot_left and self.robot_right:
            # Execute on real robots in parallel
            def send_left():
                self._execute_action(self.robot_left, action_left)
            def send_right():
                self._execute_action(self.robot_right, action_right)
                
            t1 = threading.Thread(target=send_left)
            t2 = threading.Thread(target=send_right)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        else:
            # Mock execution
            time.sleep(1.0 / self.control_freq)
            
        obs = self.get_observation()
        reward = 0.0  # Define your reward function
        done = False
        info = {}
        
        return obs, reward, done, info
        
    def _execute_action(self, robot, action: np.ndarray):
        """Execute action on a single robot.
        
        Args:
            robot: Robot instance
            action: [7] = [dx, dy, dz, drx, dry, drz, gripper]
        """
        # Get current TCP pose
        current_pose = robot.get_tcp_pose()  # [x, y, z, rx, ry, rz]
        
        # Apply delta
        target_pose = np.zeros(6)
        target_pose[:3] = current_pose[:3] + action[:3]
        target_pose[3:6] = current_pose[3:6] + action[3:6]
        
        # Send to robot
        robot.send_tcp_pose(
            target_pose,
            max_vel=self.max_vel,
            max_acc=self.max_acc,
            max_angular_vel=self.max_angular_vel,
            max_angular_acc=self.max_angular_acc,
        )
        
        # Handle gripper
        gripper_action = action[6]
        if gripper_action > 0.5:
            robot.open_gripper()
        elif gripper_action < -0.5:
            robot.close_gripper()
            
    def get_observation(self) -> np.ndarray:
        """Get observation from both arms."""
        if not self.use_mock and self.robot_left and self.robot_right:
            pose_left = self.robot_left.get_tcp_pose()
            pose_right = self.robot_right.get_tcp_pose()
            gripper_left = self.robot_left.get_gripper_state()
            gripper_right = self.robot_right.get_gripper_state()
            
            obs = np.concatenate([
                pose_left, [gripper_left],
                pose_right, [gripper_right],
            ])
        else:
            obs = np.zeros(self.obs_dim)
            
        return obs.astype(np.float32)
        
    def send_tcp_pose(self, tcp_left: np.ndarray, tcp_right: np.ndarray):
        """Send absolute TCP poses to both arms.
        
        This is used for ViveTracker absolute pose mode.
        
        Args:
            tcp_left: [7] = [x, y, z, qx, qy, qz, qw]
            tcp_right: [7] = [x, y, z, qx, qy, qz, qw]
        """
        if not self.use_mock and self.robot_left and self.robot_right:
            def send_left():
                self.robot_left.send_tcp_pose(
                    tcp_left,
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                )
            def send_right():
                self.robot_right.send_tcp_pose(
                    tcp_right,
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                )
                
            t1 = threading.Thread(target=send_left)
            t2 = threading.Thread(target=send_right)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
    def close(self):
        """Close robot connections."""
        print("[DualFlexivEnv] Closing robot connections...")
        if self.robot_left:
            self.robot_left.disconnect()
        if self.robot_right:
            self.robot_right.disconnect()


# ============================================
# Keyboard Listener for Additional Controls
# ============================================

class KeyboardController:
    """Keyboard controller for pause/resume and gripper control."""
    
    def __init__(self, tracker_device: ViveTrackerInputDevice):
        self.tracker_device = tracker_device
        self._stop = False
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
                                # Toggle pause
                                if self.tracker_device.is_paused:
                                    self.tracker_device.resume()
                                    print("[Keyboard] Tracking RESUMED")
                                else:
                                    self.tracker_device.pause()
                                    print("[Keyboard] Tracking PAUSED")
                            elif k == 'g':
                                # Toggle left gripper
                                self.tracker_device.toggle_gripper(0)
                                print("[Keyboard] Left gripper toggled")
                            elif k == 'h':
                                # Toggle right gripper
                                self.tracker_device.toggle_gripper(1)
                                print("[Keyboard] Right gripper toggled")
                    except AttributeError:
                        pass
                        
                    # Handle escape
                    if key == keyboard.Key.esc:
                        self._stop = True
                        print("[Keyboard] ESC pressed - stopping...")
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
            print("[KeyboardController] pynput not installed. Keyboard control disabled.")
            
    def stop(self):
        """Stop keyboard listener."""
        if self._listener:
            self._listener.stop()
            
    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._stop


# ============================================
# Demo Collection
# ============================================

def collect_demos(
    env: DualFlexivEnv,
    tracker: DualViveTrackerInputDevice,
    num_episodes: int = 10,
    max_steps_per_episode: int = 500,
    save_path: Optional[str] = None,
) -> List[Dict]:
    """Collect demonstrations using ViveTracker teleoperation.
    
    Args:
        env: Robot environment
        tracker: ViveTracker input device
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save demos (pickle file)
        
    Returns:
        List of demonstration episodes
    """
    print("\n" + "=" * 60)
    print("Demo Collection Mode")
    print("=" * 60)
    print(f"  Episodes to collect: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print("=" * 60 + "\n")
    
    # Setup keyboard controller
    keyboard_ctrl = KeyboardController(tracker)
    keyboard_ctrl.start()
    
    demos = []
    
    try:
        for ep in range(num_episodes):
            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
            print("Press 'P' to start tracking when ready...")
            
            # Wait for user to resume
            while tracker.is_paused and not keyboard_ctrl.should_stop:
                time.sleep(0.1)
                
            if keyboard_ctrl.should_stop:
                break
                
            # Reset environment
            obs = env.reset()
            
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "timestamps": [],
            }
            
            for step in range(max_steps_per_episode):
                if keyboard_ctrl.should_stop:
                    break
                    
                # Get action from tracker
                action, is_intervening = tracker.get_action()
                
                if not is_intervening:
                    # Not moving - skip this step
                    time.sleep(0.05)
                    continue
                    
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                # Store data
                episode_data["observations"].append(obs.copy())
                episode_data["actions"].append(action.copy())
                episode_data["rewards"].append(reward)
                episode_data["timestamps"].append(time.time())
                
                obs = next_obs
                
                if done:
                    print(f"Episode finished at step {step + 1}")
                    break
                    
            # End of episode
            tracker.pause()
            
            episode_data["observations"] = np.array(episode_data["observations"])
            episode_data["actions"] = np.array(episode_data["actions"])
            episode_data["rewards"] = np.array(episode_data["rewards"])
            episode_data["length"] = len(episode_data["rewards"])
            
            demos.append(episode_data)
            print(f"Collected episode with {episode_data['length']} steps")
            
    finally:
        keyboard_ctrl.stop()
        
    # Save demos
    if save_path and demos:
        with open(save_path, 'wb') as f:
            pickle.dump(demos, f)
        print(f"\nSaved {len(demos)} demos to {save_path}")
        
    return demos


# ============================================
# ViveTracker HIL Training
# ============================================

def run_vivetracker_hil_loop(
    env: DualFlexivEnv,
    tracker: DualViveTrackerInputDevice,
    policy: Optional[nn.Module] = None,
    device: str = "cuda",
    use_absolute_poses: bool = False,
):
    """Run ViveTracker HIL control loop.
    
    This is a simpler control loop that can be used for:
    - Testing the ViveTracker setup
    - Pure teleoperation without RL
    - HIL with a trained policy
    
    Args:
        env: Robot environment
        tracker: ViveTracker input device
        policy: Optional policy network
        device: Torch device
        use_absolute_poses: If True, send absolute poses instead of deltas
    """
    print("\n" + "=" * 60)
    print("ViveTracker HIL Control Loop")
    print("=" * 60)
    print("  Mode:", "Absolute Pose" if use_absolute_poses else "Delta Action")
    print("  Policy:", "Enabled" if policy else "Teleoperation Only")
    print("=" * 60 + "\n")
    
    # Setup keyboard controller
    keyboard_ctrl = KeyboardController(tracker)
    keyboard_ctrl.start()
    
    # Initial pause
    tracker.pause()
    print("Press 'P' to start tracking...")
    
    try:
        obs = env.reset()
        
        while not keyboard_ctrl.should_stop:
            time.sleep(1.0 / tracker.tracker_config.control_freq)
            
            # Get human action
            human_action, is_intervening = tracker.get_action()
            
            if is_intervening:
                # Human is controlling
                if use_absolute_poses:
                    # Send absolute poses directly
                    tcp_left = human_action[:7]
                    tcp_right = human_action[8:15] if len(human_action) >= 15 else human_action[:7]
                    env.send_tcp_pose(tcp_left, tcp_right)
                else:
                    # Use delta actions
                    next_obs, reward, done, info = env.step(human_action)
                    obs = next_obs
            else:
                # Human not intervening - use policy if available
                if policy is not None:
                    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action_dist = policy(obs_tensor)
                        action = action_dist.mean.cpu().numpy()[0]
                    
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    
    finally:
        keyboard_ctrl.stop()
        tracker.stop()
        env.close()
        
    print("\nControl loop ended.")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Dual Flexiv ViveTracker HIL Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dual_flexiv_hil.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run tracker calibration",
    )
    parser.add_argument(
        "--demo_only",
        action="store_true",
        help="Demo collection mode (no policy)",
    )
    parser.add_argument(
        "--save_demos",
        type=str,
        default=None,
        help="Path to save collected demos",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Run pure teleoperation (no training)",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock robot (no real hardware)",
    )
    parser.add_argument(
        "--left_serial",
        type=str,
        default=TRACKER_SERIAL_LEFT,
        help="Left tracker serial number",
    )
    parser.add_argument(
        "--right_serial",
        type=str,
        default=TRACKER_SERIAL_RIGHT,
        help="Right tracker serial number",
    )
    args = parser.parse_args()
    
    # Create ViveTracker config
    tracker_config = ViveTrackerConfig(
        serials=[args.left_serial, args.right_serial],
        init_robot_poses=[INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT],
        calib_path=CALIB_PATH,
        auto_calib_tracker=True,
        use_delta_actions=True,
        translation_scale=1.0,
        rotation_scale=1.0,
        deadzone=0.005,
        max_velocity=1.0,
        max_angular_velocity=2.0,
        control_freq=20.0,
    )
    
    # Create input device config
    input_config = InputDeviceConfig(action_dim=14)
    
    # Create tracker device
    tracker = DualViveTrackerInputDevice(
        config=input_config,
        tracker_config=tracker_config,
    )
    
    # Create robot environment
    env = DualFlexivEnv(use_mock=args.use_mock)
    
    try:
        # Start tracker
        tracker.start()
        
        # Handle calibration mode
        if args.calibrate:
            print("\n" + "=" * 60)
            print("Tracker Calibration Mode")
            print("=" * 60)
            print("1. Position the robot arms at their initial poses")
            print("2. Attach trackers to the end-effectors")
            print("3. Press Enter when ready...")
            input()
            
            success = tracker.calibrate(save=True)
            if success:
                print("Calibration complete!")
            else:
                print("Calibration failed!")
            return
            
        # Check calibration
        if not tracker.is_calibrated:
            print("\n[WARNING] Tracker is not calibrated!")
            print("Run with --calibrate flag to perform calibration first.")
            print("Continuing without calibration (may have coordinate misalignment)...")
            
        # Handle different modes
        if args.demo_only:
            collect_demos(
                env=env,
                tracker=tracker,
                num_episodes=10,
                save_path=args.save_demos,
            )
        elif args.teleop:
            run_vivetracker_hil_loop(
                env=env,
                tracker=tracker,
                policy=None,
                use_absolute_poses=False,
            )
        else:
            # TODO: Add full training loop with RL
            print("\nFull training mode not yet implemented.")
            print("Use --demo_only or --teleop for now.")
            
    finally:
        tracker.stop()
        env.close()


if __name__ == "__main__":
    main()

