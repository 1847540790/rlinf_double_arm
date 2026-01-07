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

"""Interactive test script for keyboard intervention.

This script tests the HILInterventionWrapper with real keyboard input.
Run this script and use keyboard controls to test human intervention.

Usage:
    python tests/unit_tests/test_keyboard_intervention.py

Keyboard Controls:
    CAPS LOCK - Toggle intervention ON/OFF (press once to enable, again to disable)
    
    When intervention is ON, use these keys:
    
    Movement (Translation):
        W/S - Forward/Backward (X axis)
        A/D - Left/Right (Y axis)
        Q/E - Up/Down (Z axis)
    
    Rotation:
        U/O - Roll
        I/K - Pitch
        J/L - Yaw
    
    Gripper:
        SPACE - Toggle gripper (open/close)
    
    Other:
        Ctrl+C - Exit the test
"""

import sys
import time
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add the project root to path if needed
sys.path.insert(0, "/home/flexiv/workspace/RL/git/rlinf")

from rlinf.envs.hil.input_devices import InputDeviceConfig
from rlinf.envs.hil.intervention_wrapper import (
    ActionRecordingWrapper,
    HILInterventionWrapper,
)


class MockRobotEnv(gym.Env):
    """A simple mock robot environment for testing keyboard intervention.
    
    Simulates a 7-DoF robot arm with visual feedback in the terminal.
    """
    
    def __init__(self, action_dim: int = 7, max_episode_steps: int = 1000):
        super().__init__()
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        self._step_count = 0
        self._position = np.zeros(3)  # XYZ position
        self._rotation = np.zeros(3)  # RPY rotation
        self._gripper = 1.0  # Gripper state (1=open, -1=closed)
        
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._position = np.zeros(3)
        self._rotation = np.zeros(3)
        self._gripper = 1.0
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        
        # Update simulated robot state based on action
        self._position += action[:3] * 0.01  # XYZ
        self._rotation += action[3:6] * 0.05  # RPY
        if len(action) > 6:
            self._gripper = np.sign(action[6]) if abs(action[6]) > 0.1 else self._gripper
        
        # Clip position and rotation to reasonable bounds
        self._position = np.clip(self._position, -1.0, 1.0)
        self._rotation = np.clip(self._rotation, -np.pi, np.pi)
        
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        reward = 0.0
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self._position, 
            self._rotation, 
            [self._gripper],
            np.zeros(7)  # Padding
        ])
    
    def get_state_str(self) -> str:
        """Get a string representation of the robot state."""
        gripper_str = "OPEN" if self._gripper > 0 else "CLOSED"
        return (
            f"Position: X={self._position[0]:+.3f}, Y={self._position[1]:+.3f}, Z={self._position[2]:+.3f}\n"
            f"Rotation: R={np.degrees(self._rotation[0]):+.1f}°, P={np.degrees(self._rotation[1]):+.1f}°, Y={np.degrees(self._rotation[2]):+.1f}°\n"
            f"Gripper:  {gripper_str}"
        )


def print_header():
    """Print the test header with controls."""
    print("\n" + "=" * 70)
    print("          KEYBOARD INTERVENTION TEST")
    print("=" * 70)
    print("""
CONTROLS:
  CAPS LOCK: Toggle intervention ON/OFF (press once to enable/disable)
  
  When intervention is ON:
    Movement:  W/S (X), A/D (Y), Q/E (Z)
    Rotation:  U/O (Roll), I/K (Pitch), J/L (Yaw)
    Gripper:   SPACE (toggle)
  
  Exit: Ctrl+C

The policy outputs random actions. Press CAPS LOCK then use movement keys.
""")
    print("=" * 70 + "\n")


def format_action(action: np.ndarray) -> str:
    """Format action array for display."""
    return (
        f"[X:{action[0]:+.3f}, Y:{action[1]:+.3f}, Z:{action[2]:+.3f}, "
        f"R:{action[3]:+.3f}, P:{action[4]:+.3f}, Y:{action[5]:+.3f}, G:{action[6]:+.3f}]"
    )


def run_interactive_test(
    num_steps: int = 500,
    control_freq: float = 10.0,  # Hz
    random_policy: bool = True,
):
    """Run an interactive test with keyboard intervention.
    
    Args:
        num_steps: Maximum number of steps to run
        control_freq: Control frequency in Hz
        random_policy: If True, use random policy actions; otherwise use zero actions
    """
    print_header()
    
    # Create environment with keyboard intervention
    env = MockRobotEnv()
    
    try:
        hil_env = HILInterventionWrapper(
            env,
            input_device="keyboard",
            input_device_config=InputDeviceConfig(action_dim=7),
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize keyboard device: {e}")
        print("\nMake sure 'pynput' is installed:")
        print("  pip install pynput")
        print("\nNote: On Linux, you may need to run as root or add yourself to the 'input' group:")
        print("  sudo usermod -aG input $USER")
        print("  (Then log out and back in)")
        return
    
    recording_env = ActionRecordingWrapper(hil_env)
    
    print("[INFO] Starting test loop... Press Ctrl+C to stop.\n")
    
    obs, _ = recording_env.reset()
    dt = 1.0 / control_freq
    
    intervention_total = 0
    step_count = 0
    
    try:
        for step in range(num_steps):
            step_count = step + 1
            
            # Generate policy action
            if random_policy:
                policy_action = np.random.uniform(-0.3, 0.3, size=7).astype(np.float32)
            else:
                policy_action = np.zeros(7, dtype=np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, info = recording_env.step(policy_action)
            
            is_intervention = info.get("is_intervention", False)
            if is_intervention:
                intervention_total += 1
            
            # Clear screen and print status (using ANSI escape codes)
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
            print_header()
            
            # Print status
            intervention_str = "\033[92m>>> HUMAN INTERVENTION <<<\033[0m" if is_intervention else "    Policy Control"
            print(f"Step: {step_count:4d}/{num_steps}  |  {intervention_str}")
            print("-" * 50)
            
            # Robot state
            print("\nROBOT STATE:")
            print(env.get_state_str())
            
            # Action info
            print("\nACTIONS:")
            print(f"  Policy:   {format_action(info['policy_action'])}")
            print(f"  Executed: {format_action(info['executed_action'])}")
            if is_intervention and "intervene_action" in info:
                print(f"  Human:    {format_action(info['intervene_action'])}")
            
            # Stats
            stats = hil_env.get_stats()
            intervention_rate = stats["hil/intervention_rate"] * 100
            print(f"\nSTATISTICS:")
            print(f"  Intervention Steps: {intervention_total}/{step_count}")
            print(f"  Intervention Rate:  {intervention_rate:.1f}%")
            print(f"  Intervention Episodes: {stats['hil/intervention_count']}")
            
            # Hint
            print("\n\033[93m[TIP] Press CAPS LOCK to toggle intervention, then use W/A/S/D/Q/E!\033[0m")
            
            if terminated or truncated:
                print("\n[INFO] Episode ended. Resetting...")
                obs, _ = recording_env.reset()
                time.sleep(0.5)
            
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user.")
    finally:
        # Print final summary
        print("\n" + "=" * 70)
        print("                    FINAL SUMMARY")
        print("=" * 70)
        
        stats = hil_env.get_stats()
        print(f"\nTotal Steps:           {stats['hil/total_steps']}")
        print(f"Intervention Steps:    {stats['hil/intervention_steps']}")
        print(f"Intervention Rate:     {stats['hil/intervention_rate']*100:.2f}%")
        print(f"Intervention Episodes: {stats['hil/intervention_count']}")
        
        # Save recorded data
        recorded = recording_env.get_recorded_data()
        if recorded:
            print(f"\nRecorded {len(recorded)} steps of data.")
            save_path = "/tmp/keyboard_intervention_recording.pkl"
            recording_env.save_recorded_data(save_path)
            print(f"Saved recording to: {save_path}")
        
        recording_env.close()
        print("\n[INFO] Test completed.")


def run_simple_test():
    """Run a simpler test that just checks if keyboard intervention works."""
    print("\n" + "=" * 50)
    print("     SIMPLE KEYBOARD INTERVENTION TEST")
    print("=" * 50)
    print("\nThis test will run for 100 steps.")
    print("Press CAPS LOCK to toggle intervention ON, then use W/A/S/D.\n")
    
    env = MockRobotEnv()
    
    try:
        hil_env = HILInterventionWrapper(
            env,
            input_device="keyboard",
            input_device_config=InputDeviceConfig(action_dim=7),
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        print("\nInstall pynput: pip install pynput")
        return False
    
    obs, _ = hil_env.reset()
    
    intervention_detected = False
    
    try:
        for step in range(100):
            policy_action = np.zeros(7)
            obs, reward, terminated, truncated, info = hil_env.step(policy_action)
            
            if info["is_intervention"]:
                intervention_detected = True
                print(f"[Step {step}] Intervention detected! Action: {info['intervene_action'][:3]}")
            else:
                print(f"[Step {step}] No intervention", end="\r")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    finally:
        hil_env.close()
    
    print("\n" + "-" * 50)
    if intervention_detected:
        print("✓ SUCCESS: Keyboard intervention is working!")
    else:
        print("✗ No intervention detected. Press CAPS LOCK to enable intervention first.")
    print("-" * 50 + "\n")
    
    return intervention_detected


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test keyboard intervention")
    parser.add_argument(
        "--simple", 
        action="store_true",
        help="Run a simple test instead of the full interactive test"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run (default: 500)"
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="Control frequency in Hz (default: 10.0)"
    )
    parser.add_argument(
        "--zero-policy",
        action="store_true",
        help="Use zero policy actions instead of random"
    )
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_test()
    else:
        run_interactive_test(
            num_steps=args.steps,
            control_freq=args.freq,
            random_policy=not args.zero_policy,
        )

