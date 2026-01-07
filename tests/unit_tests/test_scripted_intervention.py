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

"""Test intervention wrapper with scripted/simulated interventions.

This test doesn't require real keyboard input - it simulates human
interventions to verify the wrapper functionality works correctly.

Use this for:
    - Automated testing in CI/CD
    - Testing on headless servers or via SSH
    - Verifying intervention logic without hardware

Usage:
    python tests/unit_tests/test_scripted_intervention.py
"""

import sys
import time
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, "/home/flexiv/workspace/RL/git/rlinf")

from rlinf.envs.hil.input_devices import BaseInputDevice, InputDeviceConfig
from rlinf.envs.hil.intervention_wrapper import (
    ActionRecordingWrapper,
    HILInterventionWrapper,
)


class ScriptedInputDevice(BaseInputDevice):
    """Input device that follows a scripted intervention pattern.
    
    Useful for testing intervention logic without real hardware.
    """
    
    def __init__(
        self, 
        config: InputDeviceConfig = None,
        intervention_schedule: List[Tuple[int, int]] = None,
    ):
        """
        Args:
            config: Input device configuration
            intervention_schedule: List of (start_step, end_step) tuples 
                                   defining when to intervene
        """
        super().__init__(config)
        self.intervention_schedule = intervention_schedule or []
        self.current_step = 0
        self._intervention_action = None
        
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Return scripted action based on current step."""
        is_intervening = self._should_intervene()
        
        if is_intervening:
            # Generate a meaningful intervention action
            # Simulate human moving the robot forward
            action = np.zeros(self.config.action_dim)
            action[0] = 0.5  # Move forward in X
            action[1] = np.sin(self.current_step * 0.1) * 0.3  # Slight Y oscillation
            if self._intervention_action is not None:
                action = self._intervention_action
        else:
            action = np.zeros(self.config.action_dim)
            
        self.current_step += 1
        return action, is_intervening
    
    def _should_intervene(self) -> bool:
        """Check if current step is within any intervention window."""
        for start, end in self.intervention_schedule:
            if start <= self.current_step < end:
                return True
        return False
    
    def set_intervention_action(self, action: np.ndarray):
        """Set a specific action to use during intervention."""
        self._intervention_action = action
        
    def reset(self):
        """Reset step counter."""
        self.current_step = 0


class MockRobotEnv(gym.Env):
    """Mock robot environment for testing."""
    
    def __init__(self, action_dim: int = 7, max_episode_steps: int = 100):
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
        self._position = np.zeros(3)
        self._last_action = None
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._position = np.zeros(3)
        return np.zeros(14, dtype=np.float32), {}
    
    def step(self, action: np.ndarray):
        self._step_count += 1
        self._last_action = action.copy()
        self._position += action[:3] * 0.01
        
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        
        obs = np.concatenate([self._position, np.zeros(11)])
        return obs, 0.0, terminated, truncated, {"position": self._position.copy()}


def run_scripted_test():
    """Run a test with scripted interventions."""
    
    print("=" * 70)
    print("     SCRIPTED INTERVENTION TEST")
    print("=" * 70)
    print()
    print("This test simulates human interventions without requiring keyboard input.")
    print("Useful for automated testing and remote/SSH environments.")
    print()
    
    # Define intervention schedule: (start_step, end_step)
    # Intervene on steps 10-20, 40-50, 70-80
    intervention_schedule = [
        (10, 20),   # First intervention window
        (40, 50),   # Second intervention window
        (70, 80),   # Third intervention window
    ]
    
    print(f"Intervention schedule: {intervention_schedule}")
    print()
    
    # Create environment with scripted input device
    env = MockRobotEnv(max_episode_steps=100)
    
    scripted_device = ScriptedInputDevice(
        config=InputDeviceConfig(action_dim=7),
        intervention_schedule=intervention_schedule,
    )
    
    hil_env = HILInterventionWrapper(env, input_device=scripted_device)
    recording_env = ActionRecordingWrapper(hil_env)
    
    obs, _ = recording_env.reset()
    
    print("-" * 70)
    print(f"{'Step':>5} | {'Intervening':^12} | {'Policy Action':^15} | {'Executed Action':^15}")
    print("-" * 70)
    
    total_steps = 100
    intervention_steps = 0
    
    for step in range(total_steps):
        # Generate random policy action
        policy_action = np.random.uniform(-0.2, 0.2, size=7).astype(np.float32)
        
        # Step environment
        obs, reward, terminated, truncated, info = recording_env.step(policy_action)
        
        is_intervention = info.get("is_intervention", False)
        if is_intervention:
            intervention_steps += 1
            
        # Print status every 5 steps or when intervention state changes
        if step % 10 == 0 or step == 0:
            policy_str = f"[{policy_action[0]:+.2f}, {policy_action[1]:+.2f}, ...]"
            exec_str = f"[{info['executed_action'][0]:+.2f}, {info['executed_action'][1]:+.2f}, ...]"
            int_str = ">>> YES <<<" if is_intervention else "    no"
            print(f"{step:5d} | {int_str:^12} | {policy_str:^15} | {exec_str:^15}")
            
        if terminated or truncated:
            break
            
    print("-" * 70)
    print()
    
    # Print statistics
    stats = hil_env.get_stats()
    print("STATISTICS:")
    print(f"  Total Steps:           {stats['hil/total_steps']}")
    print(f"  Intervention Steps:    {stats['hil/intervention_steps']}")
    print(f"  Intervention Rate:     {stats['hil/intervention_rate']*100:.1f}%")
    print(f"  Intervention Episodes: {stats['hil/intervention_count']}")
    print()
    
    # Verify expected results
    expected_intervention_steps = sum(end - start for start, end in intervention_schedule)
    print("VERIFICATION:")
    print(f"  Expected intervention steps: {expected_intervention_steps}")
    print(f"  Actual intervention steps:   {stats['hil/intervention_steps']}")
    
    if stats['hil/intervention_steps'] == expected_intervention_steps:
        print("  ✓ PASS: Intervention steps match expected!")
    else:
        print("  ✗ FAIL: Intervention steps do not match!")
        
    expected_episodes = len(intervention_schedule)
    print(f"  Expected intervention episodes: {expected_episodes}")
    print(f"  Actual intervention episodes:   {stats['hil/intervention_count']}")
    
    if stats['hil/intervention_count'] == expected_episodes:
        print("  ✓ PASS: Intervention episodes match expected!")
    else:
        print("  ✗ FAIL: Intervention episodes do not match!")
    
    print()
    recording_env.close()
    
    return (stats['hil/intervention_steps'] == expected_intervention_steps and 
            stats['hil/intervention_count'] == expected_episodes)


def run_demo_with_visualization():
    """Run a demo showing robot state changes during intervention."""
    
    print("\n" + "=" * 70)
    print("     INTERVENTION DEMO WITH VISUALIZATION")
    print("=" * 70)
    print()
    
    # Simple intervention at steps 20-40
    intervention_schedule = [(20, 40)]
    
    env = MockRobotEnv(max_episode_steps=60)
    
    scripted_device = ScriptedInputDevice(
        config=InputDeviceConfig(action_dim=7),
        intervention_schedule=intervention_schedule,
    )
    # Set a specific intervention action: move strongly in +X direction
    scripted_device.set_intervention_action(
        np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )
    
    hil_env = HILInterventionWrapper(env, input_device=scripted_device)
    
    obs, _ = hil_env.reset()
    
    print("Watching robot position change:")
    print("- Steps 0-19:  Random policy (small movements)")
    print("- Steps 20-39: Human intervention (strong +X movement)")
    print("- Steps 40-59: Random policy again")
    print()
    print(f"{'Step':>5} | {'Mode':^15} | {'Position X':^12} | {'Action X':^10}")
    print("-" * 55)
    
    for step in range(60):
        # Small random policy action
        policy_action = np.random.uniform(-0.1, 0.1, size=7).astype(np.float32)
        
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        is_intervention = info.get("is_intervention", False)
        mode = ">>> HUMAN <<<" if is_intervention else "Policy"
        position_x = env._position[0]
        action_x = info['executed_action'][0]
        
        if step % 5 == 0:
            bar = "█" * int(abs(position_x) * 30)
            print(f"{step:5d} | {mode:^15} | {position_x:+.4f} {bar}")
    
    print("-" * 55)
    print(f"\nFinal position X: {env._position[0]:.4f}")
    print("Notice how position X increases rapidly during intervention (steps 20-39)")
    
    hil_env.close()


def run_action_override_test():
    """Test that intervention properly overrides policy actions."""
    
    print("\n" + "=" * 70)
    print("     ACTION OVERRIDE TEST")
    print("=" * 70)
    print()
    
    env = MockRobotEnv()
    
    # Intervene on all steps
    scripted_device = ScriptedInputDevice(
        config=InputDeviceConfig(action_dim=7),
        intervention_schedule=[(0, 100)],
    )
    # Human always outputs [1, 1, 1, ...]
    scripted_device.set_intervention_action(np.ones(7))
    
    hil_env = HILInterventionWrapper(env, input_device=scripted_device)
    hil_env.reset()
    
    # Policy always outputs [-1, -1, -1, ...]
    policy_action = -np.ones(7)
    
    print("Testing action override:")
    print(f"  Policy action:    {policy_action[:3]}...")
    print(f"  Human action:     {np.ones(3)}...")
    print()
    
    obs, reward, terminated, truncated, info = hil_env.step(policy_action)
    
    print(f"  Executed action:  {info['executed_action'][:3]}...")
    print()
    
    if np.allclose(info['executed_action'], np.ones(7)):
        print("  ✓ PASS: Human action correctly overrides policy action!")
    else:
        print("  ✗ FAIL: Executed action should be human action!")
        
    hil_env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test scripted intervention")
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run visualization demo"
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Run action override test"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_with_visualization()
    elif args.override:
        run_action_override_test()
    elif args.all:
        success1 = run_scripted_test()
        run_demo_with_visualization()
        run_action_override_test()
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
    else:
        # Default: run main test
        success = run_scripted_test()
        sys.exit(0 if success else 1)

