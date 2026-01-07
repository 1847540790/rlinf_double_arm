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

"""
Flexiv Dual Arm Real Robot Environment for RL Training.

This module provides a gym-compatible environment wrapper for dual Flexiv robot arms.
Supports:
- Two robot arms (left and right)
- Two wrist cameras (one per arm)
- Optional base/table camera
- 14-dimensional state (7 dims per arm: pos + rot + gripper)
- 14-dimensional actions (7 dims per arm)

Currently uses MOCK outputs for testing - replace with actual robot SDK calls for deployment.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import gym
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.utils import to_tensor


class FlexivDualArmEnv(gym.Env):
    """
    Real Flexiv Dual Arm Robot Environment for Reinforcement Learning.

    This environment interfaces with two physical Flexiv robot arms for RL training.
    Currently uses MOCK outputs for testing the training pipeline.

    Observation format:
    - left_wrist_images: Left arm wrist camera [num_envs, C, H, W]
    - right_wrist_images: Right arm wrist camera [num_envs, C, H, W]
    - base_images: Optional base/table camera [num_envs, C, H, W]
    - states: Robot state [num_envs, 14] (7 dims per arm)
    - task_descriptions: Task description strings

    Action format (14 dims):
    - Left arm: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
    - Right arm: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]

    Args:
        cfg: Environment configuration containing robot IPs, camera settings, etc.
        num_envs: Number of parallel environments (typically 1 for real robot)
        seed_offset: Random seed offset
        total_num_processes: Total number of env processes across all nodes
    """

    # Mock mode flag - set to False when using real hardware
    USE_MOCK = True

    def __init__(
        self, cfg: DictConfig, num_envs: int, seed_offset: int, total_num_processes: int
    ):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = cfg.seed + seed_offset

        # Random generator for mock data
        self._rng = np.random.default_rng(self.seed)

        # Environment settings
        self.num_envs = num_envs
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)

        # Dual arm robot configuration
        self.left_robot_ip = cfg.get("left_robot_ip", "192.168.0.109")
        self.right_robot_ip = cfg.get("right_robot_ip", "192.168.0.110")

        # Camera configuration
        self.left_wrist_camera_serial = cfg.get("left_wrist_camera_serial", "")
        self.right_wrist_camera_serial = cfg.get("right_wrist_camera_serial", "")
        self.base_camera_serial = cfg.get("base_camera_serial", "")
        self.use_base_camera = cfg.get("use_base_camera", False)

        # Image settings
        self.image_size = cfg.get("image_size", 224)

        # Action settings (14 dims: 7 per arm)
        self.action_dim = cfg.get("action_dim", 14)
        self.left_action_dim = 7  # pos(3) + rot(3) + gripper(1)
        self.right_action_dim = 7

        # Task configuration
        self.task_description = cfg.get(
            "task_description", "Perform the bimanual manipulation task"
        )
        self.max_episode_steps = cfg.max_episode_steps

        # State tracking
        self._is_start = True
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.get("use_rel_reward", False)

        # Mock state: simulated end-effector positions for both arms
        self._mock_left_ee_pos = np.zeros((self.num_envs, 3))
        self._mock_right_ee_pos = np.zeros((self.num_envs, 3))
        self._mock_target_pos = self._rng.uniform(-0.3, 0.3, size=(self.num_envs, 3))

        # Video recording
        self.video_cfg = cfg.get("video_cfg", {})
        self.video_cnt = 0
        self.render_images = []

        # Metrics tracking
        self._init_metrics()

        # Initialize robot connections (mock in test mode)
        self._init_robots()

        # Initialize cameras (mock in test mode)
        self._init_cameras()

        mode_str = "MOCK MODE" if self.USE_MOCK else "REAL HARDWARE"
        print(
            f"[FlexivDualArmEnv] Initialized in {mode_str}\n"
            f"  Left robot: {self.left_robot_ip}\n"
            f"  Right robot: {self.right_robot_ip}\n"
            f"  Left wrist camera: {self.left_wrist_camera_serial}\n"
            f"  Right wrist camera: {self.right_wrist_camera_serial}\n"
            f"  Base camera: {self.base_camera_serial if self.use_base_camera else 'disabled'}\n"
            f"  num_envs: {self.num_envs}"
        )

    def _init_robots(self):
        """Initialize connections to both Flexiv robots."""
        if self.USE_MOCK:
            print(
                f"[FlexivDualArmEnv] MOCK: Simulating robots at "
                f"{self.left_robot_ip} (left) and {self.right_robot_ip} (right)"
            )
            self.left_robot = None
            self.right_robot = None
        else:
            # TODO: Implement actual robot connections using Flexiv SDK
            # from flexiv_rdk import Robot
            # self.left_robot = Robot(self.left_robot_ip)
            # self.right_robot = Robot(self.right_robot_ip)
            # self.left_robot.connect()
            # self.right_robot.connect()
            print(
                f"[FlexivDualArmEnv] Connecting to robots at "
                f"{self.left_robot_ip} and {self.right_robot_ip}..."
            )
            self.left_robot = None
            self.right_robot = None

    def _init_cameras(self):
        """Initialize camera connections."""
        if self.USE_MOCK:
            print(
                f"[FlexivDualArmEnv] MOCK: Simulating cameras: "
                f"left_wrist={self.left_wrist_camera_serial}, "
                f"right_wrist={self.right_wrist_camera_serial}"
            )
            self.cameras = {}
        else:
            # TODO: Implement actual camera initialization
            # import pyrealsense2 as rs
            # self.cameras = {...}
            print("[FlexivDualArmEnv] Initializing cameras...")
            self.cameras = {}

    def _init_metrics(self):
        """Initialize metrics tracking."""
        self.metrics = defaultdict(list)
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.num_successes = 0
        self.num_failures = 0
        self.num_episodes = 0

    @property
    def total_num_group_envs(self) -> int:
        """Total number of environment groups (for reset state sampling)."""
        return 1

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool):
        self._is_start = value

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    def start_simulator(self):
        """Called before interaction begins."""
        print("[FlexivDualArmEnv] Starting environment...")

    def stop_simulator(self):
        """Called after interaction ends."""
        print("[FlexivDualArmEnv] Stopping environment...")

    def update_reset_state_ids(self):
        """Update reset state IDs (for compatibility)."""
        pass

    def _generate_mock_image(self, batch_size: int, arm: str = "left") -> torch.Tensor:
        """
        Generate mock RGB image observations.

        Args:
            batch_size: Number of images to generate
            arm: Which arm's camera ("left", "right", or "base")

        Returns:
            Image tensor [batch_size, 3, H, W]
        """
        images = self._rng.random((batch_size, 3, self.image_size, self.image_size))

        # Add some visual structure (gradient + noise)
        x = np.linspace(0, 1, self.image_size)
        y = np.linspace(0, 1, self.image_size)
        xx, yy = np.meshgrid(x, y)

        for i in range(batch_size):
            # Use appropriate EE position for the arm
            if arm == "left":
                ee_pos = self._mock_left_ee_pos[i]
                color_offset = [0.3, 0.2, 0.4]  # Slightly blue tint for left
            elif arm == "right":
                ee_pos = self._mock_right_ee_pos[i]
                color_offset = [0.4, 0.2, 0.3]  # Slightly red tint for right
            else:  # base
                ee_pos = (self._mock_left_ee_pos[i] + self._mock_right_ee_pos[i]) / 2
                color_offset = [0.3, 0.3, 0.3]  # Neutral for base

            ee_x, ee_y = (ee_pos[:2] + 0.5).clip(0, 1)
            gradient = np.exp(-((xx - ee_x) ** 2 + (yy - ee_y) ** 2) / 0.1)

            images[i, 0] = color_offset[0] + 0.4 * gradient + 0.3 * images[i, 0]
            images[i, 1] = color_offset[1] + 0.3 * gradient + 0.5 * images[i, 1]
            images[i, 2] = color_offset[2] + 0.2 * gradient + 0.4 * images[i, 2]

        images = np.clip(images, 0, 1).astype(np.float32)
        return torch.from_numpy(images)

    def _generate_mock_state(self, batch_size: int) -> torch.Tensor:
        """
        Generate mock robot proprioceptive state for dual arm.

        State format (14 dims):
        - Left arm: [ee_pos(3), ee_rot(3), gripper(1)] = 7 dims
        - Right arm: [ee_pos(3), ee_rot(3), gripper(1)] = 7 dims

        Returns:
            State tensor [batch_size, 14]
        """
        states = np.zeros((batch_size, 14), dtype=np.float32)

        # Left arm state (dims 0-6)
        states[:, 0:3] = self._mock_left_ee_pos  # Position
        states[:, 3:6] = self._rng.normal(0, 0.01, size=(batch_size, 3))  # Rotation (axis-angle)
        states[:, 6] = 0.5  # Gripper (0=closed, 1=open)

        # Right arm state (dims 7-13)
        states[:, 7:10] = self._mock_right_ee_pos  # Position
        states[:, 10:13] = self._rng.normal(0, 0.01, size=(batch_size, 3))  # Rotation
        states[:, 13] = 0.5  # Gripper

        return torch.from_numpy(states)

    def get_observation(self) -> dict:
        """
        Get current observation from robots and cameras.

        Returns:
            dict containing:
                - left_wrist_images: Left wrist camera [num_envs, C, H, W]
                - right_wrist_images: Right wrist camera [num_envs, C, H, W]
                - base_images: Base camera (optional) [num_envs, C, H, W]
                - states: Robot state [num_envs, 14]
                - task_descriptions: Task description strings
        """
        if self.USE_MOCK:
            # Generate mock observations
            left_wrist_images = self._generate_mock_image(self.num_envs, arm="left")
            right_wrist_images = self._generate_mock_image(self.num_envs, arm="right")
            states = self._generate_mock_state(self.num_envs)

            obs = {
                "left_wrist_images": left_wrist_images,
                "right_wrist_images": right_wrist_images,
                "states": states,
                "task_descriptions": [self.task_description] * self.num_envs,
            }

            # Add base camera if configured
            if self.use_base_camera:
                obs["base_images"] = self._generate_mock_image(self.num_envs, arm="base")
            else:
                obs["base_images"] = None
        else:
            # TODO: Implement actual observation collection from hardware
            obs = {
                "left_wrist_images": torch.zeros(
                    self.num_envs, 3, self.image_size, self.image_size
                ),
                "right_wrist_images": torch.zeros(
                    self.num_envs, 3, self.image_size, self.image_size
                ),
                "states": torch.zeros(self.num_envs, 14),
                "task_descriptions": [self.task_description] * self.num_envs,
                "base_images": None,
            }

        return obs

    def reset(self, env_ids: Optional[np.ndarray] = None) -> tuple[dict, dict]:
        """
        Reset the environment.

        Args:
            env_ids: Optional array of environment indices to reset

        Returns:
            Tuple of (observations, infos)
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        print(f"[FlexivDualArmEnv] Resetting environments: {env_ids}")

        # Reset state tracking
        for env_id in env_ids:
            self._elapsed_steps[env_id] = 0
            self.prev_step_reward[env_id] = 0.0
            self.episode_returns[env_id] = 0.0
            self.episode_lengths[env_id] = 0

            # Reset mock state
            if self.USE_MOCK:
                self._mock_left_ee_pos[env_id] = np.array([-0.2, 0.0, 0.0])
                self._mock_right_ee_pos[env_id] = np.array([0.2, 0.0, 0.0])
                self._mock_target_pos[env_id] = self._rng.uniform(-0.2, 0.2, size=3)

        self._is_start = True

        # Get initial observations
        obs = self.get_observation()
        infos = {"reset_env_ids": env_ids}

        return obs, infos

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute actions in the environment.

        Args:
            actions: Action tensor [num_envs, 14]
                - dims 0-6: left arm actions
                - dims 7-13: right arm actions

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else actions

        # Execute action
        self._execute_action(actions_np)

        # Update step counter
        self._elapsed_steps += 1

        # Get new observation
        obs = self.get_observation()

        # Compute reward
        rewards = self._compute_reward(obs, actions_np)

        # Check termination (success)
        terminations = self._check_termination(obs)

        # Check truncation (max steps reached)
        truncations = self._elapsed_steps >= self.max_episode_steps

        # Update metrics
        self.episode_returns += rewards
        self.episode_lengths += 1

        # Handle auto reset
        done_mask = terminations | truncations
        infos = {
            "success": terminations.copy(),
            "elapsed_steps": self._elapsed_steps.copy(),
        }

        if self.auto_reset and done_mask.any():
            reset_ids = np.where(done_mask)[0]
            self._handle_episode_end(reset_ids, terminations[reset_ids])
            obs, _ = self.reset(reset_ids)

        # Convert to tensors
        rewards_tensor = to_tensor(rewards)
        terminations_tensor = to_tensor(terminations)
        truncations_tensor = to_tensor(truncations)

        self._is_start = False

        return obs, rewards_tensor, terminations_tensor, truncations_tensor, infos

    def chunk_step(
        self, chunk_actions: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute a chunk of actions.

        Args:
            chunk_actions: Actions for multiple steps [num_envs, chunk_size, 14]

        Returns:
            Same as step() but aggregated over chunk
        """
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []

        for i in range(chunk_size):
            obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, i]
            )
            chunk_rewards.append(rewards)
            chunk_terminations.append(terminations)
            chunk_truncations.append(truncations)

        # Stack results
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(chunk_terminations, dim=1)
        chunk_truncations = torch.stack(chunk_truncations, dim=1)

        return obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    def _execute_action(self, actions: np.ndarray):
        """
        Send actions to both robot arms.

        Args:
            actions: Actions to execute [num_envs, 14]
                - dims 0-6: left arm [delta_pos(3), delta_rot(3), gripper(1)]
                - dims 7-13: right arm [delta_pos(3), delta_rot(3), gripper(1)]
        """
        if self.USE_MOCK:
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Left arm position update (scale down for mock)
            left_delta_pos = actions[:, :3] * 0.01
            self._mock_left_ee_pos += left_delta_pos

            # Right arm position update
            right_delta_pos = actions[:, 7:10] * 0.01
            self._mock_right_ee_pos += right_delta_pos

            # Clip to workspace bounds
            self._mock_left_ee_pos = np.clip(self._mock_left_ee_pos, -0.5, 0.5)
            self._mock_right_ee_pos = np.clip(self._mock_right_ee_pos, -0.5, 0.5)

            # Small delay to simulate control loop
            time.sleep(0.01)  # 100 Hz mock control
        else:
            # TODO: Implement actual action execution for both arms
            # left_delta_pos = actions[:, :3]
            # left_delta_rot = actions[:, 3:6]
            # left_gripper = actions[:, 6]
            # right_delta_pos = actions[:, 7:10]
            # right_delta_rot = actions[:, 10:13]
            # right_gripper = actions[:, 13]
            #
            # self.left_robot.move_ee_delta(left_delta_pos, left_delta_rot)
            # self.left_robot.set_gripper(left_gripper)
            # self.right_robot.move_ee_delta(right_delta_pos, right_delta_rot)
            # self.right_robot.set_gripper(right_gripper)
            time.sleep(0.05)  # 20 Hz control

    def _compute_reward(self, obs: dict, actions: np.ndarray) -> np.ndarray:
        """
        Compute reward for the current step.

        For bimanual tasks, reward can be based on:
        - Distance of both EEs to target
        - Coordination between arms
        - Task-specific metrics

        Args:
            obs: Current observations
            actions: Executed actions

        Returns:
            Reward array [num_envs]
        """
        if self.USE_MOCK:
            # Mock reward: both arms need to approach the target
            left_distance = np.linalg.norm(
                self._mock_left_ee_pos - self._mock_target_pos, axis=1
            )
            right_distance = np.linalg.norm(
                self._mock_right_ee_pos - self._mock_target_pos, axis=1
            )

            # Combined distance reward
            avg_distance = (left_distance + right_distance) / 2
            distance_reward = -avg_distance

            # Action penalty (encourage smooth actions)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            action_penalty = -0.01 * np.linalg.norm(actions, axis=1)

            # Success bonus (both arms close to target)
            both_close = (left_distance < 0.1) & (right_distance < 0.1)
            success_bonus = np.where(both_close, 10.0, 0.0)

            rewards = distance_reward + action_penalty + success_bonus

            return rewards.astype(np.float32)
        else:
            # TODO: Implement task-specific reward
            return np.zeros(self.num_envs, dtype=np.float32)

    def _check_termination(self, obs: dict) -> np.ndarray:
        """
        Check if episode should terminate (success).

        Args:
            obs: Current observations

        Returns:
            Boolean array [num_envs] indicating termination
        """
        if self.USE_MOCK:
            # Mock termination: both arms close to target
            left_distance = np.linalg.norm(
                self._mock_left_ee_pos - self._mock_target_pos, axis=1
            )
            right_distance = np.linalg.norm(
                self._mock_right_ee_pos - self._mock_target_pos, axis=1
            )
            terminations = (left_distance < 0.1) & (right_distance < 0.1)
            return terminations
        else:
            # TODO: Implement task-specific termination
            return np.zeros(self.num_envs, dtype=bool)

    def _handle_episode_end(self, env_ids: np.ndarray, successes: np.ndarray):
        """Handle end of episode for logging."""
        for i, env_id in enumerate(env_ids):
            self.num_episodes += 1
            if successes[i]:
                self.num_successes += 1
            else:
                self.num_failures += 1

            self.metrics["episode_return"].append(self.episode_returns[env_id])
            self.metrics["episode_length"].append(self.episode_lengths[env_id])

    def get_metrics(self) -> dict:
        """Get current metrics."""
        if self.num_episodes == 0:
            return {}

        return {
            "success_rate": self.num_successes / max(1, self.num_episodes),
            "num_episodes": self.num_episodes,
            "avg_return": np.mean(self.metrics["episode_return"])
            if self.metrics["episode_return"]
            else 0,
            "avg_length": np.mean(self.metrics["episode_length"])
            if self.metrics["episode_length"]
            else 0,
        }

    def close(self):
        """Clean up resources."""
        print("[FlexivDualArmEnv] Closing environment...")
        if not self.USE_MOCK:
            # TODO: Disconnect robots and cameras
            pass

