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
Flexiv Real Robot Environment for RL Training.

This module provides a gym-compatible environment wrapper for real Flexiv robots.
Currently uses MOCK outputs for testing - replace with actual robot SDK calls for real deployment.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import gym
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.utils import to_tensor


class FlexivRealEnv(gym.Env):
    """
    Real Flexiv Robot Environment for Reinforcement Learning.

    This environment interfaces with a physical Flexiv robot for RL training.
    Currently uses MOCK outputs for testing the training pipeline.

    Args:
        cfg: Environment configuration containing robot IP, camera settings, etc.
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

        # Robot hardware configuration
        self.robot_ip = cfg.get("robot_ip", "192.168.0.109")
        self.camera_serials = cfg.get("camera_serials", [])

        # Image settings
        self.image_size = cfg.get("image_size", 256)
        self.use_wrist_image = (
            cfg.get("use_wrist_image", True) and len(self.camera_serials) > 1
        )

        # Action settings
        self.action_dim = cfg.get("action_dim", 7)  # 3 pos + 3 rot + 1 gripper

        # Task configuration
        self.task_description = cfg.get(
            "task_description", "Perform the manipulation task"
        )
        self.max_episode_steps = cfg.max_episode_steps

        # State tracking
        self._is_start = True
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.get("use_rel_reward", False)

        # Mock state: simulated end-effector position for reward computation
        self._mock_ee_pos = np.zeros((self.num_envs, 3))
        self._mock_target_pos = self._rng.uniform(-0.3, 0.3, size=(self.num_envs, 3))

        # Video recording
        self.video_cfg = cfg.get("video_cfg", {})
        self.video_cnt = 0
        self.render_images = []

        # Metrics tracking
        self._init_metrics()

        # Initialize robot connection (mock in test mode)
        self._init_robot()

        # Initialize cameras (mock in test mode)
        self._init_cameras()

        mode_str = "MOCK MODE" if self.USE_MOCK else "REAL HARDWARE"
        print(
            f"[FlexivRealEnv] Initialized in {mode_str} with robot_ip={self.robot_ip}, "
            f"cameras={self.camera_serials}, num_envs={self.num_envs}"
        )

    def _init_robot(self):
        """Initialize connection to the Flexiv robot."""
        if self.USE_MOCK:
            print(f"[FlexivRealEnv] MOCK: Simulating robot at {self.robot_ip}")
            self.robot = None
        else:
            # TODO: Implement actual robot connection using Flexiv SDK
            # from flexiv_rdk import Robot
            # self.robot = Robot(self.robot_ip)
            # self.robot.connect()
            print(f"[FlexivRealEnv] Connecting to robot at {self.robot_ip}...")
            self.robot = None

    def _init_cameras(self):
        """Initialize camera connections."""
        if self.USE_MOCK:
            print(f"[FlexivRealEnv] MOCK: Simulating cameras: {self.camera_serials}")
            self.cameras = []
        else:
            # TODO: Implement actual camera initialization
            # import pyrealsense2 as rs
            # self.cameras = [...]
            print(f"[FlexivRealEnv] Initializing cameras: {self.camera_serials}")
            self.cameras = []

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
        print("[FlexivRealEnv] Starting environment...")

    def stop_simulator(self):
        """Called after interaction ends."""
        print("[FlexivRealEnv] Stopping environment...")

    def update_reset_state_ids(self):
        """Update reset state IDs (for compatibility)."""
        pass

    def _generate_mock_image(self, batch_size: int) -> torch.Tensor:
        """Generate mock RGB image observations."""
        # Generate random noise image with some structure
        # Shape: [batch_size, 3, H, W]
        images = self._rng.random((batch_size, 3, self.image_size, self.image_size))

        # Add some visual structure (gradient + noise)
        x = np.linspace(0, 1, self.image_size)
        y = np.linspace(0, 1, self.image_size)
        xx, yy = np.meshgrid(x, y)

        for i in range(batch_size):
            # Add gradient based on mock EE position
            ee_x, ee_y = (self._mock_ee_pos[i, :2] + 0.5).clip(0, 1)
            gradient = np.exp(-((xx - ee_x) ** 2 + (yy - ee_y) ** 2) / 0.1)
            images[i, 0] = 0.3 + 0.4 * gradient + 0.3 * images[i, 0]  # Red channel
            images[i, 1] = 0.2 + 0.3 * gradient + 0.5 * images[i, 1]  # Green channel
            images[i, 2] = 0.4 + 0.2 * gradient + 0.4 * images[i, 2]  # Blue channel

        images = np.clip(images, 0, 1).astype(np.float32)
        return torch.from_numpy(images)

    def _generate_mock_state(self, batch_size: int) -> torch.Tensor:
        """Generate mock robot proprioceptive state."""
        # State: [ee_pos (3), ee_quat (4)] = 7 dims
        # Or could be joint positions, etc.
        states = np.zeros((batch_size, 7), dtype=np.float32)

        # EE position (normalized)
        states[:, :3] = self._mock_ee_pos

        # EE quaternion (identity with small noise)
        states[:, 3] = 1.0  # w
        states[:, 4:7] = self._rng.normal(0, 0.01, size=(batch_size, 3))  # x, y, z

        # Normalize quaternion
        quat_norm = np.linalg.norm(states[:, 3:7], axis=1, keepdims=True)
        states[:, 3:7] /= quat_norm

        return torch.from_numpy(states)

    def get_observation(self) -> dict:
        """
        Get current observation from robot and cameras.

        Returns:
            dict containing:
                - images: Camera RGB images [num_envs, C, H, W]
                - wrist_images: Wrist camera images (optional)
                - states: Robot proprioceptive state
                - task_descriptions: Task description strings
        """
        if self.USE_MOCK:
            # Generate mock observations
            images = self._generate_mock_image(self.num_envs)
            states = self._generate_mock_state(self.num_envs)

            obs = {
                "images": images,
                "states": states,
                "task_descriptions": [self.task_description] * self.num_envs,
            }

            # Add wrist image if configured
            if self.use_wrist_image:
                obs["wrist_images"] = self._generate_mock_image(self.num_envs)
            else:
                obs["wrist_images"] = None
        else:
            # TODO: Implement actual observation collection from hardware
            obs = {
                "images": torch.zeros(
                    self.num_envs, 3, self.image_size, self.image_size
                ),
                "states": torch.zeros(self.num_envs, 7),
                "task_descriptions": [self.task_description] * self.num_envs,
                "wrist_images": None,
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

        print(f"[FlexivRealEnv] Resetting environments: {env_ids}")

        # Reset state tracking
        for env_id in env_ids:
            self._elapsed_steps[env_id] = 0
            self.prev_step_reward[env_id] = 0.0
            self.episode_returns[env_id] = 0.0
            self.episode_lengths[env_id] = 0

            # Reset mock state
            if self.USE_MOCK:
                self._mock_ee_pos[env_id] = np.zeros(3)
                self._mock_target_pos[env_id] = self._rng.uniform(-0.3, 0.3, size=3)

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
            actions: Action tensor [num_envs, action_dim]

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
            chunk_actions: Actions for multiple steps [num_envs, chunk_size, action_dim]

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
        Send action to the robot.

        Args:
            actions: Actions to execute [num_envs, action_dim]
        """
        if self.USE_MOCK:
            # Mock: Update simulated end-effector position based on action
            # Actions assumed to be: [delta_x, delta_y, delta_z, rot_x, rot_y, rot_z, gripper]
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Apply position delta (scaled down for mock)
            delta_pos = actions[:, :3] * 0.01  # Scale factor
            self._mock_ee_pos += delta_pos

            # Clip to workspace bounds
            self._mock_ee_pos = np.clip(self._mock_ee_pos, -0.5, 0.5)

            # Small delay to simulate control loop
            time.sleep(0.01)  # 100 Hz mock control
        else:
            # TODO: Implement actual action execution
            # delta_pos = actions[:, :3]
            # delta_rot = actions[:, 3:6]
            # gripper = actions[:, 6]
            # self.robot.move_ee_delta(delta_pos, delta_rot)
            # self.robot.set_gripper(gripper)
            time.sleep(0.05)  # 20 Hz control

    def _compute_reward(self, obs: dict, actions: np.ndarray) -> np.ndarray:
        """
        Compute reward for the current step.

        Args:
            obs: Current observations
            actions: Executed actions

        Returns:
            Reward array [num_envs]
        """
        if self.USE_MOCK:
            # Mock reward: negative distance to target + small action penalty
            distance = np.linalg.norm(self._mock_ee_pos - self._mock_target_pos, axis=1)

            # Dense reward: closer to target = higher reward
            distance_reward = -distance

            # Action penalty (encourage smooth actions)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            action_penalty = -0.01 * np.linalg.norm(actions, axis=1)

            # Success bonus
            success_bonus = np.where(distance < 0.05, 10.0, 0.0)

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
            # Mock termination: success if EE is close enough to target
            distance = np.linalg.norm(self._mock_ee_pos - self._mock_target_pos, axis=1)
            terminations = distance < 0.05  # Success threshold
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
        print("[FlexivRealEnv] Closing environment...")
        if not self.USE_MOCK:
            # TODO: Disconnect robot and cameras
            pass
