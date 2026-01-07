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
NmxRdk Real Robot Environment for RL Training.

This module provides a gym-compatible environment wrapper for NmxRdk robots.
Based on the NmxRdkRobot SDK.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import gym
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from rlinf.envs.utils import to_tensor
from .pose import pose_to_4x4, pose_to_7D


class NmxRdkRealEnv(gym.Env):
    """
    Real NmxRdk Robot Environment for Reinforcement Learning.

    This environment interfaces with a physical NmxRdk robot for RL training.

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
        self.action_scale = cfg.get("action_scale", 1.0)

        # Robot control parameters
        self.control_freq = cfg.get("control_freq", 20)  # Hz
        self.max_tcp_vel = cfg.get("max_tcp_vel", 0.05)  # m/s
        self.max_tcp_acc = cfg.get("max_tcp_acc", 0.5)  # m/s^2

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

        # Initialize robot connection
        self._init_robot()

        # Initialize cameras
        self._init_cameras()

        mode_str = "MOCK MODE" if self.USE_MOCK else "REAL HARDWARE"
        print(
            f"[NmxRdkRealEnv] Initialized in {mode_str} with robot_ip={self.robot_ip}, "
            f"cameras={self.camera_serials}, num_envs={self.num_envs}"
        )

    def _init_robot(self):
        """Initialize connection to the NmxRdk robot."""
        if self.USE_MOCK:
            print(f"[NmxRdkRealEnv] MOCK: Simulating robot at {self.robot_ip}")
            self.robot = None
            self.controller = None
        else:
            try:
                from .nmxrdk_robot import NmxRdkRobot

                print(f"[NmxRdkRealEnv] Connecting to robot at {self.robot_ip}...")

                # Initialize controller - this should be customized based on your SDK setup
                # The controller is expected to be a list with arm control interface
                self.controller = self._create_controller()

                # Create NmxRdkRobot instance
                config = {"robot_ip": self.robot_ip}
                self.robot = NmxRdkRobot(config, self.controller)

                # Enable robot
                self.robot.enable()
                print(f"[NmxRdkRealEnv] Robot connected and enabled.")

            except Exception as e:
                print(f"[NmxRdkRealEnv] Failed to connect to robot: {e}")
                print(f"[NmxRdkRealEnv] Falling back to MOCK mode.")
                self.USE_MOCK = True
                self.robot = None
                self.controller = None

    def _create_controller(self):
        """
        Create the robot controller.

        This method should be overridden or customized based on your actual SDK setup.
        The controller should be a list containing dict with 'arm' key.

        Returns:
            list: Controller list with arm interface
        """
        # TODO: Implement actual controller creation based on your SDK
        # Example:
        # from your_sdk import ArmController
        # arm = ArmController(self.robot_ip)
        # return [{"arm": arm}]
        raise NotImplementedError(
            "Please implement _create_controller() for your specific robot SDK setup."
        )

    def _init_cameras(self):
        """Initialize camera connections."""
        if self.USE_MOCK:
            print(f"[NmxRdkRealEnv] MOCK: Simulating cameras: {self.camera_serials}")
            self.cameras = []
        else:
            # TODO: Implement actual camera initialization
            # Example with pyrealsense2:
            # import pyrealsense2 as rs
            # self.cameras = []
            # for serial in self.camera_serials:
            #     pipeline = rs.pipeline()
            #     config = rs.config()
            #     config.enable_device(serial)
            #     config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            #     pipeline.start(config)
            #     self.cameras.append(pipeline)
            print(f"[NmxRdkRealEnv] Initializing cameras: {self.camera_serials}")
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
        print("[NmxRdkRealEnv] Starting environment...")

    def stop_simulator(self):
        """Called after interaction ends."""
        print("[NmxRdkRealEnv] Stopping environment...")
        if not self.USE_MOCK and self.robot is not None:
            self.robot.stop()

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

    def _get_real_image(self, batch_size: int) -> torch.Tensor:
        """Get real RGB image from cameras."""
        # TODO: Implement actual camera image capture
        # Example:
        # images = []
        # for camera in self.cameras:
        #     frames = camera.wait_for_frames()
        #     color_frame = frames.get_color_frame()
        #     image = np.asanyarray(color_frame.get_data())
        #     image = cv2.resize(image, (self.image_size, self.image_size))
        #     image = image.transpose(2, 0, 1) / 255.0  # CHW format
        #     images.append(image)
        # return torch.from_numpy(np.stack(images)).float()
        return torch.zeros(batch_size, 3, self.image_size, self.image_size)

    def _generate_mock_state(self, batch_size: int) -> torch.Tensor:
        """Generate mock robot proprioceptive state."""
        # State: [ee_pos (3), ee_quat (4)] = 7 dims
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

    def _get_real_state(self, batch_size: int) -> torch.Tensor:
        """Get real robot proprioceptive state."""
        if self.robot is None:
            return self._generate_mock_state(batch_size)

        # Get TCP pose from robot (6D: x, y, z, rx, ry, rz in Euler angles)
        tcp_pose_6d = self.robot.get_tcp_pose()

        # Convert to 7D (x, y, z, qw, qx, qy, qz)
        tcp_pose_7d = pose_to_7D(tcp_pose_6d)

        states = np.zeros((batch_size, 7), dtype=np.float32)
        states[0, :] = tcp_pose_7d

        # For batch_size > 1, repeat the state (real robot only has 1 env)
        for i in range(1, batch_size):
            states[i, :] = tcp_pose_7d

        return torch.from_numpy(states)

    def get_tcp_pose(self) -> np.ndarray:
        """Get current TCP pose from robot."""
        if self.USE_MOCK:
            # Return mock pose
            pose = np.zeros(7, dtype=np.float32)
            pose[:3] = self._mock_ee_pos[0]
            pose[3] = 1.0  # qw
            return pose
        else:
            return pose_to_7D(np.array(self.robot.get_tcp_pose()))

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions from robot."""
        if self.USE_MOCK:
            return np.zeros(6, dtype=np.float32)
        else:
            return self.robot.get_joint_pos()

    def get_tcp_force(self) -> np.ndarray:
        """Get current TCP force/torque from robot."""
        if self.USE_MOCK:
            return np.zeros(6, dtype=np.float32)
        else:
            return self.robot.get_tcp_force_value()

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
            wrist_images = (
                self._generate_mock_image(self.num_envs)
                if self.use_wrist_image
                else None
            )
        else:
            # Get real observations from hardware
            images = self._get_real_image(self.num_envs)
            states = self._get_real_state(self.num_envs)
            wrist_images = (
                self._get_real_image(self.num_envs) if self.use_wrist_image else None
            )

        # Format for model compatibility
        obs = {
            "images": images,
            "states": states,
            "task_descriptions": [self.task_description] * self.num_envs,
            "wrist_images": wrist_images,
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

        print(f"[NmxRdkRealEnv] Resetting environments: {env_ids}")

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

        # Move robot to home position (if not mock)
        if not self.USE_MOCK and self.robot is not None:
            self._move_to_home()

        self._is_start = True

        # Get initial observations
        obs = self.get_observation()
        infos = {"reset_env_ids": env_ids}

        return obs, infos

    def _move_to_home(self):
        """Move robot to home position."""
        if self.robot is None:
            return

        # Define home joint positions (customize for your robot)
        home_joint_pos = [0.0, -30.0, 0.0, 90.0, 0.0, 40.0]  # Example in degrees

        try:
            self.robot.move_arm_joint_offline(
                target_jnt_pos=home_joint_pos,
                target_jnt_vel=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                target_jnt_acc=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            )
            print("[NmxRdkRealEnv] Robot moved to home position.")
        except Exception as e:
            print(f"[NmxRdkRealEnv] Failed to move to home: {e}")

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
                     Format: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        """
        if self.USE_MOCK:
            # Mock: Update simulated end-effector position based on action
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Apply position delta (scaled)
            delta_pos = actions[:, :3] * 0.01 * self.action_scale
            self._mock_ee_pos += delta_pos

            # Clip to workspace bounds
            self._mock_ee_pos = np.clip(self._mock_ee_pos, -0.5, 0.5)

            # Small delay to simulate control loop
            time.sleep(1.0 / self.control_freq)
        else:
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Get current TCP pose
            current_pose = self.robot.get_tcp_pose()  # [x, y, z, rx, ry, rz]

            # Extract action components
            delta_pos = actions[0, :3] * self.action_scale  # Position delta
            delta_rot = (
                actions[0, 3:6] * self.action_scale
            )  # Rotation delta (axis-angle or Euler)
            gripper_action = actions[0, 6] if actions.shape[1] > 6 else 0.0

            # Compute target pose
            target_pose = np.zeros(6, dtype=np.float32)
            target_pose[:3] = current_pose[:3] + delta_pos  # Apply position delta
            target_pose[3:6] = current_pose[3:6] + delta_rot  # Apply rotation delta

            # Execute motion using move_line_online for smooth Cartesian control
            self.robot.move_line_online(
                target=target_pose,
                target_vel=self.max_tcp_vel,
                target_acc=self.max_tcp_acc,
                follow=True,
            )

            # Control loop timing
            time.sleep(1.0 / self.control_freq)

    def move_to_pose(self, target_pose: np.ndarray, blocking: bool = True):
        """
        Move robot to target pose.

        Args:
            target_pose: Target pose [x, y, z, rx, ry, rz] or [x, y, z, qw, qx, qy, qz]
            blocking: Whether to wait for motion to complete
        """
        if self.USE_MOCK:
            print(f"[NmxRdkRealEnv] MOCK: Moving to pose {target_pose}")
            return

        if len(target_pose) == 7:
            # Convert from 7D (quaternion) to 6D (Euler)
            xyz = target_pose[:3]
            quat_wxyz = target_pose[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            euler = R.from_quat(quat_xyzw).as_euler("xyz")
            target_pose = np.concatenate([xyz, euler])

        self.robot.move_line_offline(
            target=target_pose,
            max_v=self.max_tcp_vel,
            max_a=self.max_tcp_acc,
            block=blocking,
        )

    def move_joints(self, target_joints: np.ndarray, blocking: bool = True):
        """
        Move robot to target joint positions.

        Args:
            target_joints: Target joint positions in degrees
            blocking: Whether to wait for motion to complete
        """
        if self.USE_MOCK:
            print(f"[NmxRdkRealEnv] MOCK: Moving joints to {target_joints}")
            return

        if blocking:
            self.robot.move_arm_joint_offline(target_jnt_pos=target_joints)
        else:
            self.robot.move_arm_joint_online(target_jnt_pos=target_joints)

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
            # TODO: Implement task-specific reward for real robot
            # This could be based on:
            # - Distance to target position
            # - Force/torque feedback
            # - Visual feedback (object detection)
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
            # TODO: Implement task-specific termination for real robot
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

    def calibrate_force_sensor(self):
        """Calibrate the force sensor."""
        if self.USE_MOCK:
            print("[NmxRdkRealEnv] MOCK: Calibrating force sensor")
            return
        self.robot.cali_force_sensor()

    def enable_robot(self):
        """Enable the robot."""
        if self.USE_MOCK:
            print("[NmxRdkRealEnv] MOCK: Enabling robot")
            return
        self.robot.enable()

    def stop_robot(self):
        """Stop the robot."""
        if self.USE_MOCK:
            print("[NmxRdkRealEnv] MOCK: Stopping robot")
            return
        self.robot.stop()

    def close(self):
        """Clean up resources."""
        print("[NmxRdkRealEnv] Closing environment...")
        if not self.USE_MOCK:
            if self.robot is not None:
                self.robot.stop()
            # Close cameras
            for camera in self.cameras:
                try:
                    camera.stop()
                except Exception:
                    pass
