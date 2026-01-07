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

"""Example training script for RLPD (Reinforcement Learning with Prior Data).

This script demonstrates how to train a robot policy using Human-in-the-Loop
RLPD with the RLinf framework. RLPD combines online RL with prior demonstration
data using a 50/50 sampling strategy.

Usage:
    # Train with mock environment (for testing)
    python train_rlpd.py --config config/rlpd_mock.yaml
    
    # Train with real robot (when configured)
    python train_rlpd.py --config config/rlpd_real_robot.yaml
    
    # Resume from checkpoint
    python train_rlpd.py --config config/rlpd_mock.yaml --resume checkpoint.pt
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add rlinf to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlinf.runners.hil_sac_runner import HILSACRunner, HILSACConfig


class MockRobotEnv(gym.Env):
    """Mock robot environment for testing RLPD pipeline.
    
    This simulates a simple reaching task where the robot needs to move
    to a target position. Human interventions are simulated.
    """
    
    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 7,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        
        # Observation space: tcp_pose (7) + tcp_vel (6) + gripper (1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: xyz delta + rpy delta + gripper
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        self._step_count = 0
        self._state = None
        self._target = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._step_count = 0
        
        # Random initial state
        self._state = np.zeros(self.obs_dim, dtype=np.float32)
        self._state[:3] = np.random.uniform(-0.1, 0.1, size=3)  # Position
        
        # Random target
        self._target = np.random.uniform(-0.2, 0.2, size=3)
        
        return self._get_obs(), {}
        
    def step(self, action):
        self._step_count += 1
        
        # Simple dynamics: action directly affects position
        self._state[:3] += action[:3] * 0.02
        
        # Compute reward (distance to target)
        dist = np.linalg.norm(self._state[:3] - self._target)
        reward = -dist
        
        # Check success
        success = dist < 0.05
        if success:
            reward = 10.0
            
        # Episode termination
        terminated = success
        truncated = self._step_count >= self.max_episode_steps
        
        info = {"succeed": success}
        
        return self._get_obs(), reward, terminated, truncated, info
        
    def _get_obs(self):
        return self._state.copy()


def create_environment(cfg: DictConfig) -> gym.Env:
    """Create environment based on configuration.
    
    Args:
        cfg: Environment configuration
        
    Returns:
        Gym environment
    """
    simulator_type = cfg.get("simulator_type", "mock")
    
    if simulator_type == "mock":
        return MockRobotEnv(
            obs_dim=cfg.get("obs_dim", 14),
            action_dim=cfg.get("action_dim", 7),
            max_episode_steps=cfg.get("max_steps_per_episode", 200),
        )
    else:
        # Add support for real robot environments here
        raise NotImplementedError(f"Unknown simulator type: {simulator_type}")


def main():
    parser = argparse.ArgumentParser(description="RLPD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rlpd_mock.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--demo_paths",
        nargs="*",
        default=None,
        help="Paths to demonstration files",
    )
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        cfg = OmegaConf.create({
            "rlpd": {},
            "env": {"simulator_type": "mock", "obs_dim": 14, "action_dim": 7},
        })
        
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create environment
    env = create_environment(cfg.get("env", {}))
    print(f"Created environment: {type(env).__name__}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Create RLPD config
    rlpd_cfg = cfg.get("rlpd", {})
    rlpd_config = HILSACConfig(
        max_episode_steps=rlpd_cfg.get("max_episode_steps", 200),
        replay_buffer_capacity=rlpd_cfg.get("replay_buffer_capacity", 50000),
        demo_buffer_capacity=rlpd_cfg.get("demo_buffer_capacity", 10000),
        min_buffer_size=rlpd_cfg.get("min_buffer_size", 500),
        batch_size=rlpd_cfg.get("batch_size", 128),
        learning_rate=rlpd_cfg.get("learning_rate", 3e-4),
        gamma=rlpd_cfg.get("gamma", 0.99),
        tau=rlpd_cfg.get("tau", 0.005),
        init_temperature=rlpd_cfg.get("init_temperature", 1.0),
        utd_ratio=rlpd_cfg.get("utd_ratio", 4),
        demo_ratio=rlpd_cfg.get("demo_ratio", 0.5),
        hidden_dims=list(rlpd_cfg.get("hidden_dims", [256, 256])),
        encoder_type=rlpd_cfg.get("encoder_type", "mlp"),
        image_keys=tuple(rlpd_cfg.get("image_keys", [])),
        use_proprio=rlpd_cfg.get("use_proprio", True),
        input_device=rlpd_cfg.get("input_device", "mock"),
        log_interval=rlpd_cfg.get("log_interval", 10),
        save_interval=rlpd_cfg.get("save_interval", 1000),
        checkpoint_dir=rlpd_cfg.get("checkpoint_dir", "./checkpoints"),
        buffer_dir=rlpd_cfg.get("buffer_dir", "./buffers"),
    )
    
    # Create runner (RLPD uses the HIL-SAC runner internally)
    runner = HILSACRunner(
        env=env,
        config=rlpd_config,
        demo_paths=args.demo_paths,
        resume_path=args.resume,
    )
    
    print("\nStarting RLPD training...")
    print(f"  Input device: {rlpd_config.input_device}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  UTD ratio: {rlpd_config.utd_ratio}")
    print(f"  Demo ratio: {rlpd_config.demo_ratio}")
    
    # Train
    try:
        stats = runner.train(max_steps=args.max_steps)
        print(f"\nTraining completed!")
        print(f"  Final step: {stats['final_step']}")
        print(f"  Episodes: {stats['episodes']}")
        
        # Evaluate
        print("\nRunning evaluation...")
        eval_stats = runner.evaluate(num_episodes=10)
        print(f"  Mean return: {eval_stats['eval/mean_return']:.2f}")
        print(f"  Success rate: {eval_stats['eval/success_rate']:.2%}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        
    finally:
        env.close()
        

if __name__ == "__main__":
    main()

