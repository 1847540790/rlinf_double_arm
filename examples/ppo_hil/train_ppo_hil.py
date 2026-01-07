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

"""Training script for PPO with optional Human-in-the-Loop (HIL).

This script provides a unified interface to run:
1. Standard PPO: python train_ppo_hil.py --config config/ppo_standard.yaml
2. PPO + HIL: python train_ppo_hil.py --config config/ppo_hil.yaml

The HIL mode enables human intervention during training, allowing a human
operator to override the policy's actions when needed. This is useful for:
- Correcting policy mistakes in real-time
- Providing demonstrations for difficult situations
- Guiding exploration in sparse reward environments

Usage:
    # Standard PPO (no human intervention)
    python train_ppo_hil.py --config config/ppo_standard.yaml
    
    # PPO with HIL (keyboard control)
    python train_ppo_hil.py --config config/ppo_hil.yaml
    
    # Resume from checkpoint
    python train_ppo_hil.py --config config/ppo_hil.yaml --resume checkpoint.pt
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal

# Add rlinf to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlinf.runners.embodied_hil_runner import EmbodiedHILRunner, EmbodiedHILConfig


# ============================================
# Mock Environment for Testing
# ============================================

class MockRobotEnv(gym.Env):
    """Mock robot environment for testing PPO/HIL pipeline."""
    
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
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        self._step_count = 0
        self._state = None
        self._target = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._state = np.zeros(self.obs_dim, dtype=np.float32)
        self._state[:3] = np.random.uniform(-0.1, 0.1, size=3)
        self._target = np.random.uniform(-0.2, 0.2, size=3)
        return self._get_obs(), {}
        
    def step(self, action):
        self._step_count += 1
        self._state[:3] += action[:3] * 0.02
        
        dist = np.linalg.norm(self._state[:3] - self._target)
        reward = -dist
        
        success = dist < 0.05
        if success:
            reward = 10.0
            
        terminated = success
        truncated = self._step_count >= self.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, {"success": success}
        
    def _get_obs(self):
        return self._state.copy()


# ============================================
# Simple Policy and Value Networks
# ============================================

class GaussianPolicy(nn.Module):
    """Simple Gaussian policy for continuous control."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        if isinstance(obs, dict):
            # Handle dict observations by concatenating
            obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
        
        h = self.trunk(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return Normal(mean, std.expand_as(mean))


class ValueFunction(nn.Module):
    """Simple value function network."""
    
    def __init__(self, obs_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs):
        if isinstance(obs, dict):
            obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
        return self.net(obs)


# ============================================
# Main Training Function
# ============================================

def create_environment(cfg: DictConfig) -> gym.Env:
    """Create environment based on configuration."""
    env_cfg = cfg.get("env", {})
    simulator_type = env_cfg.get("simulator_type", "mock")
    
    if simulator_type == "mock":
        return MockRobotEnv(
            obs_dim=env_cfg.get("obs_dim", 14),
            action_dim=env_cfg.get("action_dim", 7),
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
        )
    elif simulator_type == "gym":
        env_name = env_cfg.get("env_name", "Pendulum-v1")
        return gym.make(env_name)
    else:
        raise NotImplementedError(f"Unknown simulator type: {simulator_type}")


def main():
    parser = argparse.ArgumentParser(description="PPO Training with optional HIL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ppo_standard.yaml",
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
        default=None,
        help="Override max training steps from config",
    )
    parser.add_argument(
        "--enable_hil",
        action="store_true",
        help="Enable HIL (overrides config)",
    )
    parser.add_argument(
        "--no_hil",
        action="store_true",
        help="Disable HIL (overrides config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration (standard PPO)")
        cfg = OmegaConf.create({
            "hil": {"enable": False},
            "ppo": {},
            "env": {"simulator_type": "mock"},
            "logging": {},
            "max_steps": 10000,
        })
    
    # Override HIL setting if specified
    if args.enable_hil:
        cfg.hil.enable = True
    elif args.no_hil:
        cfg.hil.enable = False
    
    # Override max_steps if specified
    if args.max_steps:
        cfg.max_steps = args.max_steps
    
    print("\n" + "=" * 60)
    print("PPO Training Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")
    
    # Create environment
    env = create_environment(cfg)
    print(f"Created environment: {type(env).__name__}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Get dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_dim = sum(s.shape[0] for s in env.observation_space.spaces.values())
    else:
        obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy and value function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = GaussianPolicy(obs_dim, action_dim).to(device)
    value_fn = ValueFunction(obs_dim).to(device)
    
    print(f"\nUsing device: {device}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value function parameters: {sum(p.numel() for p in value_fn.parameters()):,}")
    
    # Create runner config
    hil_cfg = cfg.get("hil", {})
    ppo_cfg = cfg.get("ppo", {})
    rollout_cfg = cfg.get("rollout", {})
    logging_cfg = cfg.get("logging", {})
    env_cfg = cfg.get("env", {})
    
    runner_config = EmbodiedHILConfig(
        # HIL settings
        enable_hil=hil_cfg.get("enable", False),
        input_device=hil_cfg.get("input_device", "keyboard"),
        bc_weight=hil_cfg.get("bc_weight", 0.1),
        intervention_advantage_boost=hil_cfg.get("intervention_advantage_boost", 1.5),
        
        # PPO settings
        clip_ratio=ppo_cfg.get("clip_ratio", 0.2),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        num_epochs=ppo_cfg.get("num_epochs", 10),
        minibatch_size=ppo_cfg.get("minibatch_size", 64),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        
        # Rollout settings
        rollout_steps=rollout_cfg.get("rollout_steps", 2048),
        num_envs=rollout_cfg.get("num_envs", 1),
        
        # Environment settings
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        
        # Logging settings
        log_interval=logging_cfg.get("log_interval", 10),
        eval_interval=logging_cfg.get("eval_interval", 100),
        save_interval=logging_cfg.get("save_interval", 1000),
        checkpoint_dir=logging_cfg.get("checkpoint_dir", "./checkpoints"),
    )
    
    # Create runner
    runner = EmbodiedHILRunner(
        env=env,
        policy=policy,
        value_fn=value_fn,
        config=runner_config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        runner.load_checkpoint(args.resume)
    
    # Train
    max_steps = cfg.get("max_steps", 10000)
    
    try:
        stats = runner.train(max_steps=max_steps)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Final step: {stats['final_step']}")
        print(f"  Episodes: {stats['episodes']}")
        if runner_config.enable_hil:
            print(f"  Total interventions: {stats['total_interventions']}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_stats = runner.evaluate(num_episodes=10)
        print(f"  Mean return: {eval_stats['eval/mean_return']:.2f}")
        print(f"  Success rate: {eval_stats['eval/success_rate']:.2%}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        runner.save_checkpoint()
        
    finally:
        env.close()


if __name__ == "__main__":
    main()

