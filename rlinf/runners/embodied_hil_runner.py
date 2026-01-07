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

"""Embodied Runner with optional Human-in-the-Loop (HIL) support.

This runner extends the standard EmbodiedRunner to support HIL during PPO training.
It can be configured to run in three modes:

1. **Standard PPO** (enable_hil=False):
   - Normal PPO training without any human intervention
   - Same as EmbodiedRunner

2. **PPO + HIL** (enable_hil=True):
   - Human can intervene during rollouts
   - Adds behavior cloning loss for intervention steps
   - Intervention-aware advantage computation

3. **Demo-only mode** (enable_hil=True, policy_frozen=True):
   - Human provides all actions (pure demonstration collection)
   - Useful for collecting initial demonstrations

Usage:
    # Standard PPO (no HIL)
    python train_embodied.py --config config/ppo_standard.yaml
    
    # PPO with HIL
    python train_embodied.py --config config/ppo_hil.yaml
    
Configuration:
    hil:
      enable: true/false
      input_device: "keyboard"  # or "vivetracker", "spacemouse"
      bc_weight: 0.1
      intervention_advantage_boost: 1.5
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.algorithms.ppo_hil import (
    PPOHILConfig,
    compute_bc_loss,
    compute_intervention_aware_advantages,
    compute_ppo_hil_loss,
)
from rlinf.envs.hil.intervention_wrapper import HILInterventionWrapper
from rlinf.envs.hil.input_devices import InputDeviceConfig
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger


@dataclass
class EmbodiedHILConfig:
    """Configuration for Embodied training with optional HIL."""
    
    # HIL settings
    enable_hil: bool = False  # Master switch for HIL
    input_device: str = "keyboard"  # "keyboard", "vivetracker", "spacemouse", "mock"
    
    # BC loss weight (only used when enable_hil=True)
    bc_weight: float = 0.1
    intervention_advantage_boost: float = 1.5
    
    # PPO settings
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training settings
    num_envs: int = 1
    rollout_steps: int = 2048
    num_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    
    # Environment settings
    max_episode_steps: int = 200
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    def to_ppo_hil_config(self) -> PPOHILConfig:
        """Convert to PPOHILConfig for loss computation."""
        return PPOHILConfig(
            clip_ratio=self.clip_ratio,
            vf_coef=self.vf_coef,
            entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
            bc_weight=self.bc_weight if self.enable_hil else 0.0,
            intervention_advantage_boost=self.intervention_advantage_boost,
            use_intervention_weighting=self.enable_hil,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )


class EmbodiedHILRunner:
    """Runner for Embodied PPO training with optional HIL support.
    
    This runner provides a unified interface for:
    - Standard PPO training (when enable_hil=False)
    - PPO with Human-in-the-Loop (when enable_hil=True)
    
    The runner handles:
    - Environment wrapping with HIL intervention (if enabled)
    - Rollout collection with intervention tracking
    - Advantage computation (with intervention awareness if HIL)
    - PPO training with optional BC loss
    
    Example:
        # Create environment
        env = gym.make("MyRobotEnv-v0")
        
        # Create runner (HIL enabled via config)
        config = EmbodiedHILConfig(enable_hil=True, input_device="keyboard")
        runner = EmbodiedHILRunner(env, policy, value_fn, config)
        
        # Train
        runner.train(max_steps=100000)
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
        value_fn: nn.Module,
        config: EmbodiedHILConfig,
        device: Union[str, torch.device] = "cuda",
    ):
        self.config = config
        self.device = torch.device(device)
        self.ppo_config = config.to_ppo_hil_config()
        
        # Wrap environment with HIL if enabled
        if config.enable_hil:
            self.env = HILInterventionWrapper(
                env,
                input_device_type=config.input_device,
                input_device_config=InputDeviceConfig(
                    action_dim=env.action_space.shape[0],
                ),
            )
            print(f"[EmbodiedHILRunner] HIL enabled with {config.input_device} input device")
        else:
            self.env = env
            print("[EmbodiedHILRunner] Running standard PPO (no HIL)")
        
        # Models
        self.policy = policy.to(self.device)
        self.value_fn = value_fn.to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_fn.parameters(), lr=config.learning_rate
        )
        
        # Metrics
        self.global_step = 0
        self.episode_count = 0
        self.total_interventions = 0
        
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = None
        
    def _init_logging(self, log_path: str, experiment_name: str):
        """Initialize metric logger."""
        self.metric_logger = MetricLogger(
            DictConfig({
                "runner": {
                    "logger": {
                        "log_path": log_path,
                        "experiment_name": experiment_name,
                        "logger_backends": ["tensorboard"],
                    }
                }
            })
        )
    
    def collect_rollout(self) -> Dict[str, torch.Tensor]:
        """Collect rollout data with potential human interventions.
        
        Returns:
            Dictionary containing:
            - observations: [T, obs_dim]
            - actions: [T, action_dim]
            - rewards: [T]
            - dones: [T]
            - values: [T+1]
            - old_logprobs: [T]
            - is_intervention: [T] (all False if HIL disabled)
        """
        config = self.config
        
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        logprobs = []
        is_interventions = []
        
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        
        for step in range(config.rollout_steps):
            # Convert observation to tensor
            if isinstance(obs, dict):
                obs_tensor = {k: torch.tensor(v, device=self.device).unsqueeze(0) 
                             for k, v in obs.items()}
            else:
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            # Get policy action and value
            with torch.no_grad():
                action_dist = self.policy(obs_tensor)
                value = self.value_fn(obs_tensor)
                
                policy_action = action_dist.sample()
                policy_logprob = action_dist.log_prob(policy_action)
                
                # Sum log probs if multi-dimensional action
                if policy_logprob.dim() > 1:
                    policy_logprob = policy_logprob.sum(dim=-1)
            
            # Convert action to numpy for environment
            action_np = policy_action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Handle HIL intervention
            if config.enable_hil:
                executed_action = info.get("executed_action", action_np)
                is_intervention = info.get("is_intervention", False)
                
                if is_intervention:
                    self.total_interventions += 1
                    # Compute log_prob of human action under current policy
                    human_action_tensor = torch.tensor(
                        executed_action, device=self.device, dtype=torch.float32
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logprob = action_dist.log_prob(human_action_tensor)
                        if logprob.dim() > 1:
                            logprob = logprob.sum(dim=-1)
                else:
                    executed_action = action_np
                    logprob = policy_logprob
            else:
                executed_action = action_np
                is_intervention = False
                logprob = policy_logprob
            
            # Store transition
            observations.append(obs)
            actions.append(executed_action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze().item())
            logprobs.append(logprob.squeeze().item())
            is_interventions.append(is_intervention)
            
            # Update counters
            episode_return += reward
            episode_length += 1
            self.global_step += 1
            
            # Handle episode end
            if done:
                self.episode_count += 1
                if self.metric_logger:
                    self.metric_logger.log({
                        "episode/return": episode_return,
                        "episode/length": episode_length,
                    }, self.global_step)
                
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                obs = next_obs
        
        # Get final value for bootstrapping
        if isinstance(obs, dict):
            obs_tensor = {k: torch.tensor(v, device=self.device).unsqueeze(0) 
                         for k, v in obs.items()}
        else:
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            final_value = self.value_fn(obs_tensor).squeeze().item()
        values.append(final_value)
        
        # Convert to tensors
        def to_tensor(x):
            if isinstance(x[0], dict):
                return {k: torch.tensor(np.array([xi[k] for xi in x]), 
                        device=self.device, dtype=torch.float32) for k in x[0].keys()}
            return torch.tensor(np.array(x), device=self.device, dtype=torch.float32)
        
        return {
            "observations": to_tensor(observations),
            "actions": to_tensor(actions),
            "rewards": torch.tensor(rewards, device=self.device, dtype=torch.float32),
            "dones": torch.tensor(dones, device=self.device, dtype=torch.bool),
            "values": torch.tensor(values, device=self.device, dtype=torch.float32),
            "old_logprobs": torch.tensor(logprobs, device=self.device, dtype=torch.float32),
            "is_intervention": torch.tensor(is_interventions, device=self.device, dtype=torch.bool),
        }
    
    def compute_advantages(
        self, 
        rollout: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns.
        
        If HIL is enabled, uses intervention-aware advantage computation.
        """
        if self.config.enable_hil:
            return compute_intervention_aware_advantages(
                rewards=rollout["rewards"],
                values=rollout["values"],
                dones=rollout["dones"],
                is_intervention=rollout["is_intervention"],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                intervention_boost=self.config.intervention_advantage_boost,
            )
        else:
            # Standard GAE (no intervention boost)
            return compute_intervention_aware_advantages(
                rewards=rollout["rewards"],
                values=rollout["values"],
                dones=rollout["dones"],
                is_intervention=rollout["is_intervention"],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                intervention_boost=1.0,  # No boost
            )
    
    def train_step(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run PPO training on collected rollout.
        
        Args:
            rollout: Rollout data from collect_rollout()
            
        Returns:
            Training metrics
        """
        config = self.config
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rollout)
        
        # Flatten observations if dict
        observations = rollout["observations"]
        
        # Training loop
        all_metrics = defaultdict(list)
        num_samples = len(rollout["rewards"])
        
        for epoch in range(config.num_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            
            for start in range(0, num_samples, config.minibatch_size):
                end = min(start + config.minibatch_size, num_samples)
                mb_indices = indices[start:end]
                
                # Get minibatch
                if isinstance(observations, dict):
                    mb_obs = {k: v[mb_indices] for k, v in observations.items()}
                else:
                    mb_obs = observations[mb_indices]
                mb_actions = rollout["actions"][mb_indices]
                mb_old_logprobs = rollout["old_logprobs"][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_is_intervention = rollout["is_intervention"][mb_indices]
                
                # Forward pass
                action_dist = self.policy(mb_obs)
                new_logprobs = action_dist.log_prob(mb_actions)
                if new_logprobs.dim() > 1:
                    new_logprobs = new_logprobs.sum(dim=-1)
                entropy = action_dist.entropy()
                if entropy.dim() > 1:
                    entropy = entropy.sum(dim=-1)
                new_values = self.value_fn(mb_obs).squeeze(-1)
                
                # Compute loss
                loss, metrics = compute_ppo_hil_loss(
                    logprobs=new_logprobs,
                    old_logprobs=mb_old_logprobs,
                    advantages=mb_advantages,
                    is_intervention=mb_is_intervention,
                    values=new_values,
                    returns=mb_returns,
                    config=self.ppo_config,
                    entropy=entropy,
                )
                
                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    config.max_grad_norm
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Collect metrics
                for k, v in metrics.items():
                    all_metrics[k].append(v)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        # Add HIL-specific metrics
        if self.config.enable_hil:
            avg_metrics["hil/total_interventions"] = self.total_interventions
            avg_metrics["hil/intervention_rate"] = (
                rollout["is_intervention"].float().mean().item()
            )
        
        return avg_metrics
    
    def train(self, max_steps: int) -> Dict[str, Any]:
        """Main training loop.
        
        Args:
            max_steps: Maximum training steps
            
        Returns:
            Final training statistics
        """
        print(f"\n{'='*60}")
        print(f"Starting {'PPO + HIL' if self.config.enable_hil else 'Standard PPO'} Training")
        print(f"{'='*60}")
        print(f"  Max steps: {max_steps}")
        print(f"  Rollout steps: {self.config.rollout_steps}")
        print(f"  HIL enabled: {self.config.enable_hil}")
        if self.config.enable_hil:
            print(f"  Input device: {self.config.input_device}")
            print(f"  BC weight: {self.config.bc_weight}")
        print(f"{'='*60}\n")
        
        pbar = tqdm(total=max_steps, desc="Training")
        
        while self.global_step < max_steps:
            with self.timer("rollout"):
                rollout = self.collect_rollout()
            
            with self.timer("training"):
                metrics = self.train_step(rollout)
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                time_metrics = self.timer.consume_durations()
                all_metrics = {**metrics, **{f"time/{k}": v for k, v in time_metrics.items()}}
                
                if self.metric_logger:
                    self.metric_logger.log(all_metrics, self.global_step)
                
                # Print progress
                pbar.set_postfix({
                    "policy_loss": f"{metrics.get('policy_loss', 0):.4f}",
                    "vf_loss": f"{metrics.get('vf_loss', 0):.4f}",
                    "bc_loss": f"{metrics.get('bc_loss', 0):.4f}" if self.config.enable_hil else "N/A",
                    "episodes": self.episode_count,
                })
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint()
            
            pbar.update(self.config.rollout_steps)
        
        pbar.close()
        self.env.close()
        
        return {
            "final_step": self.global_step,
            "episodes": self.episode_count,
            "total_interventions": self.total_interventions,
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy (no HIL, deterministic actions).
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        # Use base environment for evaluation (no HIL wrapper)
        eval_env = self.env.env if self.config.enable_hil else self.env
        
        returns = []
        lengths = []
        successes = []
        
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                if isinstance(obs, dict):
                    obs_tensor = {k: torch.tensor(v, device=self.device).unsqueeze(0) 
                                 for k, v in obs.items()}
                else:
                    obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    action_dist = self.policy(obs_tensor)
                    action = action_dist.mode if hasattr(action_dist, 'mode') else action_dist.mean
                
                obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(info.get("success", 0))
        
        return {
            "eval/mean_return": np.mean(returns),
            "eval/std_return": np.std(returns),
            "eval/mean_length": np.mean(lengths),
            "eval/success_rate": np.mean(successes),
        }
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.global_step}.pt"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_fn_state_dict": self.value_fn.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "total_interventions": self.total_interventions,
            "config": self.config,
        }, path)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_fn.load_state_dict(checkpoint["value_fn_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.episode_count = checkpoint["episode_count"]
        self.total_interventions = checkpoint.get("total_interventions", 0)
        
        print(f"Checkpoint loaded from {path}")

