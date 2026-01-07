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

"""PPO with Human-in-the-Loop (HIL) support.

This module extends PPO to support human interventions during online training.
There are two main ways to incorporate human demonstrations into PPO:

1. **Intervention-Aware PPO**: When human intervenes during rollout, we:
   - Record the human action instead of the policy's action
   - Compute log_prob of human action under current policy
   - Weight intervention steps differently in the loss

2. **PPO + Imitation Learning**: Add an auxiliary BC loss:
   - L_total = L_ppo + λ * L_bc
   - L_bc pulls the policy toward human demonstrations
   - This is similar to the RLHF approach

Usage:
    # Wrap environment with HIL intervention
    env = HILInterventionWrapper(robot_env, input_device="keyboard")
    
    # Collect rollout with interventions
    rollout_data = collect_rollout_with_hil(env, policy)
    
    # Compute PPO loss with intervention awareness
    loss = compute_ppo_hil_loss(rollout_data, policy, bc_weight=0.1)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOHILConfig:
    """Configuration for PPO with HIL."""
    
    # PPO config
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # HIL config
    bc_weight: float = 0.1  # Weight for behavior cloning loss
    intervention_advantage_boost: float = 1.5  # Boost advantage for intervention steps
    use_intervention_weighting: bool = True  # Weight intervention steps higher
    
    # Rollout config
    gamma: float = 0.99
    gae_lambda: float = 0.95


def compute_intervention_aware_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    is_intervention: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    intervention_boost: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages with intervention awareness.
    
    Human interventions are assumed to be "better" actions, so we can:
    1. Boost the advantage for intervention steps
    2. Or use intervention as implicit positive reward signal
    
    Args:
        rewards: Rewards [T]
        values: Value estimates [T+1]
        dones: Done flags [T]
        is_intervention: Whether each step was human intervention [T]
        gamma: Discount factor
        gae_lambda: GAE lambda
        intervention_boost: Multiplier for intervention step advantages
        
    Returns:
        Tuple of (advantages, returns)
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    
    # Boost advantages for intervention steps
    if intervention_boost != 1.0:
        intervention_mask = is_intervention.float()
        advantages = advantages * (1.0 + (intervention_boost - 1.0) * intervention_mask)
    
    returns = advantages + values[:-1]
    return advantages, returns


def compute_bc_loss(
    policy_logprobs: torch.Tensor,
    is_intervention: torch.Tensor,
) -> torch.Tensor:
    """Compute behavior cloning loss for intervention steps.
    
    Maximizes log probability of human actions under the policy.
    Only applies to steps where human intervened.
    
    Args:
        policy_logprobs: Log probs of actions under current policy [B, T]
        is_intervention: Whether each step was human intervention [B, T]
        
    Returns:
        BC loss (to be minimized)
    """
    # Only compute loss for intervention steps
    intervention_mask = is_intervention.float()
    
    if intervention_mask.sum() == 0:
        return torch.tensor(0.0, device=policy_logprobs.device)
    
    # BC loss = -log_prob (maximize log prob of human actions)
    bc_loss = -policy_logprobs * intervention_mask
    bc_loss = bc_loss.sum() / (intervention_mask.sum() + 1e-8)
    
    return bc_loss


def compute_ppo_hil_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    is_intervention: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    config: PPOHILConfig,
    entropy: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO loss with HIL awareness.
    
    L_total = L_ppo + bc_weight * L_bc + vf_coef * L_vf - entropy_coef * H
    
    Args:
        logprobs: Current policy log probs [B, T]
        old_logprobs: Old policy log probs [B, T]
        advantages: Advantage estimates [B, T]
        is_intervention: Intervention flags [B, T]
        values: Value predictions [B, T]
        returns: Computed returns [B, T]
        config: PPO HIL configuration
        entropy: Optional entropy bonus [B, T]
        
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Standard PPO ratio and clipping
    ratio = torch.exp(logprobs - old_logprobs)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value function loss
    vf_loss = F.mse_loss(values, returns)
    
    # Behavior cloning loss for intervention steps
    bc_loss = compute_bc_loss(logprobs, is_intervention)
    
    # Entropy bonus
    if entropy is not None:
        entropy_loss = -entropy.mean()
    else:
        entropy_loss = torch.tensor(0.0, device=logprobs.device)
    
    # Total loss
    total_loss = (
        policy_loss
        + config.vf_coef * vf_loss
        + config.bc_weight * bc_loss
        + config.entropy_coef * entropy_loss
    )
    
    # Metrics
    metrics = {
        "policy_loss": policy_loss.detach().item(),
        "vf_loss": vf_loss.detach().item(),
        "bc_loss": bc_loss.detach().item(),
        "entropy_loss": entropy_loss.detach().item(),
        "total_loss": total_loss.detach().item(),
        "ratio_mean": ratio.mean().detach().item(),
        "intervention_rate": is_intervention.float().mean().detach().item(),
    }
    
    return total_loss, metrics


class PPOHILTrainer:
    """Trainer for PPO with Human-in-the-Loop.
    
    This trainer extends standard PPO to support:
    1. Human intervention during rollouts
    2. Behavior cloning loss for intervention data
    3. Intervention-aware advantage computation
    
    Usage:
        trainer = PPOHILTrainer(policy, value_fn, config)
        
        # Collect rollout with interventions
        rollout = trainer.collect_rollout(env)
        
        # Train with intervention awareness
        loss, metrics = trainer.train_step(rollout)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module,
        config: PPOHILConfig,
        device: str = "cuda",
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.config = config
        self.device = device
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=3e-4
        )
        self.value_optimizer = torch.optim.Adam(
            value_fn.parameters(), lr=1e-3
        )
        
    def collect_rollout(
        self,
        env,  # Should be wrapped with HILInterventionWrapper
        num_steps: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """Collect rollout data with potential human interventions.
        
        Args:
            env: Environment wrapped with HILInterventionWrapper
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary with rollout data including intervention flags
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        logprobs = []
        is_interventions = []
        
        obs, _ = env.reset()
        
        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
            
            # Get policy action and value
            with torch.no_grad():
                action_dist = self.policy(obs_tensor)
                value = self.value_fn(obs_tensor)
                
                policy_action = action_dist.sample()
                policy_logprob = action_dist.log_prob(policy_action)
            
            # Step environment (HIL wrapper may override action)
            next_obs, reward, terminated, truncated, info = env.step(
                policy_action.cpu().numpy()[0]
            )
            done = terminated or truncated
            
            # Get actual executed action (may be human override)
            executed_action = info.get("executed_action", policy_action.cpu().numpy()[0])
            is_intervention = info.get("is_intervention", False)
            
            # If human intervened, compute log_prob of human action
            if is_intervention:
                human_action_tensor = torch.tensor(
                    executed_action, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    logprob = action_dist.log_prob(human_action_tensor)
            else:
                logprob = policy_logprob
            
            # Store transition
            observations.append(obs)
            actions.append(executed_action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            logprobs.append(logprob.item())
            is_interventions.append(is_intervention)
            
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
        
        # Get final value
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
            final_value = self.value_fn(obs_tensor).item()
        values.append(final_value)
        
        return {
            "observations": torch.tensor(observations, device=self.device),
            "actions": torch.tensor(actions, device=self.device),
            "rewards": torch.tensor(rewards, device=self.device),
            "dones": torch.tensor(dones, device=self.device),
            "values": torch.tensor(values, device=self.device),
            "old_logprobs": torch.tensor(logprobs, device=self.device),
            "is_intervention": torch.tensor(is_interventions, device=self.device),
        }
    
    def train_step(
        self,
        rollout: Dict[str, torch.Tensor],
        num_epochs: int = 10,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """Train policy and value function on rollout data.
        
        Args:
            rollout: Rollout data from collect_rollout
            num_epochs: Number of training epochs
            minibatch_size: Size of minibatches
            
        Returns:
            Training metrics
        """
        # Compute advantages with intervention awareness
        advantages, returns = compute_intervention_aware_advantages(
            rewards=rollout["rewards"],
            values=rollout["values"],
            dones=rollout["dones"],
            is_intervention=rollout["is_intervention"],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            intervention_boost=self.config.intervention_advantage_boost,
        )
        
        # Training loop
        all_metrics = []
        num_samples = len(rollout["observations"])
        
        for epoch in range(num_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            
            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = rollout["observations"][mb_indices]
                mb_actions = rollout["actions"][mb_indices]
                mb_old_logprobs = rollout["old_logprobs"][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_is_intervention = rollout["is_intervention"][mb_indices]
                
                # Forward pass
                action_dist = self.policy(mb_obs)
                new_logprobs = action_dist.log_prob(mb_actions)
                entropy = action_dist.entropy()
                new_values = self.value_fn(mb_obs).squeeze(-1)
                
                # Compute loss
                loss, metrics = compute_ppo_hil_loss(
                    logprobs=new_logprobs,
                    old_logprobs=mb_old_logprobs,
                    advantages=mb_advantages,
                    is_intervention=mb_is_intervention,
                    values=new_values,
                    returns=mb_returns,
                    config=self.config,
                    entropy=entropy,
                )
                
                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.config.max_grad_norm
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0].keys()
        }
        
        return avg_metrics

