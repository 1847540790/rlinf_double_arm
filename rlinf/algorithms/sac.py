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

"""Soft Actor-Critic (SAC) algorithm implementation.

This module provides loss functions and utilities for SAC training.
SAC is an off-policy actor-critic algorithm that:
    - Maximizes expected return + entropy
    - Uses twin Q-functions to reduce overestimation
    - Automatically tunes temperature (alpha)

References:
    - SAC: https://arxiv.org/abs/1801.01290
    - SAC v2: https://arxiv.org/abs/1812.05905
    - RLPD: https://arxiv.org/abs/2302.02948

Usage with RLinf registry:
    from rlinf.algorithms.sac import compute_sac_critic_loss, compute_sac_actor_loss
    
    critic_loss, critic_info = compute_sac_critic_loss(
        q_values=q1_pred,
        q_values_2=q2_pred,
        target_q=target_q,
        ...
    )
"""

from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rlinf.algorithms.registry import register_policy_loss


def soft_target_update(
    target_params: Dict[str, torch.Tensor],
    source_params: Dict[str, torch.Tensor],
    tau: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Soft update target network parameters.
    
    target = tau * source + (1 - tau) * target
    
    Args:
        target_params: Target network parameters (state_dict)
        source_params: Source network parameters (state_dict)
        tau: Interpolation factor (0 = no update, 1 = hard update)
        
    Returns:
        Updated target parameters
    """
    updated = {}
    for key in target_params:
        updated[key] = tau * source_params[key] + (1 - tau) * target_params[key]
    return updated


def compute_target_q(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    next_log_probs: torch.Tensor,
    gamma: float,
    alpha: torch.Tensor,
    backup_entropy: bool = True,
) -> torch.Tensor:
    """Compute SAC target Q-value.
    
    target_q = r + gamma * (1 - done) * (min(Q1', Q2') - alpha * log_pi)
    
    Args:
        rewards: Rewards [B]
        dones: Done flags [B]
        next_q1: Q1 values for next state [B]
        next_q2: Q2 values for next state [B]
        next_log_probs: Log probabilities of next actions [B]
        gamma: Discount factor
        alpha: Temperature parameter
        backup_entropy: Whether to include entropy in backup
        
    Returns:
        Target Q-values [B]
    """
    # Minimum of twin Q-values
    next_q_min = torch.min(next_q1, next_q2)
    
    # Entropy bonus
    if backup_entropy:
        next_v = next_q_min - alpha * next_log_probs
    else:
        next_v = next_q_min
        
    # TD target
    masks = 1.0 - dones.float()
    target_q = rewards + gamma * masks * next_v
    
    return target_q.detach()


@register_policy_loss("sac_critic")
def compute_sac_critic_loss(
    q_values: torch.Tensor,
    q_values_2: Optional[torch.Tensor],
    target_q: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Compute SAC critic loss.
    
    Loss = MSE(Q1(s,a), target) + MSE(Q2(s,a), target)
    
    Args:
        q_values: Q1 predictions [B]
        q_values_2: Q2 predictions [B] (optional, for twin critics)
        target_q: Target Q-values [B]
        loss_mask: Optional mask for valid entries [B]
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Q1 loss
    q1_loss = F.mse_loss(q_values, target_q, reduction='none')
    
    # Q2 loss (if using twin critics)
    if q_values_2 is not None:
        q2_loss = F.mse_loss(q_values_2, target_q, reduction='none')
        critic_loss = q1_loss + q2_loss
    else:
        q2_loss = torch.zeros_like(q1_loss)
        critic_loss = q1_loss
        
    # Apply mask if provided
    if loss_mask is not None:
        critic_loss = critic_loss * loss_mask
        q1_loss = q1_loss * loss_mask
        q2_loss = q2_loss * loss_mask
        critic_loss = critic_loss.sum() / (loss_mask.sum() + 1e-8)
    else:
        critic_loss = critic_loss.mean()
        
    metrics = {
        "critic_loss": critic_loss.detach().item(),
        "q1_loss": q1_loss.mean().detach().item(),
        "q2_loss": q2_loss.mean().detach().item(),
        "q1_mean": q_values.mean().detach().item(),
        "target_q_mean": target_q.mean().detach().item(),
    }
    
    if q_values_2 is not None:
        metrics["q2_mean"] = q_values_2.mean().detach().item()
        
    return critic_loss, metrics


@register_policy_loss("sac_actor")
def compute_sac_actor_loss(
    log_probs: torch.Tensor,
    q_values: torch.Tensor,
    alpha: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Compute SAC actor loss.
    
    Loss = E[alpha * log_pi(a|s) - Q(s, a)]
    
    The actor tries to maximize Q while also maximizing entropy.
    
    Args:
        log_probs: Log probabilities of sampled actions [B]
        q_values: Q-values for sampled actions [B]
        alpha: Temperature parameter (scalar or [1])
        loss_mask: Optional mask for valid entries [B]
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Actor objective: maximize Q - alpha * log_pi
    # So loss is: alpha * log_pi - Q
    actor_loss = alpha * log_probs - q_values
    
    # Apply mask if provided
    if loss_mask is not None:
        actor_loss = actor_loss * loss_mask
        actor_loss = actor_loss.sum() / (loss_mask.sum() + 1e-8)
    else:
        actor_loss = actor_loss.mean()
        
    entropy = -log_probs.mean().detach()
    
    metrics = {
        "actor_loss": actor_loss.detach().item(),
        "entropy": entropy.item(),
        "alpha": alpha.detach().item() if torch.is_tensor(alpha) else alpha,
        "log_probs_mean": log_probs.mean().detach().item(),
    }
    
    return actor_loss, metrics


@register_policy_loss("sac_temperature")
def compute_sac_temperature_loss(
    log_probs: torch.Tensor,
    log_alpha: torch.Tensor,
    target_entropy: float,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Compute SAC temperature (alpha) loss.
    
    Loss = -alpha * (log_pi + target_entropy)
    
    This adjusts alpha to maintain target entropy:
        - If entropy < target: increase alpha to encourage exploration
        - If entropy > target: decrease alpha to reduce randomness
    
    Args:
        log_probs: Log probabilities of sampled actions [B]
        log_alpha: Log of temperature parameter (learnable)
        target_entropy: Target entropy (usually -action_dim)
        loss_mask: Optional mask for valid entries [B]
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    alpha = log_alpha.exp()
    
    # Temperature loss
    alpha_loss = -alpha * (log_probs.detach() + target_entropy)
    
    # Apply mask if provided
    if loss_mask is not None:
        alpha_loss = alpha_loss * loss_mask
        alpha_loss = alpha_loss.sum() / (loss_mask.sum() + 1e-8)
    else:
        alpha_loss = alpha_loss.mean()
        
    metrics = {
        "temperature_loss": alpha_loss.detach().item(),
        "alpha": alpha.detach().item(),
        "log_alpha": log_alpha.detach().item(),
        "target_entropy": target_entropy,
    }
    
    return alpha_loss, metrics


class SACLosses:
    """Helper class that computes all SAC losses together.
    
    This provides a convenient interface for computing critic, actor,
    and temperature losses in one call.
    
    Args:
        gamma: Discount factor
        tau: Soft target update rate
        target_entropy: Target entropy (default: -action_dim)
        backup_entropy: Include entropy in Q backup
        critic_update_freq: How often to update critic (vs actor)
        
    Example:
        sac_losses = SACLosses(gamma=0.99, target_entropy=-7)
        
        losses, metrics = sac_losses.compute_losses(
            batch=batch,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            log_alpha=log_alpha,
        )
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.backup_entropy = backup_entropy
        self.init_temperature = init_temperature
        
    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        critic,
        target_critic,
        actor,
        log_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute critic loss with target Q computation.
        
        Args:
            observations: Current observations [B, obs_dim]
            actions: Actions taken [B, action_dim]
            rewards: Rewards received [B]
            next_observations: Next observations [B, obs_dim]
            dones: Done flags [B]
            critic: Critic network (twin Q-functions)
            target_critic: Target critic network
            actor: Actor network
            log_alpha: Log temperature parameter
            
        Returns:
            Tuple of (loss, metrics)
        """
        alpha = log_alpha.exp().detach()
        
        # Sample next actions and get log probs
        with torch.no_grad():
            next_actions, next_log_probs = actor.sample_with_log_prob(next_observations)
            
            # Get target Q-values
            next_q1, next_q2 = target_critic(next_observations, next_actions)
            
            # Compute target
            target_q = compute_target_q(
                rewards=rewards,
                dones=dones,
                next_q1=next_q1,
                next_q2=next_q2,
                next_log_probs=next_log_probs,
                gamma=self.gamma,
                alpha=alpha,
                backup_entropy=self.backup_entropy,
            )
            
        # Current Q-values
        q1, q2 = critic(observations, actions)
        
        # Compute loss
        loss, metrics = compute_sac_critic_loss(
            q_values=q1,
            q_values_2=q2,
            target_q=target_q,
        )
        
        return loss, metrics
        
    def compute_actor_loss(
        self,
        observations: torch.Tensor,
        actor,
        critic,
        log_alpha: torch.Tensor,
        use_min_q: bool = False,  # HIL-SERL uses mean, not min
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute actor loss.
        
        Args:
            observations: Current observations [B, obs_dim]
            actor: Actor network
            critic: Critic network
            log_alpha: Log temperature parameter
            use_min_q: If True, use min(Q1, Q2). If False (HIL-SERL default), use mean.
            
        Returns:
            Tuple of (loss, metrics)
        """
        alpha = log_alpha.exp()
        
        # Sample actions and get log probs
        actions, log_probs = actor.sample_with_log_prob(observations)
        
        # Get Q-values for sampled actions
        q1, q2 = critic(observations, actions)
        
        # HIL-SERL uses mean of ensemble, not min
        # (min is used in target Q computation, not actor loss)
        if use_min_q:
            q_value = torch.min(q1, q2)
        else:
            q_value = (q1 + q2) / 2  # Mean of twin Q (HIL-SERL style)
        
        # Compute loss
        loss, metrics = compute_sac_actor_loss(
            log_probs=log_probs,
            q_values=q_value,
            alpha=alpha,
        )
        
        return loss, metrics
        
    def compute_temperature_loss(
        self,
        observations: torch.Tensor,
        actor,
        log_alpha: torch.Tensor,
        action_dim: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute temperature loss.
        
        Args:
            observations: Current observations [B, obs_dim]
            actor: Actor network  
            log_alpha: Log temperature parameter
            action_dim: Dimension of action space
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Set target entropy if not specified
        # HIL-SERL uses -action_dim / 2 as default
        target_entropy = self.target_entropy
        if target_entropy is None:
            target_entropy = -action_dim / 2  # HIL-SERL default
            
        # Sample actions and get log probs
        with torch.no_grad():
            _, log_probs = actor.sample_with_log_prob(observations)
            
        # Compute loss
        loss, metrics = compute_sac_temperature_loss(
            log_probs=log_probs,
            log_alpha=log_alpha,
            target_entropy=target_entropy,
        )
        
        return loss, metrics
        
    def compute_all_losses(
        self,
        batch: Dict[str, torch.Tensor],
        actor,
        critic,
        target_critic,
        log_alpha: torch.Tensor,
        action_dim: int,
        update_actor: bool = True,
        update_temperature: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Compute all SAC losses.
        
        Args:
            batch: Batch dictionary with observations, actions, rewards, etc.
            actor: Actor network
            critic: Critic network
            target_critic: Target critic network
            log_alpha: Log temperature parameter
            action_dim: Dimension of action space
            update_actor: Whether to compute actor loss
            update_temperature: Whether to compute temperature loss
            
        Returns:
            Tuple of (losses_dict, metrics_dict)
        """
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        
        losses = {}
        all_metrics = {}
        
        # Critic loss (always computed)
        critic_loss, critic_metrics = self.compute_critic_loss(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            critic=critic,
            target_critic=target_critic,
            actor=actor,
            log_alpha=log_alpha,
        )
        losses["critic"] = critic_loss
        all_metrics.update({f"critic/{k}": v for k, v in critic_metrics.items()})
        
        # Actor loss (optional, for high UTD)
        if update_actor:
            actor_loss, actor_metrics = self.compute_actor_loss(
                observations=observations,
                actor=actor,
                critic=critic,
                log_alpha=log_alpha,
            )
            losses["actor"] = actor_loss
            all_metrics.update({f"actor/{k}": v for k, v in actor_metrics.items()})
            
        # Temperature loss (optional)
        if update_temperature:
            temp_loss, temp_metrics = self.compute_temperature_loss(
                observations=observations,
                actor=actor,
                log_alpha=log_alpha,
                action_dim=action_dim,
            )
            losses["temperature"] = temp_loss
            all_metrics.update({f"temperature/{k}": v for k, v in temp_metrics.items()})
            
        return losses, all_metrics

