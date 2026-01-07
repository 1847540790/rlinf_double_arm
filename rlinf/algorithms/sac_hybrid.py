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

"""SAC Hybrid Agent - SAC for continuous actions + DQN for discrete gripper.

Ported from HIL-SERL's sac_hybrid_single.py and sac_hybrid_dual.py (JAX) to PyTorch.

Key differences from standard SAC:
    - Continuous actions (6-DoF end-effector) learned via SAC
    - Discrete gripper actions (open/close/hold) learned via DQN
    - Separate grasp critic network for gripper Q-values
    - Grasp penalty added to reward for gripper actions

Supports:
    - Single arm (7-DoF action: 6 continuous + 1 discrete)
    - Dual arm (14-DoF action: 12 continuous + 2 discrete)

Usage:
    hybrid_losses = SACHybridLosses(
        setup_mode="single-arm-learned-gripper",
        gamma=0.97,
    )
    
    losses, metrics = hybrid_losses.compute_all_losses(
        batch=batch,
        actor=actor,
        critic=critic,
        grasp_critic=grasp_critic,
        ...
    )
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_grasp_critic_loss_single_arm(
    grasp_q_preds: torch.Tensor,
    target_grasp_q: torch.Tensor,
    grasp_actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """Compute grasp critic loss for single arm (DQN-style).
    
    Args:
        grasp_q_preds: Q-values for all grasp actions [B, 3] (open/hold/close)
        target_grasp_q: Target Q-values [B]
        grasp_actions: Taken grasp actions [B] in range {-1, 0, 1}
        
    Returns:
        Tuple of (loss, metrics)
    """
    batch_size = grasp_q_preds.shape[0]
    
    # Convert actions from {-1, 0, 1} to {0, 1, 2}
    action_indices = (grasp_actions + 1).long()
    
    # Get predicted Q for taken actions
    predicted_q = grasp_q_preds[torch.arange(batch_size), action_indices]
    
    # MSE loss
    loss = F.mse_loss(predicted_q, target_grasp_q)
    
    metrics = {
        "grasp_critic_loss": loss.detach().item(),
        "grasp_q_mean": predicted_q.mean().detach().item(),
        "grasp_target_q_mean": target_grasp_q.mean().detach().item(),
    }
    
    return loss, metrics


def compute_grasp_critic_loss_dual_arm(
    grasp_q_preds: torch.Tensor,
    target_grasp_q: torch.Tensor,
    grasp_actions_1: torch.Tensor,
    grasp_actions_2: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """Compute grasp critic loss for dual arm (DQN-style).
    
    Uses joint action space: 3 actions per arm = 9 total combinations.
    
    Args:
        grasp_q_preds: Q-values for all joint grasp actions [B, 9]
        target_grasp_q: Target Q-values [B]
        grasp_actions_1: Arm 1 grasp actions [B] in range {-1, 0, 1}
        grasp_actions_2: Arm 2 grasp actions [B] in range {-1, 0, 1}
        
    Returns:
        Tuple of (loss, metrics)
    """
    batch_size = grasp_q_preds.shape[0]
    
    # Convert to indices
    action_idx_1 = (grasp_actions_1 + 1).long()  # {0, 1, 2}
    action_idx_2 = (grasp_actions_2 + 1).long()  # {0, 1, 2}
    
    # Joint action index: 0-8
    joint_action_idx = action_idx_1 * 3 + action_idx_2
    
    # Get predicted Q for taken actions
    predicted_q = grasp_q_preds[torch.arange(batch_size), joint_action_idx]
    
    # MSE loss
    loss = F.mse_loss(predicted_q, target_grasp_q)
    
    metrics = {
        "grasp_critic_loss": loss.detach().item(),
        "grasp_q_mean": predicted_q.mean().detach().item(),
        "grasp_target_q_mean": target_grasp_q.mean().detach().item(),
    }
    
    return loss, metrics


@dataclass
class SACHybridConfig:
    """Configuration for SAC Hybrid agent."""
    
    # Setup mode
    setup_mode: str = "single-arm-learned-gripper"
    # Options: "single-arm-learned-gripper", "dual-arm-learned-gripper",
    #          "single-arm-fixed-gripper", "dual-arm-fixed-gripper"
    
    # SAC config
    gamma: float = 0.97
    tau: float = 0.005
    target_entropy: Optional[float] = None  # Default: -action_dim / 2
    backup_entropy: bool = False
    
    # Ensemble
    critic_ensemble_size: int = 2
    critic_subsample_size: Optional[int] = None  # For REDQ
    
    # Grasp critic
    grasp_hidden_dims: Tuple[int, ...] = (128, 128)
    
    # Temperature
    init_temperature: float = 0.01
    
    @property
    def is_learned_gripper(self) -> bool:
        return "learned-gripper" in self.setup_mode
    
    @property
    def is_dual_arm(self) -> bool:
        return "dual-arm" in self.setup_mode
    
    @property
    def continuous_action_dim(self) -> int:
        """Dimension of continuous actions (end-effector)."""
        if self.is_dual_arm:
            return 12  # 6 per arm
        return 6
    
    @property
    def discrete_action_dim(self) -> int:
        """Number of discrete gripper action options."""
        if self.is_dual_arm:
            return 9  # 3 x 3 joint actions
        return 3  # open, hold, close


class SACHybridLosses:
    """Loss computation for SAC Hybrid agent.
    
    Combines:
        - SAC losses for continuous end-effector actions
        - DQN losses for discrete gripper actions
    """
    
    def __init__(self, config: SACHybridConfig):
        self.config = config
    
    def compute_target_q(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_q1: torch.Tensor,
        next_q2: torch.Tensor,
        next_log_probs: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SAC target Q for continuous actions."""
        next_q_min = torch.min(next_q1, next_q2)
        
        if self.config.backup_entropy:
            next_v = next_q_min - alpha * next_log_probs
        else:
            next_v = next_q_min
        
        masks = 1.0 - dones.float()
        target_q = rewards + self.config.gamma * masks * next_v
        
        return target_q.detach()
    
    def compute_grasp_target_q(
        self,
        rewards: torch.Tensor,
        grasp_penalties: torch.Tensor,
        dones: torch.Tensor,
        next_grasp_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DQN target Q for gripper actions."""
        grasp_rewards = rewards + grasp_penalties
        masks = 1.0 - dones.float()
        target_q = grasp_rewards + self.config.gamma * masks * next_grasp_q
        
        return target_q.detach()
    
    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        critic: nn.Module,
        target_critic: nn.Module,
        actor: nn.Module,
        log_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute SAC critic loss for continuous actions.
        
        For hybrid agents, only uses the continuous portion of actions.
        """
        alpha = log_alpha.exp().detach()
        
        # Extract continuous actions
        if self.config.is_dual_arm:
            # Dual arm: actions[:, 0:6] and actions[:, 7:13] are continuous
            cont_actions = torch.cat([actions[:, :6], actions[:, 7:13]], dim=-1)
        else:
            # Single arm: actions[:, :-1] is continuous
            cont_actions = actions[:, :-1]
        
        # Sample next actions
        with torch.no_grad():
            next_cont_actions, next_log_probs = actor.sample_with_log_prob(next_observations)
            next_q1, next_q2 = target_critic(next_observations, next_cont_actions)
            
            target_q = self.compute_target_q(
                rewards=rewards,
                dones=dones,
                next_q1=next_q1,
                next_q2=next_q2,
                next_log_probs=next_log_probs,
                alpha=alpha,
            )
        
        # Current Q predictions
        q1, q2 = critic(observations, cont_actions)
        
        # MSE loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        metrics = {
            "critic_loss": critic_loss.detach().item(),
            "q1_mean": q1.mean().detach().item(),
            "q2_mean": q2.mean().detach().item(),
            "target_q_mean": target_q.mean().detach().item(),
        }
        
        return critic_loss, metrics
    
    def compute_grasp_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        grasp_penalties: torch.Tensor,
        grasp_critic: nn.Module,
        target_grasp_critic: nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute DQN grasp critic loss."""
        
        # Get grasp Q-values
        grasp_q_preds = grasp_critic(observations)  # [B, num_actions]
        
        with torch.no_grad():
            # Get next grasp Q-values (Double DQN style)
            next_grasp_q_online = grasp_critic(next_observations)
            next_grasp_q_target = target_grasp_critic(next_observations)
            
            # Select actions with online network, evaluate with target
            best_next_actions = next_grasp_q_online.argmax(dim=-1)
            batch_size = next_grasp_q_target.shape[0]
            next_grasp_q = next_grasp_q_target[torch.arange(batch_size), best_next_actions]
            
            target_grasp_q = self.compute_grasp_target_q(
                rewards=rewards,
                grasp_penalties=grasp_penalties,
                dones=dones,
                next_grasp_q=next_grasp_q,
            )
        
        if self.config.is_dual_arm:
            # Dual arm: grasp actions at indices 6 and 13
            grasp_actions_1 = actions[:, 6]
            grasp_actions_2 = actions[:, 13]
            loss, metrics = compute_grasp_critic_loss_dual_arm(
                grasp_q_preds, target_grasp_q, grasp_actions_1, grasp_actions_2
            )
        else:
            # Single arm: grasp action at last index
            grasp_actions = actions[:, -1]
            loss, metrics = compute_grasp_critic_loss_single_arm(
                grasp_q_preds, target_grasp_q, grasp_actions
            )
        
        return loss, metrics
    
    def compute_actor_loss(
        self,
        observations: torch.Tensor,
        actor: nn.Module,
        critic: nn.Module,
        log_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute SAC actor loss (continuous actions only)."""
        alpha = log_alpha.exp()
        
        # Sample continuous actions
        actions, log_probs = actor.sample_with_log_prob(observations)
        
        # Get Q-values (mean of ensemble, as in HIL-SERL)
        q1, q2 = critic(observations, actions)
        q_mean = (q1 + q2) / 2
        
        # Actor objective: maximize Q - alpha * log_pi
        actor_objective = q_mean - alpha * log_probs
        actor_loss = -actor_objective.mean()
        
        metrics = {
            "actor_loss": actor_loss.detach().item(),
            "entropy": (-log_probs).mean().detach().item(),
            "alpha": alpha.detach().item(),
        }
        
        return actor_loss, metrics
    
    def compute_temperature_loss(
        self,
        observations: torch.Tensor,
        actor: nn.Module,
        log_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute SAC temperature loss."""
        target_entropy = self.config.target_entropy
        if target_entropy is None:
            target_entropy = -self.config.continuous_action_dim / 2
        
        alpha = log_alpha.exp()
        
        with torch.no_grad():
            _, log_probs = actor.sample_with_log_prob(observations)
        
        # Temperature loss
        alpha_loss = -alpha * (log_probs.detach() + target_entropy).mean()
        
        metrics = {
            "temperature_loss": alpha_loss.detach().item(),
            "alpha": alpha.detach().item(),
            "entropy": (-log_probs).mean().detach().item(),
            "target_entropy": target_entropy,
        }
        
        return alpha_loss, metrics
    
    def compute_all_losses(
        self,
        batch: Dict[str, torch.Tensor],
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        log_alpha: torch.Tensor,
        grasp_critic: Optional[nn.Module] = None,
        target_grasp_critic: Optional[nn.Module] = None,
        update_actor: bool = True,
        update_temperature: bool = True,
        update_grasp_critic: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Compute all losses for hybrid SAC agent.
        
        Args:
            batch: Batch with observations, actions, rewards, next_obs, dones, grasp_penalty
            actor: Actor network (continuous actions only)
            critic: Twin critic network (continuous actions)
            target_critic: Target twin critic
            log_alpha: Log temperature parameter
            grasp_critic: Grasp critic network (if learned gripper)
            target_grasp_critic: Target grasp critic
            update_actor: Whether to update actor
            update_temperature: Whether to update temperature
            update_grasp_critic: Whether to update grasp critic
            
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
        
        # Grasp critic loss (if learned gripper)
        if self.config.is_learned_gripper and update_grasp_critic:
            if grasp_critic is None or target_grasp_critic is None:
                raise ValueError("Grasp critic required for learned gripper mode")
            
            grasp_penalties = batch.get("grasp_penalty", torch.zeros_like(rewards))
            
            grasp_loss, grasp_metrics = self.compute_grasp_critic_loss(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
                grasp_penalties=grasp_penalties,
                grasp_critic=grasp_critic,
                target_grasp_critic=target_grasp_critic,
            )
            losses["grasp_critic"] = grasp_loss
            all_metrics.update({f"grasp/{k}": v for k, v in grasp_metrics.items()})
        
        # Actor loss
        if update_actor:
            actor_loss, actor_metrics = self.compute_actor_loss(
                observations=observations,
                actor=actor,
                critic=critic,
                log_alpha=log_alpha,
            )
            losses["actor"] = actor_loss
            all_metrics.update({f"actor/{k}": v for k, v in actor_metrics.items()})
        
        # Temperature loss
        if update_temperature:
            temp_loss, temp_metrics = self.compute_temperature_loss(
                observations=observations,
                actor=actor,
                log_alpha=log_alpha,
            )
            losses["temperature"] = temp_loss
            all_metrics.update({f"temperature/{k}": v for k, v in temp_metrics.items()})
        
        return losses, all_metrics


def select_grasp_action_single_arm(
    grasp_q_values: torch.Tensor,
    argmax: bool = False,
) -> torch.Tensor:
    """Select gripper action for single arm.
    
    Args:
        grasp_q_values: Q-values [B, 3]
        argmax: If True, select best action. If False, sample (epsilon-greedy could be added).
        
    Returns:
        Actions in range {-1, 0, 1}
    """
    action_idx = grasp_q_values.argmax(dim=-1)  # [B]
    return action_idx - 1  # Map {0,1,2} to {-1,0,1}


def select_grasp_action_dual_arm(
    grasp_q_values: torch.Tensor,
    argmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select gripper actions for dual arm.
    
    Args:
        grasp_q_values: Q-values [B, 9]
        argmax: If True, select best action.
        
    Returns:
        Tuple of (action_1, action_2) each in range {-1, 0, 1}
    """
    joint_action_idx = grasp_q_values.argmax(dim=-1)  # [B]
    
    action_idx_1 = joint_action_idx // 3  # {0, 1, 2}
    action_idx_2 = joint_action_idx % 3   # {0, 1, 2}
    
    return action_idx_1 - 1, action_idx_2 - 1

