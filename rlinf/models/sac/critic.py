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

"""Critic networks for SAC.

Provides Q-function networks for value estimation:
    - QCritic: Single Q-function
    - TwinQCritic: Twin Q-functions (standard SAC)
    - EnsembleQCritic: Ensemble of Q-functions (REDQ-style)
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class QCritic(nn.Module):
    """Single Q-function network.
    
    Q(s, a) -> scalar value estimate
    
    Args:
        obs_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dims: Hidden layer dimensions
        encoder: Optional encoder for observations
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = encoder
        
        # Determine input dimension
        if encoder is not None:
            in_dim = encoder.output_dim + action_dim
        else:
            in_dim = obs_dim + action_dim
            
        # Build MLP
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))
        
        self.q_net = nn.Sequential(*layers)
        
        # Initialize last layer with small weights
        nn.init.uniform_(self.q_net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_net[-1].bias, -3e-3, 3e-3)
        
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs: Observations
            action: Actions
            
        Returns:
            Q-values of shape (B,)
        """
        # Encode observations
        if self.encoder is not None:
            if isinstance(obs, dict):
                h = self.encoder(obs)
            else:
                h = self.encoder(obs)
        else:
            h = obs
            
        # Concatenate with action
        x = torch.cat([h, action], dim=-1)
        q = self.q_net(x).squeeze(-1)
        
        return q


class TwinQCritic(nn.Module):
    """Twin Q-function network for SAC.
    
    Uses two independent Q-functions to reduce overestimation bias.
    The minimum of the two Q-values is used for the target.
    
    Args:
        obs_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dims: Hidden layer dimensions
        encoder: Optional encoder for observations (shared or separate)
        share_encoder: Whether to share encoder between Q1 and Q2
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        encoder: Optional[nn.Module] = None,
        share_encoder: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.share_encoder = share_encoder
        
        # Create Q-networks
        self.q1 = QCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
        )
        
        if share_encoder and encoder is not None:
            # Share encoder between Q1 and Q2
            encoder2 = encoder
        else:
            # Create separate encoder for Q2
            encoder2 = copy.deepcopy(encoder) if encoder is not None else None
            
        self.q2 = QCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder2,
        )
        
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q-values.
        
        Args:
            obs: Observations
            action: Actions
            
        Returns:
            Tuple of (Q1, Q2) values, each of shape (B,)
        """
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2
        
    def q1_forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for Q1 only."""
        return self.q1(obs, action)
        
    def min_q(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum of twin Q-values."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class EnsembleQCritic(nn.Module):
    """Ensemble of Q-functions.
    
    Uses multiple Q-functions for uncertainty estimation or
    REDQ-style training with random subsampling.
    
    Args:
        obs_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dims: Hidden layer dimensions
        encoder: Optional encoder for observations
        ensemble_size: Number of Q-functions in ensemble
        subsample_size: Number of Q-functions to subsample for target (REDQ)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        encoder: Optional[nn.Module] = None,
        ensemble_size: int = 10,
        subsample_size: int = 2,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        
        # Create ensemble
        self.critics = nn.ModuleList([
            QCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                encoder=copy.deepcopy(encoder) if encoder else None,
            )
            for _ in range(ensemble_size)
        ])
        
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning all Q-values.
        
        Args:
            obs: Observations
            action: Actions
            
        Returns:
            Q-values of shape (ensemble_size, B)
        """
        q_values = [critic(obs, action) for critic in self.critics]
        return torch.stack(q_values, dim=0)
        
    def min_q(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value across ensemble."""
        q_values = self.forward(obs, action)
        return q_values.min(dim=0)[0]
        
    def mean_q(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get mean Q-value across ensemble."""
        q_values = self.forward(obs, action)
        return q_values.mean(dim=0)
        
    def subsample_min_q(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value from random subsample (REDQ-style)."""
        q_values = self.forward(obs, action)
        
        # Random subsample
        indices = torch.randperm(self.ensemble_size)[:self.subsample_size]
        q_subset = q_values[indices]
        
        return q_subset.min(dim=0)[0]


def create_critic(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int] = [256, 256],
    encoder: Optional[nn.Module] = None,
    critic_type: str = "twin",
    **kwargs,
) -> nn.Module:
    """Factory function to create a critic network.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        encoder: Optional observation encoder
        critic_type: Type of critic ("single", "twin", "ensemble")
        **kwargs: Additional arguments passed to critic
        
    Returns:
        Critic network
    """
    if critic_type == "single":
        return QCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
        )
    elif critic_type == "twin":
        return TwinQCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
            **kwargs,
        )
    elif critic_type == "ensemble":
        return EnsembleQCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown critic type: {critic_type}")

