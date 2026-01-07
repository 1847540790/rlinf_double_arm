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

"""Actor networks for SAC.

Provides stochastic policy networks that output action distributions:
    - GaussianActor: Outputs unbounded Gaussian actions
    - SquashedGaussianActor: Outputs tanh-squashed Gaussian actions (standard SAC)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rlinf.models.sac.encoders import MLPEncoder, MultiModalEncoder


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """Gaussian policy network for continuous actions.
    
    Outputs a Gaussian distribution over actions given observations.
    Actions are NOT squashed (unbounded).
    
    Args:
        obs_dim: Dimension of observations (or encoder output)
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
            in_dim = encoder.output_dim
        else:
            in_dim = obs_dim
            
        # Build MLP
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        self.trunk = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        
        # Initialize output layers with small weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        
    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std.
        
        Args:
            obs: Observations
            
        Returns:
            Tuple of (mean, log_std)
        """
        # Encode observations if encoder provided
        if self.encoder is not None:
            if isinstance(obs, dict):
                h = self.encoder(obs)
            else:
                h = self.encoder(obs)
        else:
            h = obs
            
        h = self.trunk(h)
        
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
        
    def get_distribution(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Normal:
        """Get action distribution.
        
        Args:
            obs: Observations
            
        Returns:
            Normal distribution over actions
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)
        
    def sample(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample actions.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action
            
        Returns:
            Sampled actions
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return mean
            
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist.rsample()
        
    def sample_with_log_prob(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.
        
        Args:
            obs: Observations
            
        Returns:
            Tuple of (actions, log_probs)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian policy for SAC.
    
    Outputs a tanh-squashed Gaussian distribution, which bounds actions
    to [-1, 1]. This is the standard policy for SAC.
    
    The log probability is corrected for the tanh squashing transformation.
    
    Args:
        obs_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dims: Hidden layer dimensions
        encoder: Optional encoder for observations
        action_scale: Scale for output actions (default 1.0)
        action_bias: Bias for output actions (default 0.0)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        encoder: Optional[nn.Module] = None,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = encoder
        self.action_scale = action_scale
        self.action_bias = action_bias
        
        # Determine input dimension
        if encoder is not None:
            in_dim = encoder.output_dim
        else:
            in_dim = obs_dim
            
        # Build MLP
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        self.trunk = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        
        # Initialize output layers
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std.
        
        Args:
            obs: Observations
            
        Returns:
            Tuple of (mean, log_std)
        """
        # Encode observations
        if self.encoder is not None:
            if isinstance(obs, dict):
                h = self.encoder(obs)
            else:
                h = self.encoder(obs)
        else:
            h = obs
            
        h = self.trunk(h)
        
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
        
    def sample(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample actions with tanh squashing.
        
        Args:
            obs: Observations
            deterministic: If True, return tanh(mean)
            
        Returns:
            Squashed actions in [-action_scale, action_scale]
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            actions = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            x = dist.rsample()
            actions = torch.tanh(x)
            
        return actions * self.action_scale + self.action_bias
        
    def sample_with_log_prob(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.
        
        Includes the correction for tanh squashing:
            log pi(a|s) = log mu(u|s) - sum(log(1 - tanh^2(u)))
        where u is the pre-squashing action.
        
        Args:
            obs: Observations
            
        Returns:
            Tuple of (actions, log_probs)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Sample from Gaussian
        dist = Normal(mean, std)
        x = dist.rsample()
        
        # Apply tanh squashing
        actions = torch.tanh(x)
        
        # Compute log probability with tanh correction
        # log pi(a|s) = log mu(u|s) - sum(log(1 - tanh^2(u)))
        log_probs = dist.log_prob(x)
        
        # Tanh correction: log(1 - tanh^2(x)) = log(1 - a^2)
        # Using numerically stable version
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)
        
        # Scale actions
        actions = actions * self.action_scale + self.action_bias
        
        return actions, log_probs
        
    def log_prob(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of given actions.
        
        Args:
            obs: Observations
            actions: Actions (already squashed)
            
        Returns:
            Log probabilities
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Undo action scaling
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        
        # Compute pre-tanh values using atanh
        # Clamp to avoid numerical issues at boundaries
        actions_clipped = torch.clamp(actions_unscaled, -0.999, 0.999)
        x = torch.atanh(actions_clipped)
        
        # Gaussian log prob
        dist = Normal(mean, std)
        log_probs = dist.log_prob(x)
        
        # Tanh correction
        log_probs = log_probs - torch.log(1 - actions_unscaled.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)
        
        return log_probs


def create_actor(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int] = [256, 256],
    encoder: Optional[nn.Module] = None,
    squashed: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to create an actor network.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        encoder: Optional observation encoder
        squashed: Whether to use tanh squashing (default True for SAC)
        **kwargs: Additional arguments passed to actor
        
    Returns:
        Actor network
    """
    if squashed:
        return SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
            **kwargs,
        )
    else:
        return GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder=encoder,
        )

