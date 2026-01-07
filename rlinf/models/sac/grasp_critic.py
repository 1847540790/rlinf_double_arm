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

"""Grasp Critic network for SAC Hybrid agent (DQN-style gripper control).

Ported from HIL-SERL's GraspCritic (JAX) to PyTorch.

The grasp critic outputs Q-values for discrete gripper actions:
    - Single arm: 3 actions (open, hold, close)
    - Dual arm: 9 joint actions (3 x 3)

Usage:
    grasp_critic = GraspCritic(
        obs_dim=256,  # Encoder output dim
        hidden_dims=[128, 128],
        num_actions=3,  # 3 for single arm, 9 for dual arm
    )
    
    q_values = grasp_critic(encoded_obs)  # [B, num_actions]
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspCritic(nn.Module):
    """DQN-style critic for discrete gripper actions.
    
    Outputs Q-values for all possible gripper actions given observations.
    
    Args:
        obs_dim: Dimension of encoded observations
        hidden_dims: Hidden layer dimensions
        num_actions: Number of discrete actions (3 for single arm, 9 for dual)
        activation: Activation function ("tanh", "relu")
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [128, 128],
        num_actions: int = 3,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # Build network
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs: Encoded observations [B, obs_dim]
            
        Returns:
            Q-values for all actions [B, num_actions]
        """
        return self.network(obs)
    
    def get_best_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get greedy action (argmax Q).
        
        Args:
            obs: Encoded observations [B, obs_dim]
            
        Returns:
            Action indices [B]
        """
        q_values = self.forward(obs)
        return q_values.argmax(dim=-1)


class GraspCriticWithEncoder(nn.Module):
    """Grasp critic with image encoder.
    
    Combines an image encoder with the grasp critic network.
    
    Args:
        encoder: Image/observation encoder
        hidden_dims: Hidden layer dimensions for Q-network
        num_actions: Number of discrete actions
        activation: Activation function
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dims: List[int] = [128, 128],
        num_actions: int = 3,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.num_actions = num_actions
        
        # Get encoder output dim
        encoder_out_dim = getattr(encoder, 'output_dim', 256)
        
        # Q-network head
        self.q_head = GraspCritic(
            obs_dim=encoder_out_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
    
    def forward(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            observations: Raw observations (images + state)
            
        Returns:
            Q-values for all actions [B, num_actions]
        """
        encoded = self.encoder(observations)
        return self.q_head(encoded)
    
    def get_best_action(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Get greedy action.
        
        Args:
            observations: Raw observations
            
        Returns:
            Action indices [B]
        """
        q_values = self.forward(observations)
        return q_values.argmax(dim=-1)


def create_grasp_critic(
    encoder: nn.Module,
    setup_mode: str = "single-arm-learned-gripper",
    hidden_dims: List[int] = [128, 128],
    activation: str = "tanh",
    use_layer_norm: bool = True,
) -> GraspCriticWithEncoder:
    """Create a grasp critic for the specified setup mode.
    
    Args:
        encoder: Observation encoder
        setup_mode: Robot setup mode
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        use_layer_norm: Whether to use layer norm
        
    Returns:
        GraspCriticWithEncoder instance
    """
    if "dual-arm" in setup_mode:
        num_actions = 9  # 3 x 3 joint actions
    else:
        num_actions = 3  # open, hold, close
    
    return GraspCriticWithEncoder(
        encoder=encoder,
        hidden_dims=hidden_dims,
        num_actions=num_actions,
        activation=activation,
        use_layer_norm=use_layer_norm,
    )

