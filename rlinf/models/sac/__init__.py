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

"""SAC model components.

Includes models for:
    - Standard SAC (GaussianActor, TwinQCritic)
    - SAC Hybrid (with GraspCritic for discrete gripper actions)
    - Various encoders (MLP, CNN, ResNet-10, ResNet-pretrained)
"""

from rlinf.models.sac.actor import GaussianActor, SquashedGaussianActor
from rlinf.models.sac.critic import TwinQCritic, QCritic, EnsembleQCritic
from rlinf.models.sac.encoders import (
    MLPEncoder,
    CNNEncoder,
    ResNetEncoder,
    ResNet10Encoder,
    PreTrainedResNetEncoder,
    MultiModalEncoder,
    SpatialLearnedEmbeddings,
    create_encoder,
    ENCODER_REGISTRY,
)
from rlinf.models.sac.grasp_critic import (
    GraspCritic,
    GraspCriticWithEncoder,
    create_grasp_critic,
)

__all__ = [
    # Actor
    "GaussianActor",
    "SquashedGaussianActor",
    # Critic
    "TwinQCritic",
    "QCritic",
    "EnsembleQCritic",
    # Grasp Critic (for hybrid SAC)
    "GraspCritic",
    "GraspCriticWithEncoder",
    "create_grasp_critic",
    # Encoders
    "MLPEncoder",
    "CNNEncoder",
    "ResNetEncoder",
    "ResNet10Encoder",
    "PreTrainedResNetEncoder",
    "MultiModalEncoder",
    "SpatialLearnedEmbeddings",
    "create_encoder",
    "ENCODER_REGISTRY",
]

