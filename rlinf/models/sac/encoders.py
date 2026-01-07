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

"""Encoder networks for SAC with image observations.

Provides encoders matching HIL-SERL architecture:
    - MLPEncoder: For state/vector observations
    - CNNEncoder: For image observations  
    - ResNet10Encoder: Custom ResNet-10 matching HIL-SERL
    - PreTrainedResNetEncoder: Frozen ResNet with trainable head (resnet-pretrained)
    - SpatialLearnedEmbeddings: Learned spatial pooling layer
"""

import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m: nn.Module):
    """Custom weight initialization for better training stability."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLPEncoder(nn.Module):
    """MLP encoder for state/vector observations.
    
    Matches HIL-SERL MLP configuration with layer norm and tanh activation.
    
    Args:
        input_dim: Dimension of input observations
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of encoder output
        activation: Activation function ("relu", "tanh", "elu")
        use_layer_norm: Whether to use layer normalization (HIL-SERL default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 256,
        activation: str = "tanh",  # HIL-SERL default
        use_layer_norm: bool = True,  # HIL-SERL default
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(self._get_activation(activation))
        
        self.encoder = nn.Sequential(*layers)
        self.apply(weight_init)
        
    def _get_activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "elu":
            return nn.ELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialLearnedEmbeddings(nn.Module):
    """Spatial learned embeddings pooling layer (matching HIL-SERL).
    
    Learns a set of spatial embeddings that are combined with feature maps
    to produce a compact representation.
    
    Args:
        height: Height of input feature map
        width: Width of input feature map  
        channel: Number of channels in input feature map
        num_features: Number of spatial embedding features (default: 8)
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        num_features: int = 8,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        
        # Learnable spatial kernel
        self.kernel = nn.Parameter(
            torch.randn(height, width, channel, num_features) * 0.02
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, C, H, W) or (B, H, W, C)
            
        Returns:
            Tensor of shape (B, channel * num_features)
        """
        # Convert to (B, H, W, C) if needed
        if features.dim() == 4 and features.shape[1] == self.channel:
            features = features.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            
        batch_size = features.shape[0]
        
        # Compute spatial embeddings: sum over H, W dimensions
        # features: (B, H, W, C), kernel: (H, W, C, F)
        # result: (B, C, F) after summing over H, W
        features_expanded = features.unsqueeze(-1)  # (B, H, W, C, 1)
        kernel_expanded = self.kernel.unsqueeze(0)  # (1, H, W, C, F)
        
        result = (features_expanded * kernel_expanded).sum(dim=(1, 2))  # (B, C, F)
        result = result.reshape(batch_size, -1)  # (B, C * F)
        
        return result


class SpatialSoftmax(nn.Module):
    """Spatial softmax pooling layer.
    
    Computes expected 2D positions for each feature channel.
    
    Args:
        height: Height of input feature map
        width: Width of input feature map
        channel: Number of channels
        temperature: Softmax temperature (default: 1.0)
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = temperature
        
        # Create position grids
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, C, H, W)
            
        Returns:
            Tensor of shape (B, 2 * C) - expected x, y positions per channel
        """
        batch_size, num_channels, h, w = features.shape
        
        # Reshape to (B, C, H*W)
        features_flat = features.view(batch_size, num_channels, -1)
        
        # Softmax attention
        attention = F.softmax(features_flat / self.temperature, dim=-1)
        
        # Expected positions
        expected_x = (attention * self.pos_x).sum(dim=-1)  # (B, C)
        expected_y = (attention * self.pos_y).sum(dim=-1)  # (B, C)
        
        return torch.cat([expected_x, expected_y], dim=-1)  # (B, 2*C)


class ResNetBlock(nn.Module):
    """ResNet basic block (matching HIL-SERL)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_groups: int = 4,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(norm_groups, out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(norm_groups, out_channels)
        
        # Projection for residual connection if dimensions change
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(norm_groups, out_channels),
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.proj is not None:
            residual = self.proj(x)
            
        out = out + residual
        out = self.relu(out)
        
        return out


class ResNet10Encoder(nn.Module):
    """ResNet-10 encoder matching HIL-SERL architecture.
    
    Uses stage_sizes = (1, 1, 1, 1) - one block per stage.
    Supports frozen backbone mode for resnet-pretrained.
    
    Args:
        image_shape: Input image shape (C, H, W)
        num_filters: Base number of filters (default: 64)
        pooling_method: Pooling method ("avg", "spatial_learned_embeddings", "spatial_softmax")
        num_spatial_blocks: Number of spatial features for learned embeddings
        bottleneck_dim: Output dimension after bottleneck (default: 256)
        freeze_backbone: Whether to freeze backbone weights
        pre_pooling: If True, returns features before pooling (for PreTrainedResNetEncoder)
    """
    
    # Stage sizes for ResNet-10: 1 block per stage
    STAGE_SIZES = (1, 1, 1, 1)
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        num_filters: int = 64,
        pooling_method: str = "spatial_learned_embeddings",
        num_spatial_blocks: int = 8,
        bottleneck_dim: int = 256,
        freeze_backbone: bool = False,
        pre_pooling: bool = False,
        norm_groups: int = 4,
    ):
        super().__init__()
        
        self.image_shape = image_shape
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        self.pre_pooling = pre_pooling
        
        in_channels = image_shape[0]
        
        # Initial convolution (matching HIL-SERL: 7x7, stride 2)
        self.conv1 = nn.Conv2d(
            in_channels, num_filters,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.gn1 = nn.GroupNorm(norm_groups, num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        self.stage1 = self._make_stage(num_filters, num_filters, self.STAGE_SIZES[0], stride=1, norm_groups=norm_groups)
        self.stage2 = self._make_stage(num_filters, num_filters * 2, self.STAGE_SIZES[1], stride=2, norm_groups=norm_groups)
        self.stage3 = self._make_stage(num_filters * 2, num_filters * 4, self.STAGE_SIZES[2], stride=2, norm_groups=norm_groups)
        self.stage4 = self._make_stage(num_filters * 4, num_filters * 8, self.STAGE_SIZES[3], stride=2, norm_groups=norm_groups)
        
        # Compute feature map size after backbone
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            feat = self._forward_backbone(dummy)
            self.feature_channels = feat.shape[1]
            self.feature_height = feat.shape[2]
            self.feature_width = feat.shape[3]
            
        # Pooling layer
        if not pre_pooling:
            self._build_pooling_head()
            
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            
        # Initialize weights
        self.apply(weight_init)
        
    def _make_stage(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int,
        norm_groups: int,
    ) -> nn.Sequential:
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, norm_groups))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, norm_groups))
        return nn.Sequential(*layers)
        
    def _build_pooling_head(self):
        if self.pooling_method == "spatial_learned_embeddings":
            self.pool = SpatialLearnedEmbeddings(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
                num_features=self.num_spatial_blocks,
            )
            pool_out_dim = self.feature_channels * self.num_spatial_blocks
            self.dropout = nn.Dropout(0.1)
        elif self.pooling_method == "spatial_softmax":
            self.pool = SpatialSoftmax(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
            )
            pool_out_dim = 2 * self.feature_channels
            self.dropout = None
        elif self.pooling_method == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pool_out_dim = self.feature_channels
            self.dropout = None
        elif self.pooling_method == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            pool_out_dim = self.feature_channels
            self.dropout = None
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
            
        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(pool_out_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.Tanh(),
        )
        
    def _freeze_backbone(self):
        for name, param in self.named_parameters():
            if not any(x in name for x in ['pool', 'bottleneck', 'dropout']):
                param.requires_grad = False
                
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone only."""
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Image tensor of shape (B, C, H, W)
            
        Returns:
            If pre_pooling: features of shape (B, C, H, W)
            Else: encoded features of shape (B, bottleneck_dim)
        """
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Forward through backbone
        x = self._forward_backbone(x)
        
        if self.pre_pooling:
            return x
            
        # Pooling
        if self.pooling_method in ["avg", "max"]:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        else:
            # For spatial_learned_embeddings, need to permute to (B, H, W, C)
            x = x.permute(0, 2, 3, 1)
            x = self.pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
                
        # Bottleneck
        x = self.bottleneck(x)
        
        return x


class PreTrainedResNetEncoder(nn.Module):
    """Pre-trained ResNet encoder matching HIL-SERL's resnet-pretrained.
    
    Uses a frozen ResNet-10 backbone with trainable pooling head.
    
    Args:
        image_shape: Input image shape (C, H, W)
        bottleneck_dim: Output dimension (default: 256)
        pooling_method: Pooling method (default: "spatial_learned_embeddings")
        num_spatial_blocks: Number of spatial features (default: 8)
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        bottleneck_dim: int = 256,
        pooling_method: str = "spatial_learned_embeddings",
        num_spatial_blocks: int = 8,
    ):
        super().__init__()
        
        self.image_shape = image_shape
        self.bottleneck_dim = bottleneck_dim
        self.pooling_method = pooling_method
        
        # Frozen backbone (pre_pooling=True returns features before pooling)
        self.backbone = ResNet10Encoder(
            image_shape=image_shape,
            pooling_method=pooling_method,
            num_spatial_blocks=num_spatial_blocks,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=True,
            pre_pooling=True,  # Return features before pooling
        )
        
        # Get feature dimensions
        self.feature_channels = self.backbone.feature_channels
        self.feature_height = self.backbone.feature_height
        self.feature_width = self.backbone.feature_width
        
        # Trainable pooling head
        if pooling_method == "spatial_learned_embeddings":
            self.pool = SpatialLearnedEmbeddings(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
                num_features=num_spatial_blocks,
            )
            pool_out_dim = self.feature_channels * num_spatial_blocks
            self.dropout = nn.Dropout(0.1)
        elif pooling_method == "spatial_softmax":
            self.pool = SpatialSoftmax(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
            )
            pool_out_dim = 2 * self.feature_channels
            self.dropout = None
        elif pooling_method == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pool_out_dim = self.feature_channels
            self.dropout = None
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
            
        # Trainable bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(pool_out_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh(),
        )
        
    @property
    def output_dim(self) -> int:
        return self.bottleneck_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Image tensor of shape (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Encoded features of shape (B, bottleneck_dim)
        """
        # Handle frame stacking - use only the latest frame
        if x.dim() == 5:
            x = x[:, -1]  # Take latest frame
            
        # Forward through frozen backbone
        with torch.no_grad():
            features = self.backbone(x)
            
        # Pooling (trainable)
        if self.pooling_method in ["avg", "max"]:
            x = self.pool(features)
            x = x.view(x.size(0), -1)
        else:
            # For spatial methods, permute to (B, H, W, C)
            x = features.permute(0, 2, 3, 1)
            x = self.pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
                
        # Bottleneck
        x = self.bottleneck(x)
        
        return x


# Legacy CNNEncoder for compatibility
class CNNEncoder(nn.Module):
    """CNN encoder for image observations.
    
    Uses a simple CNN architecture similar to DrQ/RAD.
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        output_dim: int = 256,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.image_shape = image_shape
        self.output_dim = output_dim
        
        layers = []
        in_channels = image_shape[0]
        
        for i in range(num_layers):
            out_channels = num_filters * (2 ** min(i, 2))
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=2 if i < 3 else 1,
                padding=kernel_size // 2,
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels
            
        self.conv = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)
            
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )
        
        self.apply(weight_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)
            
        if x.max() > 1.0:
            x = x / 255.0
            
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


# Legacy ResNetEncoder using torchvision (for fallback)
class ResNetEncoder(nn.Module):
    """Pre-trained ResNet encoder using torchvision.
    
    For compatibility - use PreTrainedResNetEncoder for HIL-SERL matching.
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        output_dim: int = 256,
        resnet_type: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        try:
            import torchvision.models as models
            
            if resnet_type == "resnet18":
                resnet = models.resnet18(pretrained=pretrained)
                backbone_dim = 512
            elif resnet_type == "resnet34":
                resnet = models.resnet34(pretrained=pretrained)
                backbone_dim = 512
            elif resnet_type == "resnet50":
                resnet = models.resnet50(pretrained=pretrained)
                backbone_dim = 2048
            else:
                raise ValueError(f"Unknown ResNet type: {resnet_type}")
                
        except ImportError:
            raise ImportError("torchvision is required for ResNetEncoder")
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, -1]
            
        if x.max() > 1.0:
            x = x / 255.0
            
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        h = self.backbone(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        
        return self.projection(h)


class MultiModalEncoder(nn.Module):
    """Encoder for multi-modal observations (images + state).
    
    Matches HIL-SERL's EncodingWrapper functionality.
    
    Args:
        image_keys: Keys for image observations
        image_shape: Shape of images (C, H, W)
        state_dim: Dimension of state observations (proprio)
        output_dim: Dimension of combined output
        encoder_type: Type of image encoder ("cnn", "resnet", "resnet-pretrained")
        use_proprio: Whether to include proprioceptive state
    """
    
    def __init__(
        self,
        image_keys: Tuple[str, ...] = ("image",),
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        state_dim: int = 0,
        output_dim: int = 256,
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = True,
    ):
        super().__init__()
        
        self.image_keys = image_keys
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.use_proprio = use_proprio
        
        # Create image encoder for each key
        self.image_encoders = nn.ModuleDict()
        
        for key in image_keys:
            if encoder_type == "cnn":
                self.image_encoders[key] = CNNEncoder(
                    image_shape=image_shape,
                    output_dim=output_dim,
                )
            elif encoder_type == "resnet":
                self.image_encoders[key] = ResNet10Encoder(
                    image_shape=image_shape,
                    bottleneck_dim=output_dim,
                    pooling_method="spatial_learned_embeddings",
                    freeze_backbone=False,
                )
            elif encoder_type == "resnet-pretrained":
                self.image_encoders[key] = PreTrainedResNetEncoder(
                    image_shape=image_shape,
                    bottleneck_dim=output_dim,
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
                
        # Combine features
        combined_dim = len(image_keys) * output_dim
        if use_proprio and state_dim > 0:
            combined_dim += state_dim
            
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )
        
    def forward(
        self, 
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            observations: Dictionary with image and state observations
            
        Returns:
            Combined encoding of shape (B, output_dim)
        """
        features = []
        
        # Encode images
        for key in self.image_keys:
            if key in observations:
                img_feat = self.image_encoders[key](observations[key])
                features.append(img_feat)
                
        # Add state features
        if self.use_proprio and "state" in observations:
            state = observations["state"]
            if isinstance(state, dict):
                state_parts = []
                for v in state.values():
                    if torch.is_tensor(v):
                        state_parts.append(v.view(v.size(0), -1))
                state = torch.cat(state_parts, dim=-1)
            features.append(state)
            
        combined = torch.cat(features, dim=-1)
        return self.combiner(combined)


# Encoder registry for easy instantiation
ENCODER_REGISTRY = {
    "mlp": MLPEncoder,
    "cnn": CNNEncoder,
    "resnet": ResNet10Encoder,
    "resnet-pretrained": PreTrainedResNetEncoder,
    "resnet-torchvision": ResNetEncoder,
}


def create_encoder(encoder_type: str, **kwargs):
    """Create an encoder by type.
    
    Args:
        encoder_type: Type of encoder
        **kwargs: Encoder-specific arguments
        
    Returns:
        Encoder module
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[encoder_type](**kwargs)
