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

"""Data augmentation utilities for image-based RL (DrQ-style).

Ported from HIL-SERL's data_augmentations.py (JAX) to PyTorch.

Supports:
    - Random crop (with edge padding)
    - Color jitter (brightness, contrast, saturation, hue)
    - Gaussian blur
    - Random flip
    - Grayscale conversion

Usage:
    augment_fn = make_batch_augmentation_func(image_keys=["wrist_1", "wrist_2"])
    augmented_batch = augment_fn(batch)
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def random_crop(
    img: torch.Tensor,
    padding: int = 4,
) -> torch.Tensor:
    """Apply random crop with edge padding.
    
    Args:
        img: Image tensor of shape (C, H, W) or (H, W, C)
        padding: Padding size on each side
        
    Returns:
        Cropped image of same shape as input
    """
    # Handle both CHW and HWC formats
    if img.dim() == 3:
        if img.shape[0] in [1, 3, 4]:  # CHW format
            c, h, w = img.shape
            is_chw = True
        else:  # HWC format
            h, w, c = img.shape
            is_chw = False
            img = img.permute(2, 0, 1)  # Convert to CHW
    else:
        raise ValueError(f"Expected 3D tensor, got shape {img.shape}")
    
    # Pad with edge replication
    padded = F.pad(img, (padding, padding, padding, padding), mode='replicate')
    
    # Random crop back to original size
    crop_y = torch.randint(0, 2 * padding + 1, (1,)).item()
    crop_x = torch.randint(0, 2 * padding + 1, (1,)).item()
    cropped = padded[:, crop_y:crop_y + h, crop_x:crop_x + w]
    
    # Convert back to original format
    if not is_chw:
        cropped = cropped.permute(1, 2, 0)
    
    return cropped


def batched_random_crop(
    images: torch.Tensor,
    padding: int = 4,
) -> torch.Tensor:
    """Apply random crop to a batch of images.
    
    Args:
        images: Batch of images (B, C, H, W) or (B, H, W, C)
        padding: Padding size
        
    Returns:
        Cropped images of same shape
    """
    batch_size = images.shape[0]
    results = []
    
    for i in range(batch_size):
        results.append(random_crop(images[i], padding))
    
    return torch.stack(results)


def adjust_brightness(
    img: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Adjust brightness by adding delta to all pixels."""
    return img + delta


def adjust_contrast(
    img: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Adjust contrast by scaling around mean."""
    # img shape: (C, H, W) or (B, C, H, W)
    if img.dim() == 3:
        mean = img.mean(dim=(-2, -1), keepdim=True)
    else:
        mean = img.mean(dim=(-2, -1), keepdim=True)
    return factor * (img - mean) + mean


def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB to HSV color space."""
    v = torch.maximum(torch.maximum(r, g), b)
    range_ = v - torch.minimum(torch.minimum(r, g), b)
    
    s = torch.where(v > 0, range_ / (v + 1e-8), torch.zeros_like(v))
    norm = torch.where(range_ != 0, 1.0 / (6.0 * range_ + 1e-8), torch.full_like(range_, 1e9))
    
    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0
    
    h = torch.where(r == v, hr, torch.where(g == v, hg, hb))
    h = h * (range_ > 0).float()
    h = h + (h < 0).float()
    
    return h, s, v


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert HSV to RGB color space."""
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - torch.abs(fmodu - 1))
    hcat = torch.floor(dh).long()
    
    rr = torch.where(
        (hcat == 0) | (hcat == 5), c,
        torch.where((hcat == 1) | (hcat == 4), x, torch.zeros_like(c))
    ) + m
    gg = torch.where(
        (hcat == 1) | (hcat == 2), c,
        torch.where((hcat == 0) | (hcat == 3), x, torch.zeros_like(c))
    ) + m
    bb = torch.where(
        (hcat == 3) | (hcat == 4), c,
        torch.where((hcat == 2) | (hcat == 5), x, torch.zeros_like(c))
    ) + m
    
    return rr, gg, bb


def adjust_saturation(
    img: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Adjust saturation by factor."""
    # img shape: (C, H, W) with C=3
    r, g, b = img[0], img[1], img[2]
    h, s, v = rgb_to_hsv(r, g, b)
    s = torch.clamp(s * factor, 0.0, 1.0)
    rr, gg, bb = hsv_to_rgb(h, s, v)
    return torch.stack([rr, gg, bb], dim=0)


def adjust_hue(
    img: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Adjust hue by delta."""
    r, g, b = img[0], img[1], img[2]
    h, s, v = rgb_to_hsv(r, g, b)
    h = (h + delta) % 1.0
    rr, gg, bb = hsv_to_rgb(h, s, v)
    return torch.stack([rr, gg, bb], dim=0)


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB image to grayscale (3 channels)."""
    # img shape: (C, H, W) or (B, C, H, W)
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img.device)
    
    if img.dim() == 3:
        gray = torch.tensordot(img.permute(1, 2, 0), rgb_weights, dims=1)
        return gray.unsqueeze(0).repeat(3, 1, 1)
    else:
        gray = torch.tensordot(img.permute(0, 2, 3, 1), rgb_weights, dims=1)
        return gray.unsqueeze(1).repeat(1, 3, 1, 1)


def gaussian_blur(
    img: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian blur.
    
    Args:
        img: Image tensor of shape (C, H, W) or (B, C, H, W)
        kernel_size: Size of blur kernel (will be made odd)
        sigma: Standard deviation of Gaussian
        
    Returns:
        Blurred image
    """
    # Ensure odd kernel size
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=img.device).float() - kernel_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel via outer product
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    
    # Add batch and channel dimensions
    was_3d = img.dim() == 3
    if was_3d:
        img = img.unsqueeze(0)
    
    channels = img.shape[1]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    # Apply depthwise convolution
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding, groups=channels)
    
    if was_3d:
        blurred = blurred.squeeze(0)
    
    return blurred


def color_jitter(
    img: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    apply_prob: float = 0.8,
) -> torch.Tensor:
    """Apply random color jittering.
    
    Args:
        img: Image tensor (C, H, W), values in [0, 1]
        brightness: Max brightness change
        contrast: Max contrast change  
        saturation: Max saturation change
        hue: Max hue change
        apply_prob: Probability of applying any jitter
        
    Returns:
        Color-jittered image
    """
    if torch.rand(1).item() > apply_prob:
        return img
    
    # Random order of transforms
    order = torch.randperm(4)
    
    for idx in order:
        if idx == 0 and brightness > 0:
            delta = torch.empty(1).uniform_(-brightness, brightness).item()
            img = adjust_brightness(img, delta)
        elif idx == 1 and contrast > 0:
            factor = torch.empty(1).uniform_(1 - contrast, 1 + contrast).item()
            img = adjust_contrast(img, factor)
        elif idx == 2 and saturation > 0:
            factor = torch.empty(1).uniform_(1 - saturation, 1 + saturation).item()
            img = adjust_saturation(img, factor)
        elif idx == 3 and hue > 0:
            delta = torch.empty(1).uniform_(-hue, hue).item()
            img = adjust_hue(img, delta)
    
    return torch.clamp(img, 0.0, 1.0)


def random_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Random horizontal flip."""
    if torch.rand(1).item() < p:
        return torch.flip(img, dims=[-1])
    return img


class DrQAugmentation:
    """DrQ-style data augmentation for image-based RL.
    
    Applies random crop (with edge padding) to images.
    This is the primary augmentation used in DrQ and HIL-SERL.
    
    Args:
        image_keys: List of image keys in observation dict
        padding: Padding size for random crop
        enable_color_jitter: Whether to apply color jitter
    """
    
    def __init__(
        self,
        image_keys: List[str] = ["image"],
        padding: int = 4,
        enable_color_jitter: bool = False,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.image_keys = image_keys
        self.padding = padding
        self.enable_color_jitter = enable_color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to batch.
        
        Args:
            batch: Dictionary containing observations with image keys
            
        Returns:
            Augmented batch
        """
        augmented = {}
        
        for key, value in batch.items():
            if key == "observations" or key == "next_observations":
                # Handle observation dicts
                if isinstance(value, dict):
                    aug_obs = {}
                    for obs_key, obs_value in value.items():
                        if obs_key in self.image_keys and torch.is_tensor(obs_value):
                            aug_obs[obs_key] = self._augment_images(obs_value)
                        else:
                            aug_obs[obs_key] = obs_value
                    augmented[key] = aug_obs
                else:
                    augmented[key] = value
            else:
                augmented[key] = value
        
        return augmented
    
    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """Augment a batch of images."""
        # images shape: (B, C, H, W) or (B, T, C, H, W)
        
        if images.dim() == 5:
            # Frame stacking: (B, T, C, H, W)
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            augmented = self._augment_batch(images_flat)
            return augmented.view(B, T, C, H, W)
        else:
            return self._augment_batch(images)
    
    def _augment_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Augment a batch of images (B, C, H, W)."""
        # Random crop
        augmented = batched_random_crop(images, self.padding)
        
        # Optional color jitter
        if self.enable_color_jitter:
            results = []
            for i in range(augmented.shape[0]):
                results.append(color_jitter(
                    augmented[i],
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                ))
            augmented = torch.stack(results)
        
        return augmented


def make_batch_augmentation_func(
    image_keys: List[str] = ["image"],
    padding: int = 4,
    enable_color_jitter: bool = False,
) -> Callable:
    """Create a batch augmentation function.
    
    Args:
        image_keys: List of image keys in observation dict
        padding: Padding for random crop
        enable_color_jitter: Whether to apply color jitter
        
    Returns:
        Callable that augments a batch
    """
    augmenter = DrQAugmentation(
        image_keys=image_keys,
        padding=padding,
        enable_color_jitter=enable_color_jitter,
    )
    return augmenter

