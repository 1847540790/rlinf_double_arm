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

"""Data utilities for RLinf.

Includes:
    - Replay buffers for off-policy RL
    - Data augmentation for image-based RL (DrQ-style)
    - Tokenizers and datasets
"""

from rlinf.data.augmentations import (
    DrQAugmentation,
    make_batch_augmentation_func,
    random_crop,
    batched_random_crop,
    color_jitter,
    gaussian_blur,
)

__all__ = [
    "DrQAugmentation",
    "make_batch_augmentation_func",
    "random_crop",
    "batched_random_crop",
    "color_jitter",
    "gaussian_blur",
]
