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
"""
Dual Arm Policy Transforms for OpenPi models.

This module provides data transforms for dual arm robot setups with:
- Left wrist camera
- Right wrist camera
- Optional base/table camera
- 14-dimensional state (7 dims per arm: pos + rot + gripper)
"""
import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_dual_arm_example() -> dict:
    """Creates a random input example for the dual arm policy."""
    return {
        "left_wrist_images": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "right_wrist_images": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "states": np.random.rand(14),  # 7 dims per arm
        "task_descriptions": "pick up the object with both arms",
    }


def _parse_image(image) -> np.ndarray:
    """
    Parse image to uint8 (H,W,C) format.
    Handles torch tensors (C,H,W or H,W,C) and numpy arrays.
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = np.asarray(image)

    # Convert float to uint8 if needed
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Convert C,H,W to H,W,C if needed
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


@dataclasses.dataclass(frozen=True)
class DualArmInputs(transforms.DataTransformFn):
    """
    Transform for dual arm robot setups with left and right wrist cameras.

    Input observation format:
    - left_wrist_images: Left arm wrist camera [C,H,W] or [H,W,C]
    - right_wrist_images: Right arm wrist camera [C,H,W] or [H,W,C]
    - base_images: Optional third-person camera [C,H,W] or [H,W,C]
    - states: Robot state vector (14 dims: 7 per arm)
    - task_descriptions: Task description string

    Output format for OpenPi model:
    - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    - image_mask: dict indicating which images are valid
    - state: normalized state vector
    - prompt: task description
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse wrist images (required)
        left_wrist = _parse_image(data["left_wrist_images"])
        right_wrist = _parse_image(data["right_wrist_images"])

        # Parse optional base image
        has_base_image = "base_images" in data and data["base_images"] is not None
        if has_base_image:
            base_image = _parse_image(data["base_images"])
        else:
            # Use zeros for base image if not provided
            base_image = np.zeros_like(left_wrist)

        # Parse state
        state = data["states"]
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state = np.asarray(state)

        # Build inputs dict
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                # Mask base image if not provided
                "base_0_rgb": np.True_ if has_base_image else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Handle actions (only during training)
        if "actions" in data:
            actions = data["actions"]
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
            inputs["actions"] = np.asarray(actions)

        # Handle prompt
        prompt_key = "task_descriptions" if "task_descriptions" in data else "prompt"
        if prompt_key in data:
            inputs["prompt"] = data[prompt_key]

        return inputs


@dataclasses.dataclass(frozen=True)
class DualArmOutputs(transforms.DataTransformFn):
    """
    Output transform for dual arm - returns 14 action dimensions.

    Action format (14 dims):
    - Left arm: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper] (7 dims)
    - Right arm: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper] (7 dims)
    """

    action_dim: int = 14  # 7 per arm

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

