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
Data Configuration for Dual Arm Robot Training.

This module configures data transforms for dual arm robot setups with:
- Left and right wrist cameras
- Optional base/table camera
- 14-dimensional state (7 dims per arm)
- 14-dimensional actions (7 dims per arm)
"""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dual_arm_policy


@dataclasses.dataclass(frozen=True)
class DualArmDataConfig(DataConfigFactory):
    """
    Data configuration for dual arm robot training.

    This config sets up transforms for:
    - Dual wrist camera observations (left and right)
    - 14-dim state vector (7 per arm: pos + rot + gripper)
    - 14-dim action vector (7 per arm)

    Args:
        extra_delta_transform: If True, converts absolute actions to delta actions.
            Set to True if your dataset uses absolute actions (e.g., target joint angles).
        action_dim: Number of action dimensions (default: 14 for dual arm).
        gripper_dims: Number of gripper dimensions per arm (default: 1).
    """

    extra_delta_transform: bool = True
    action_dim: int = 14  # 7 per arm
    gripper_dims: int = 1  # gripper dim per arm

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack transform maps dataset keys to inference pipeline keys
        # This is only applied to dataset data, not during inference
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # Map dataset keys to standard dual arm format
                        "left_wrist_images": "left_wrist_image",
                        "right_wrist_images": "right_wrist_image",
                        "base_images": "base_image",  # Optional
                        "states": "state",
                        "actions": "actions",
                        "task_descriptions": "prompt",
                    }
                )
            ]
        )

        # Data transforms applied to both dataset and inference data
        data_transforms = _transforms.Group(
            inputs=[dual_arm_policy.DualArmInputs(model_type=model_config.model_type)],
            outputs=[dual_arm_policy.DualArmOutputs(action_dim=self.action_dim)],
        )

        # Delta action transform (converts absolute to delta actions)
        # For dual arm: apply delta to position+rotation, keep gripper absolute
        # Left arm: dims 0-5 delta, dim 6 absolute (gripper)
        # Right arm: dims 7-12 delta, dim 13 absolute (gripper)
        if self.extra_delta_transform:
            # Create mask: True = apply delta, False = keep absolute
            # [pos(3) + rot(3) + gripper(1)] * 2 arms
            left_arm_mask = [True] * 6 + [False] * self.gripper_dims  # 7 dims
            right_arm_mask = [True] * 6 + [False] * self.gripper_dims  # 7 dims
            delta_action_mask = _transforms.make_bool_mask(
                len(left_arm_mask + right_arm_mask), -1
            )
            # Override with our specific mask
            delta_action_mask = left_arm_mask + right_arm_mask

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms (tokenization, etc.) - don't modify
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

