'''
OpenPi Policy Integration for Dual-Arm Robot Control
The policy must be trained in the robot coordinate system.
Author: Wang Zheng
'''

import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from policy.base import BasePolicy
import dill
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Tuple, Set
import torch
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Any
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PIL import Image

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import bimanual_flexiv_policy
from openpi.shared import download
import openpi.transforms as _transforms

import hydra
from utils.logger_config import logger
from skimage.transform import resize
import scipy.spatial.transform as st
import cv2
from utils.dsl_pose_utils import pose_to_mat, mat_to_pose10d, mat_to_pose, pose10d_to_mat, convert_pose_mat_rep, get_real_umi_action
from utils.calibration_utils import load_tcp_umi_transforms
from utils.transform import process_quaternion
from scipy.spatial.transform import Rotation


def center_crop_square_resize(
    img: np.ndarray,
    crop_min_ratio: float = 0.05,
    crop_max_ratio: float = 0.95,
    tgt: int = 224,
) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    1) center-crop to square based on the shorter edge
    2) crop to [crop_min_ratio, crop_max_ratio] range of the square on both axes
    3) resize → tgt×tgt
    """
    H, W, C = img.shape
    # Center-crop to square based on the shorter edge
    shorter_edge = min(H, W)
    h_off = (H - shorter_edge) // 2
    w_off = (W - shorter_edge) // 2
    img = img[h_off:h_off + shorter_edge, w_off:w_off + shorter_edge]

    # Dynamic crop of the square dimension (e.g., [15:285] when size==300 and ratios are 0.05/0.95)
    # margin = int(shorter_edge * crop_min_ratio)
    # end = int(shorter_edge * crop_max_ratio)
    #
    # margin = int(shorter_edge * 0)
    # end = int(shorter_edge * 1)
    #
    # if end > margin:
    #     img = img[margin:end, margin:end]

    img = resize(img, (tgt, tgt), preserve_range=True, anti_aliasing=True).astype(
        np.uint8
    )
    return img

def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: (..., 4)  [x, y, z, w]
    return (..., 3) axis-angle (OpenCV convention, norm = angle)
    """
    rot = st.Rotation.from_quat(quat[..., [0, 1, 2, 3]])  # scipy: [x,y,z,w]
    return rot.as_rotvec()

def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: (..., 3)
    return (..., 4) quat
    """
    rot = st.Rotation.from_rotvec(axis_angle)
    return rot.as_quat()

class OpenPiPolicy(BasePolicy):
    def __init__(
        self,
        pi_config_name: str, 
        ckpt_dir: str,
        task: str = "", 
        device: str = 'cuda',
        obs_horizon: int = 2,
        downsample: int = 1,
        obs_time_interval: float = 0.04,
        tcp_umi_json_path: Optional[str] = None,
        enable_visualization: bool = True,
        camera_devices: Optional[List[str]] = None,
        robot_device: Optional[str] = None,
        mask_inference: bool = False,
        left_mask_offset_path: Optional[str] = None,
        right_mask_offset_path: Optional[str] = None,
        visualize_mask_offset: bool = False,
        mask_offset_visual_scale: float = 0.5,
        apply_camera_mask: bool = False,
        left_mask_image_path: Optional[str] = None,
        right_mask_image_path: Optional[str] = None,
        visualize_mask_application: bool = False
    ) -> None:
        super().__init__()
        self.pi_config_name = pi_config_name
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.obs_horizon = obs_horizon
        self.downsample = downsample
        self.obs_time_interval = obs_time_interval
        self._policy, self._input_transform, self._output_transform = self.load_openpi_policy(self.pi_config_name, self.ckpt_dir, self.device)
        self.min_infer_obs = 2

        self.enable_visualization = enable_visualization
        if tcp_umi_json_path is not None:
            self.tcp_umi_T, self.umi_tcp_T = self._load_calibration(tcp_umi_json_path)
        else:
            self.tcp_umi_T, self.umi_tcp_T = None, None
        self.camera_devices: List[str] = list(camera_devices) if camera_devices else []
        self.robot_device: Optional[str] = robot_device
        self.mask_inference: bool = mask_inference
        self.visualize_mask_offset: bool = visualize_mask_offset
        self.mask_offset_visual_scale: float = mask_offset_visual_scale if mask_offset_visual_scale > 0.0 else 0.5
        self.apply_camera_mask: bool = apply_camera_mask
        self.visualize_mask_application: bool = visualize_mask_application
        self.mask_offset_configs: Dict[str, Tuple[float, float]] = {}
        self._pending_mask_offsets: Dict[str, Tuple[float, float]] = {}
        self._mask_visual_windows: Set[str] = set()
        self._mask_size_logged: Set[str] = set()
        self.mask_image_configs: Dict[str, np.ndarray] = {}
        self._pending_mask_images: Dict[str, np.ndarray] = {}
        self._offset_visualization_failed: bool = False
        if self.mask_inference:
            self._pending_mask_offsets = self._load_mask_offsets(
                left_mask_offset_path,
                right_mask_offset_path
            )
            if self.camera_devices:
                self._assign_mask_offsets(self.camera_devices)
        if self.apply_camera_mask:
            self._pending_mask_images = self._load_mask_images(
                left_mask_image_path,
                right_mask_image_path
            )
            if self.camera_devices:
                self._assign_mask_images(self.camera_devices)

    def _load_calibration(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        tcp_umi, umi_tcp = load_tcp_umi_transforms(json_path)
        logger.info(f"Loaded TCP↔UMI calibration from {json_path}")
        return tcp_umi, umi_tcp

    def load_openpi_policy(self, pi_config_name: str, ckpt_dir: str, device='cuda'):
        self.pi_config = _config.get_config(pi_config_name)
        print(f"Loading policy from {ckpt_dir}")
        policy = _policy_config.create_trained_policy(self.pi_config, ckpt_dir)
        input_transform = bimanual_flexiv_policy.BimanualFlexivInputs(model_type=self.pi_config.model.model_type)
        output_transform = bimanual_flexiv_policy.BimanualFlexivOutputs()
        return policy, input_transform, output_transform

    def _load_mask_offsets(
        self,
        left_mask_offset_path: Optional[str],
        right_mask_offset_path: Optional[str]
    ) -> Dict[str, Tuple[float, float]]:
        offsets: Dict[str, Tuple[float, float]] = {}

        if left_mask_offset_path:
            left_offset = self._parse_mask_offset_file(left_mask_offset_path)
            if left_offset is not None:
                offsets['left'] = left_offset
            else:
                logger.warning("Failed to load left mask offset from %s", left_mask_offset_path)

        if right_mask_offset_path:
            right_offset = self._parse_mask_offset_file(right_mask_offset_path)
            if right_offset is not None:
                offsets['right'] = right_offset
            else:
                logger.warning("Failed to load right mask offset from %s", right_mask_offset_path)

        return offsets

    def _assign_mask_offsets(self, camera_names: List[str]) -> None:
        if not camera_names:
            logger.warning("Mask inference enabled but no camera devices are configured.")
            return

        if not self._pending_mask_offsets:
            return

        assigned = False
        left_offset = self._pending_mask_offsets.get('left')
        right_offset = self._pending_mask_offsets.get('right')

        if left_offset is not None and camera_names:
            target_camera = camera_names[0]
            if target_camera not in self.mask_offset_configs:
                self.mask_offset_configs[target_camera] = left_offset
                assigned = True
                logger.info(f"Applied mask offset for {target_camera}: {left_offset}")

        if right_offset is not None and camera_names:
            target_index = 1 if len(camera_names) > 1 else 0
            target_camera = camera_names[target_index]
            if target_camera not in self.mask_offset_configs:
                self.mask_offset_configs[target_camera] = right_offset
                assigned = True
                logger.info(f"Applied mask offset for {target_camera}: {right_offset}")

        if not assigned:
            logger.warning("Mask offsets parsed but no camera assignment performed. camera_names=%s", camera_names)

    def _load_mask_images(
        self,
        left_mask_image_path: Optional[str],
        right_mask_image_path: Optional[str]
    ) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}

        if left_mask_image_path:
            left_mask = self._parse_mask_image_file(left_mask_image_path)
            if left_mask is not None:
                masks['left'] = left_mask
            else:
                logger.warning("Failed to load left mask image from %s", left_mask_image_path)

        if right_mask_image_path:
            right_mask = self._parse_mask_image_file(right_mask_image_path)
            if right_mask is not None:
                masks['right'] = right_mask
            else:
                logger.warning("Failed to load right mask image from %s", right_mask_image_path)

        return masks

    def _assign_mask_images(self, camera_names: List[str]) -> None:
        if not camera_names:
            logger.warning("Mask images provided but no camera devices are configured.")
            return

        if not self._pending_mask_images:
            return

        assigned = False
        left_mask = self._pending_mask_images.get('left')
        right_mask = self._pending_mask_images.get('right')

        if left_mask is not None and camera_names:
            target_camera = camera_names[0]
            if target_camera not in self.mask_image_configs:
                self.mask_image_configs[target_camera] = left_mask
                assigned = True
                logger.info(f"Applied mask image for {target_camera}")

        if right_mask is not None and camera_names:
            target_index = 1 if len(camera_names) > 1 else 0
            target_camera = camera_names[target_index]
            if target_camera not in self.mask_image_configs:
                self.mask_image_configs[target_camera] = right_mask
                assigned = True
                logger.info(f"Applied mask image for {target_camera}")

        if not assigned:
            logger.warning("Mask images loaded but no camera assignment performed. camera_names=%s", camera_names)

    def _parse_mask_offset_file(self, file_path: str) -> Optional[Tuple[float, float]]:
        try:
            path_obj = Path(file_path)
            if not path_obj.is_file():
                logger.error(f"Mask offset file not found: {file_path}")
                return None
            with path_obj.open('r', encoding='utf-8') as file:
                data = json.load(file)

            # Try new format first: data['offset']['x'] and data['offset']['y']
            offset_data = data.get('offset', {})
            offset_x = offset_data.get('x')
            offset_y = offset_data.get('y')

            # Fallback to old format: data['offset_results']['centroid']['offset_x']
            if offset_x is None or offset_y is None:
                centroid_data = data.get('offset_results', {}).get('centroid', {})
                offset_x = centroid_data.get('offset_x')
                offset_y = centroid_data.get('offset_y')

            if offset_x is None or offset_y is None:
                logger.error(f"Mask offset file missing centroid offsets: {file_path}")
                return None
            return float(offset_x), float(offset_y)
        except Exception as exc:
            logger.error(f"Failed to parse mask offset file {file_path}: {exc}")
            return None

    def _parse_mask_image_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            path_obj = Path(file_path)
            if not path_obj.is_file():
                logger.error(f"Mask image file not found: {file_path}")
                return None
            mask_image = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                logger.error(f"Failed to read mask image file: {file_path}")
                return None

            if mask_image.ndim == 2:
                mask_image = mask_image[..., None]
            if mask_image.shape[2] == 4:
                mask_image = mask_image[..., :3]
            if mask_image.shape[2] == 1:
                mask_image = np.repeat(mask_image, 3, axis=2)

            mask_float = mask_image.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float /= 255.0
            return mask_float
        except Exception as exc:
            logger.error(f"Failed to parse mask image file {file_path}: {exc}")
            return None

    def _apply_mask_offset_if_needed(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        if not self.mask_inference or camera_name not in self.mask_offset_configs:
            return image

        offset_x, offset_y = self.mask_offset_configs[camera_name]
        height, width = image.shape[:2]
        # Negate the offset to move image in opposite direction to align mask with image center
        translation_matrix = np.array([[1.0, 0.0, -offset_x], [0.0, 1.0, -offset_y]], dtype=np.float32)
        shifted_image = cv2.warpAffine(
            image,
            translation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )
        if self.visualize_mask_offset:
            self._visualize_mask_offset(camera_name, image, shifted_image)
        return shifted_image

    def _apply_camera_mask_if_needed(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        if not self.apply_camera_mask or camera_name not in self.mask_image_configs:
            return image

        mask = self.mask_image_configs[camera_name]
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)

        mask = mask.astype(np.float32)
        image_float = image.astype(np.float32)
        masked_image = np.clip(image_float * mask, 0.0, 255.0).astype(np.uint8)

        if self.visualize_mask_application:
            self._visualize_mask_application(camera_name, masked_image)

        return masked_image

    def _visualize_mask_offset(self, camera_name: str, original: np.ndarray, shifted: np.ndarray) -> None:
        if self._offset_visualization_failed:
            return
        try:
            window_overlay = f"{camera_name}_offset_overlay"
            display_original = self._resize_for_visualization(original)
            display_shifted = self._resize_for_visualization(shifted)
            display_overlay = self._create_colored_overlay(display_original, display_shifted)
            if camera_name not in self._mask_size_logged:
                original_size = tuple(int(dim) for dim in original.shape)
                shifted_size = tuple(int(dim) for dim in shifted.shape)
                logger.info(
                    f"Mask offset visualization size | camera={camera_name} | original={original_size} | shifted={shifted_size}"
                )
                self._mask_size_logged.add(camera_name)
            if window_overlay not in self._mask_visual_windows:
                cv2.namedWindow(window_overlay, cv2.WINDOW_NORMAL)
                self._mask_visual_windows.add(window_overlay)
            cv2.imshow(window_overlay, display_overlay)
            cv2.waitKey(1)
        except Exception as exc:
            self._offset_visualization_failed = True
            logger.error(f"Failed to visualize mask offset for {camera_name}: {exc}")

    def _visualize_mask_application(self, camera_name: str, masked_image: np.ndarray) -> None:
        if self._offset_visualization_failed:
            return
        try:
            window_name = f"{camera_name}_mask_applied"
            display_image = self._resize_for_visualization(masked_image)
            if window_name not in self._mask_visual_windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                self._mask_visual_windows.add(window_name)
            cv2.imshow(window_name, display_image)
            cv2.waitKey(1)
        except Exception as exc:
            logger.error(f"Failed to visualize mask application for {camera_name}: {exc}")

    def _resize_for_visualization(self, image: np.ndarray) -> np.ndarray:
        if self.mask_offset_visual_scale == 1.0:
            return image
        height, width = image.shape[:2]
        scale = self.mask_offset_visual_scale
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    def _create_colored_overlay(self, original: np.ndarray, shifted: np.ndarray) -> np.ndarray:
        original_float = original.astype(np.float32)
        shifted_float = shifted.astype(np.float32)
        overlay = np.zeros_like(original_float)
        overlay[..., 1] = original_float[..., 1]
        overlay[..., 0] = 0.6 * shifted_float[..., 0]
        overlay[..., 2] = 0.6 * shifted_float[..., 2]
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    def _convert_obs_to_pi(self, obs_batch_robot: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raw_data = {}
        states = np.concatenate(
            [
                obs_batch_robot["robot0_eef_pos"],
                obs_batch_robot["robot0_eef_rot_axis_angle"],
                obs_batch_robot["robot0_gripper_width"],
                obs_batch_robot["robot1_eef_pos"],
                obs_batch_robot["robot1_eef_rot_axis_angle"],
                obs_batch_robot["robot1_gripper_width"],
            ],
            axis=1,
        ).astype(np.float32)
        raw_data['state'] = states[1,:]
        raw_data["left_wrist_image"] = obs_batch_robot['camera0_rgb'][-1, :]
        raw_data["right_wrist_image"] = obs_batch_robot['camera1_rgb'][-1, :]
        raw_data["task"] = self.task
        repacked_data = {
            "observation/left_wrist_image": raw_data["left_wrist_image"],
            "observation/right_wrist_image": raw_data["right_wrist_image"], 
            "observation/state": raw_data["state"],
            "prompt": raw_data["task"],
        }
        repacked_data = {k:torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in repacked_data.items()}
        transformed_data = self._input_transform(repacked_data)
        sample = {
            'state': transformed_data["state"],
            'left_wrist_image': transformed_data["image"]["left_wrist_0_rgb"],
            'right_wrist_image': transformed_data["image"]["right_wrist_0_rgb"],
            'prompt': transformed_data.get("prompt", "pick up object"),
        }
        obs_input = {
            "observation/state": sample['state'],
            "observation/left_wrist_image": sample['left_wrist_image'],
            "observation/right_wrist_image": sample['right_wrist_image'],
            "prompt": sample['prompt'],
        }
        return obs_input

    def predict(self, observation, action_configs: List[Dict[str, Any]], chunk_length: int = 1) -> Dict[str, np.ndarray]:
        # 1. Organize observation data for the policy
        obs_batch_robot, robot_state = self.sensor_data_to_umi_data(observation, target_frame="robot") # obs_batch_robot is in the **umi** coordinate system
        robot_current_pose_left = robot_state[:6]
        robot_current_pose_right = robot_state[7:13]
        robot_current_mat_left = pose_to_mat(robot_current_pose_left)
        robot_current_mat_right = pose_to_mat(robot_current_pose_right)

        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot)
        with torch.no_grad():
            robot_rel_action = self._policy.infer(obs_batch_for_policy)["actions"] # action is the absolute pose in the **umi** coordinate system
        
        # 2. Convert the relative action in **rpy** representation to **rotation matrix** in the **robot** coordinate system to the
        robot_rel_action_pos_left, robot_rel_action_rpy_left, robot_rel_action_gripper = robot_rel_action[...,:3], robot_rel_action[...,3:6], robot_rel_action[...,6:7]
        robot_rel_action_pos_right, robot_rel_action_rpy_right, robot_rel_action_gripper_right = robot_rel_action[...,7:10], robot_rel_action[...,10:13], robot_rel_action[...,13:14]
        robot_rel_action_axis_angle_left = Rotation.from_euler('xyz', robot_rel_action_rpy_left, degrees=False).as_rotvec()
        robot_rel_action_axis_angle_right = Rotation.from_euler('xyz', robot_rel_action_rpy_right, degrees=False).as_rotvec()
        robot_rel_action_pose_left = np.concatenate([robot_rel_action_pos_left, robot_rel_action_axis_angle_left], axis=-1)
        robot_rel_action_pose_right = np.concatenate([robot_rel_action_pos_right, robot_rel_action_axis_angle_right], axis=-1)
        robot_rel_action_mat_left = pose_to_mat(robot_rel_action_pose_left)
        robot_rel_action_mat_right = pose_to_mat(robot_rel_action_pose_right)

        # 3. Compute the absolute action in the **robot** coordinate system
        robot_abs_action_mat_left = convert_pose_mat_rep(robot_rel_action_mat_left, robot_current_mat_left, pose_rep='relative', backward=True)
        robot_abs_action_mat_right = convert_pose_mat_rep(robot_rel_action_mat_right, robot_current_mat_right, pose_rep='relative', backward=True)
        robot_abs_action_pose_left = mat_to_pose(robot_abs_action_mat_left)
        robot_abs_action_pose_right = mat_to_pose(robot_abs_action_mat_right)
        robot_abs_action = np.concatenate([robot_abs_action_pose_left, robot_rel_action_gripper, robot_abs_action_pose_right, robot_rel_action_gripper_right], axis=-1)

        # 4. Convert the absolute action in the **robot** coordinate system to the executable dual pose
        dual_pose = self.action_to_dual_pose(robot_abs_action)
        dual_pose[:, 7] = np.clip(dual_pose[:, 7], 0.01, 0.08)
        dual_pose[:, 15] = np.clip(dual_pose[:, 15], 0.01, 0.08)
        return {action_configs[0]['device_name']: dual_pose.astype(np.float64)}

    def sensor_data_to_umi_data(self, obs_dict: Dict[str, List[np.ndarray]], target_frame: str = "umi") -> Dict[str, np.ndarray]:
        batch_data: Dict[str, np.ndarray] = {}
        cameras = self.camera_devices or [name for name in obs_dict.keys() if 'camera' in name.lower()]
        if self.mask_inference and not self.mask_offset_configs and cameras:
            self._assign_mask_offsets(cameras)
        if self.apply_camera_mask and not self.mask_image_configs and cameras:
            self._assign_mask_images(cameras)
        for idx, camera_name in enumerate(cameras):
            raw_images = obs_dict[camera_name]
            if isinstance(raw_images, np.ndarray):
                image_sequence = [raw_images]
            else:
                image_sequence = [np.asarray(frame) for frame in raw_images]
            processed_images = []
            for image in image_sequence:
                adjusted_image = self._apply_mask_offset_if_needed(camera_name, image)
                masked_image = self._apply_camera_mask_if_needed(camera_name, adjusted_image)
                processed_images.append(center_crop_square_resize(masked_image))
            stacked_images = np.stack(processed_images)
            batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)

        robot_device = self.robot_device
        if robot_device is None:
            robot_device = next(iter([key for key in obs_dict.keys() if key.lower().startswith('robot') or key.lower().startswith('rizon')]), None)
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")
        left_tcp_pose_w_gripper = np.stack([obs[14:22] for obs in obs_dict[robot_device]])
        right_tcp_pose_w_gripper = np.stack([obs[36:44] for obs in obs_dict[robot_device]])

        left_pos = left_tcp_pose_w_gripper[:, :3]
        left_quat_xyzw = np.stack([process_quaternion(left_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(left_tcp_pose_w_gripper))])
        left_axis_angle = quaternion_to_axis_angle(left_quat_xyzw)

        right_pos = right_tcp_pose_w_gripper[:, :3]
        right_quat_xyzw = np.stack([process_quaternion(right_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(right_tcp_pose_w_gripper))])
        right_axis_angle = quaternion_to_axis_angle(right_quat_xyzw)
        left_gripper = left_tcp_pose_w_gripper[:, 7:8]
        right_gripper = right_tcp_pose_w_gripper[:, 7:8]

        robot_left_pose = np.concatenate([left_pos, left_axis_angle, left_gripper], axis=-1)
        robot_right_pose = np.concatenate([right_pos, right_axis_angle, right_gripper], axis=-1)
        robot_pose = copy.deepcopy(np.concatenate([robot_left_pose, robot_right_pose], axis=-1)[-1,:])

        batch_data['robot0_eef_pos'] = left_pos
        batch_data['robot0_eef_rot_axis_angle'] = left_axis_angle
        # TODO: remove this hack later
        batch_data['robot0_gripper_width'] = left_gripper
        # batch_data['robot0_gripper_width'] = (left_gripper / 60.0) * 80.0
        batch_data['robot1_eef_pos'] = right_pos
        batch_data['robot1_eef_rot_axis_angle'] = right_axis_angle
        # TODO: remove this hack later
        batch_data['robot1_gripper_width'] = right_gripper
        # batch_data['robot1_gripper_width'] = (right_gripper / 60.0) * 80.0
        return batch_data, robot_pose

    def _robot_pose_to_umi(self, pos: np.ndarray, axis_angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.tcp_umi_T is None:
            return pos, axis_angle
        pose_mat = pose_to_mat(np.concatenate([pos, axis_angle], axis=-1))
        transformed = pose_mat @ self.tcp_umi_T
        pose = mat_to_pose(transformed)
        return pose[:, :3], pose[:, 3:]

    def _umi_pose_to_robot_mat(self, pose_mat: np.ndarray) -> np.ndarray:
        if self.tcp_umi_T is None: return pose_mat
        return self.tcp_umi_T @ pose_mat @ self.umi_tcp_T

    def action_to_dual_pose(self, umi_action):
        left_pos, left_axis_angle, left_gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6 + 1]
        right_pos, right_axis_angle, right_gripper = umi_action[:, 7:10], umi_action[:, 10:13], umi_action[:, 13:13 + 1]
        left_quat = axis_angle_to_quaternion(left_axis_angle)
        left_pose = np.concatenate([left_pos, left_quat, left_gripper], axis=-1)
        right_quat = axis_angle_to_quaternion(right_axis_angle)
        right_pose = np.concatenate([right_pos, right_quat, right_gripper], axis=-1)
        dual_pose = np.concatenate([left_pose, right_pose], axis=-1)
        return dual_pose


if __name__=='__main__':
    policy = OpenPiPolicy(
        pi_config_name='pi05_pick_1015merged',
        ckpt_dir='/mnt/model/Codes/openpi/checkpoints/pi05_pick_1015merged/exp_pi05_1015merged/29999',
        device='cuda',
        obs_horizon=2,
        downsample=1,
        obs_time_interval=0.04,
        tcp_umi_json_path='./calibration/tcp_to_vive_transform.json',
        mask_inference=True,
        left_mask_offset_path='./calibration/mask_offset/left/mask_offset_analysis_20250929_151954.json',
        right_mask_offset_path='./calibration/mask_offset/right/mask_offset_analysis_20250929_152016.json',
        visualize_mask_offset=False,
        mask_offset_visual_scale=0.4,
        apply_camera_mask=True,
        left_mask_image_path='./calibration/masks/umi/mapped/left_mask.png',
        right_mask_image_path='./calibration/masks/umi/mapped/right_mask.png',
        visualize_mask_application=False,
    )
    # create a dummy batch
    obs_horizon = 2
    observation = {
        'OpenCVCameraDevice_0': [np.random.randint(0, 255, (1200, 1600, 3)).astype(np.uint8) for _ in range(obs_horizon)],
        'OpenCVCameraDevice_2': [np.random.randint(0, 255, (1200, 1600, 3)).astype(np.uint8) for _ in range(obs_horizon)],
        'RizonRobot_1': [np.random.rand(44).astype(np.float64) for _ in range(obs_horizon)],
    }
    action_configs = [
        {'device_name': 'RizonRobot_1', 'action_dim': 16, 'shm_name': 'RizonRobot_1_control', 'metadata':{}},
    ]
    action = policy.predict(observation, action_configs)
    print(action)