'''
Diffusion Policy - runs a diffusion policy.

Author: Han Xue, Wang Zheng
'''

import sys
import os
import json
from pathlib import Path
from policy.base import BasePolicy
import dill
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import torch
import hydra
from utils.logger_config import logger
from skimage.transform import resize
import scipy.spatial.transform as st
import cv2
ROOT_DIR = Path(__file__).resolve().parents[1]
UMI_BASE_DIR = ROOT_DIR / 'third_party' / 'umi_base'
if str(UMI_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(UMI_BASE_DIR))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party/data-scaling-laws'))
############### Dependency on data-scaling-laws ###############
from umi.common.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose, pose10d_to_mat
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
#################################################
from utils.calibration_utils import load_tcp_umi_transforms
from utils.transform import process_quaternion



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
    margin = int(shorter_edge * crop_min_ratio)
    end = int(shorter_edge * crop_max_ratio)
    if end > margin:
        img = img[margin:end, margin:end]

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

class DiffusionPolicy(BasePolicy):
    def __init__(
        self,
        ckpt_dir: str,
        num_inference_steps: int = 16,
        device: str = 'cuda',
        obs_horizon: int = 2,
        downsample: int = 3,
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
        visualize_mask_application: bool = False,
        preprocess_resize_height: int = 300,
        preprocess_resize_width: int = 400,
        center_crop_min_ratio: float = 0.05,
        center_crop_max_ratio: float = 0.95,
    ) -> None:
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.obs_horizon = obs_horizon
        self.obs_time_interval = obs_time_interval
        self.downsample = downsample
        self.enable_visualization = enable_visualization
        assert self.obs_horizon > 1, "obs_horizon must be greater than 1"
        assert self.downsample > 0, "downsample must be greater than 0"
        self.min_infer_obs = (self.obs_horizon-1) * self.downsample + 1
        self._policy, self.obs_pose_repr, self.action_pose_repr = self.load_diffusion_policy(self.ckpt_dir, self.num_inference_steps, self.device)
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
        self._last_num_robots: int = 2  # cache latest robot count for single/dual compatibility
        # Preprocessing configuration
        self.preprocess_resize_height: int = preprocess_resize_height
        self.preprocess_resize_width: int = preprocess_resize_width
        self.center_crop_min_ratio: float = center_crop_min_ratio
        self.center_crop_max_ratio: float = center_crop_max_ratio
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

    def load_diffusion_policy(self, ckpt_dir, num_inference_steps=16, device='cuda'):
        ckpt_path = os.path.join(ckpt_dir, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        logger.info(f"model_name: {cfg.policy.obs_encoder.model_name}")
        if 'dataset_path' in cfg.task.dataset:
            logger.info("dataset_path:", cfg.task.dataset.dataset_path)
        obs_res = get_real_obs_resolution(cfg.task.shape_meta)
        logger.info(f"obs_res: {obs_res}")
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.num_inference_steps = num_inference_steps
        policy.eval()
        policy.to(device)

        obs_pose_rep = cfg.task.pose_repr.obs_pose_repr if "pose_repr" in cfg.task else 'relative'
        action_pose_repr = cfg.task.pose_repr.action_pose_repr if "pose_repr" in cfg.task else 'relative'
        logger.info(f'obs_pose_rep: {obs_pose_rep}')
        logger.info(f'action_pose_repr: {action_pose_repr}')
        return policy, obs_pose_rep, action_pose_repr

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
            # Support new format: offset_results.centroid.offset_x/offset_y
            centroid_data = data.get('offset_results', {}).get('centroid', {})
            offset_x = centroid_data.get('offset_x')
            offset_y = centroid_data.get('offset_y')
            # Fallback to old format: offset.x / offset.y
            if offset_x is None or offset_y is None:
                legacy = data.get('offset', {})
                offset_x = legacy.get('x')
                offset_y = legacy.get('y')
            if offset_x is None or offset_y is None:
                logger.error(f"Mask offset file missing recognized offsets: {file_path}")
                return None
            # Apply compensation: negate offsets to counteract recorded displacement
            return -float(offset_x), -float(offset_y)
        except Exception as exc:
            logger.error(f"Failed to parse mask offset file {file_path}: {exc}")
            return None

    def _parse_mask_image_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            path_obj = Path(file_path)
            if not path_obj.is_file():
                logger.error(f"Mask image file not found: {file_path}")
                return None
            # Load as grayscale mask and binarize similar to preprocessing
            mask_gray = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                logger.error(f"Failed to read mask image file: {file_path}")
                return None
            # Threshold to binary {0,255}
            if mask_gray.max() <= 1:
                mask_gray = (mask_gray > 0).astype(np.uint8) * 255
            else:
                _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            return mask_gray.astype(np.uint8)
        except Exception as exc:
            logger.error(f"Failed to parse mask image file {file_path}: {exc}")
            return None

    def _apply_mask_offset_if_needed(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        if not self.mask_inference or camera_name not in self.mask_offset_configs:
            return image

        offset_x, offset_y = self.mask_offset_configs[camera_name]
        height, width = image.shape[:2]
        translation_matrix = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y]], dtype=np.float32)
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
        # Ensure mask size matches image HxW
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Binary application: set pixels to 0 where mask==0
        if mask.ndim != 2:
            # If somehow mask has channels, reduce to single channel
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masked_image = image.copy()
        masked_image[mask == 0] = 0

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
            # Convert RGB -> BGR for correct OpenCV display
            if display_overlay.ndim == 3 and display_overlay.shape[-1] == 3:
                display_overlay = cv2.cvtColor(display_overlay, cv2.COLOR_RGB2BGR)
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
            # Convert RGB -> BGR for correct OpenCV display
            if display_image.ndim == 3 and display_image.shape[-1] == 3:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
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

    def predict(self, observation, action_configs: List[Dict[str, Any]], chunk_length: int = 1) -> Dict[str, np.ndarray]:
        obs_batch_robot = self.sensor_data_to_umi_data(observation, target_frame="robot")
        num_robots = getattr(self, "_last_num_robots", 2)
        obs_batch_for_policy = obs_batch_robot if self.tcp_umi_T is None else self._convert_obs_to_umi(obs_batch_robot, num_robots=num_robots)
        processed_batch = self.convert_umi_obs_repr(obs_batch_for_policy.copy(), num_robots=num_robots)
        processed_batch = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in processed_batch.items()}

        with torch.no_grad():
            action = self._policy.predict_action(processed_batch)

        transformed_action = self.umi_action_to_robot(action['action'][0].cpu().numpy())
        robot_abs_action = get_real_umi_action(transformed_action, obs_batch_robot, self.action_pose_repr)
        # dual-arm data
        if transformed_action.shape[-1] == 20:
            dual_pose = self.action_to_dual_pose(robot_abs_action)
            # TODO: remove this hack for gripper limits(only for robotiq gripper)
            dual_pose[:, 7] = np.clip(dual_pose[:, 7], 0.01, 0.08)
            dual_pose[:, 15] = np.clip(dual_pose[:, 15], 0.01, 0.08)
            return {action_configs[0]['device_name']: dual_pose.astype(np.float64)}
        # single-arm data
        else:
            single_pose = self.action_to_single_pose(robot_abs_action)
            # TODO: remove this hack for gripper limits(only for robotiq gripper)
            single_pose[:, 7] = np.clip(single_pose[:, 7], 0.01, 0.08)
            return {action_configs[0]['device_name']: single_pose.astype(np.float64)}


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
                # 1) Resize to match mask/offset working resolution (configurable HxW)
                target_h = self.preprocess_resize_height
                target_w = self.preprocess_resize_width
                if image.shape[:2] != (target_h, target_w):
                    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                # 2) Apply offset compensation first
                adjusted_image = self._apply_mask_offset_if_needed(camera_name, image)
                masked_image = self._apply_camera_mask_if_needed(camera_name, adjusted_image)
                cropped_resized_image = center_crop_square_resize(
                    masked_image,
                    crop_min_ratio=self.center_crop_min_ratio,
                    crop_max_ratio=self.center_crop_max_ratio,
                )
                processed_images.append(cropped_resized_image)
                # # Debug visualization
                # cv2.imshow(f"{camera_name}_original", image)
                # cv2.imshow(f"{camera_name}_processed", cropped_resized_image)
                # cv2.waitKey(1)
            stacked_images = np.stack(processed_images)
            batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)

        robot_device = self.robot_device
        if robot_device is None:
            robot_device = next(iter([key for key in obs_dict.keys() if key.lower().startswith('robot') or key.lower().startswith('rizon')]), None)
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")

        robot_obs_seq = obs_dict[robot_device]
        if isinstance(robot_obs_seq, np.ndarray):
            robot_obs_array = robot_obs_seq if robot_obs_seq.ndim > 1 else robot_obs_seq[None, :]
        else:
            robot_obs_array = np.asarray([np.asarray(obs) for obs in robot_obs_seq])
        if robot_obs_array.ndim == 1:
            robot_obs_array = robot_obs_array[None, :]

        obs_dim = robot_obs_array.shape[-1]
        if obs_dim < 22:
            raise ValueError(f"Unsupported robot observation dimension: expected at least 22, got {obs_dim}")
        is_dual_arm = obs_dim >= 44
        self._last_num_robots = 2 if is_dual_arm else 1  # remember whether this batch was single or dual arm

        left_tcp_pose_w_gripper = robot_obs_array[:, 14:22]
        left_pos = left_tcp_pose_w_gripper[:, :3]
        left_quat_xyzw = np.stack([process_quaternion(left_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(left_tcp_pose_w_gripper))])
        left_axis_angle = quaternion_to_axis_angle(left_quat_xyzw)

        right_tcp_pose_w_gripper = right_pos = right_axis_angle = None
        if is_dual_arm:
            right_tcp_pose_w_gripper = robot_obs_array[:, 36:44]
            right_pos = right_tcp_pose_w_gripper[:, :3]
            right_quat_xyzw = np.stack([process_quaternion(right_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(right_tcp_pose_w_gripper))])
            right_axis_angle = quaternion_to_axis_angle(right_quat_xyzw)

        if target_frame == "umi" and self.tcp_umi_T is not None:
            left_pos, left_axis_angle = self._robot_pose_to_umi(left_pos, left_axis_angle)
            if is_dual_arm and right_pos is not None and right_axis_angle is not None:
                right_pos, right_axis_angle = self._robot_pose_to_umi(right_pos, right_axis_angle)

        left_gripper = left_tcp_pose_w_gripper[:, 7:8]
        batch_data['robot0_eef_pos'] = left_pos
        batch_data['robot0_eef_rot_axis_angle'] = left_axis_angle
        batch_data['robot0_gripper_width'] = left_gripper
        if is_dual_arm and right_tcp_pose_w_gripper is not None:
            right_gripper = right_tcp_pose_w_gripper[:, 7:8]
            batch_data['robot1_eef_pos'] = right_pos
            batch_data['robot1_eef_rot_axis_angle'] = right_axis_angle
            batch_data['robot1_gripper_width'] = right_gripper
        return batch_data

    def convert_umi_obs_repr(self, obs_dict, num_robots=2):
        """Convert the coordinates to relative or abs based on the checkpoint model's training setting"""
        obs_pose_repr = self.obs_pose_repr
        for robot_id in range(num_robots):
            # convert pose to mat
            if f'robot{robot_id}_eef_pos' not in obs_dict or f'robot{robot_id}_eef_rot_axis_angle' not in obs_dict:
                continue  # allow single-arm batches to pass num_robots=1 safely
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle'],
            ], axis=-1))
            obs_pose_mat = convert_pose_mat_rep(
                    pose_mat, 
                    base_pose_mat=pose_mat[-1],
                    pose_rep=obs_pose_repr,
                    backward=False
                    )
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
        return obs_dict

    def umi_action_to_robot(self, action: np.ndarray) -> np.ndarray:
        robot_actions = []
        n_robots = int(action.shape[-1] // 10)
        for robot_idx in range(n_robots):
            start = robot_idx * 10
            action_pose10d = action[..., start:start + 9]
            action_grip = action[..., start + 9:start + 10]
            action_pose_mat = pose10d_to_mat(action_pose10d)
            if self.tcp_umi_T is not None:
                action_pose_mat = self._umi_pose_to_robot_mat(action_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
            robot_actions.append(action_pose)
            robot_actions.append(action_grip)
        return np.concatenate(robot_actions, axis=-1)

    def _robot_pose_to_umi(self, pos: np.ndarray, axis_angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.tcp_umi_T is None:
            return pos, axis_angle
        pose_mat = pose_to_mat(np.concatenate([pos, axis_angle], axis=-1))
        transformed = pose_mat @ self.tcp_umi_T
        pose = mat_to_pose(transformed)
        return pose[:, :3], pose[:, 3:]

    def _umi_pose_to_robot_mat(self, pose_mat: np.ndarray) -> np.ndarray:
        if self.tcp_umi_T is None:
            return pose_mat
        return self.tcp_umi_T @ pose_mat @ self.umi_tcp_T

    def _convert_obs_to_umi(self, robot_batch: Dict[str, np.ndarray], num_robots: int) -> Dict[str, np.ndarray]:
        umi_batch: Dict[str, np.ndarray] = {}
        for key, value in robot_batch.items():
            umi_batch[key] = value.copy() if isinstance(value, np.ndarray) else value
        for robot_idx in range(num_robots):
            pos_key = f'robot{robot_idx}_eef_pos'
            rot_key = f'robot{robot_idx}_eef_rot_axis_angle'
            if pos_key not in umi_batch or rot_key not in umi_batch:
                continue  # skip absent arm entries in single-arm mode
            umi_pos, umi_rot = self._robot_pose_to_umi(umi_batch[pos_key], umi_batch[rot_key])
            umi_batch[pos_key] = umi_pos
            umi_batch[rot_key] = umi_rot
        return umi_batch
    
    def action_to_single_pose(self, umi_action):
        pos, axis_angle, gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6+1]
        quat = axis_angle_to_quaternion(axis_angle)
        single_pose = np.concatenate([pos, quat, gripper], axis=-1)
        return single_pose
    
    def action_to_dual_pose(self, umi_action):
        left_pos, left_axis_angle, left_gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6 + 1]
        right_pos, right_axis_angle, right_gripper = umi_action[:, 7:10], umi_action[:, 10:13], umi_action[:, 13:13 + 1]
        left_quat = axis_angle_to_quaternion(left_axis_angle)
        left_pose = np.concatenate([left_pos, left_quat, left_gripper], axis=-1)
        right_quat = axis_angle_to_quaternion(right_axis_angle)
        right_pose = np.concatenate([right_pos, right_quat, right_gripper], axis=-1)
        dual_pose = np.concatenate([left_pose, right_pose], axis=-1)
        return dual_pose


        
