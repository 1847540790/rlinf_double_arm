"""
Interpolation Chunk Manager - smooth action transitions via linear + slerp interpolation.

Author: OpenAI Codex
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from utils.logger_config import logger
from .action_manager import BaseChunkManager


class InterpolationChunkManager(BaseChunkManager):
    """
    Interpolate between the last executed action and the incoming action chunk.

    Args:
        interpolation_ratio (int): Upsampling ratio between consecutive actions.
        quaternion_format (str): Quaternion ordering, "xyzw" (default) or "wxyz".
    """

    def __init__(self, interpolation_ratio: int = 1, quaternion_format: str = "xyzw") -> None:
        super().__init__()
        ratio = int(interpolation_ratio)
        if ratio < 1:
            logger.warning(
                "InterpolationChunkManager: interpolation_ratio %s < 1, defaulting to 1",
                interpolation_ratio,
            )
            ratio = 1
        self.interpolation_ratio = ratio
        quat_format = quaternion_format.lower()
        if quat_format not in ("xyzw", "wxyz"):
            logger.warning(
                "InterpolationChunkManager: invalid quaternion_format '%s', defaulting to 'xyzw'",
                quaternion_format,
            )
            quat_format = "xyzw"
        self.quaternion_format = quat_format

    def put(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: Optional[int] = None,
    ) -> None:
        if not action_chunk_dict:
            logger.warning("InterpolationChunkManager: Empty action_chunk_dict, skipping")
            return

        if self.interpolation_ratio <= 1:
            super().put(action_chunk_dict, observation_timestamp)
            return

        last_action_dict = self._get_last_executed_action()

        interpolated_chunk_dict: Dict[str, np.ndarray] = {}
        for device_name, chunk in action_chunk_dict.items():
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            if chunk.size == 0:
                interpolated_chunk_dict[device_name] = chunk.copy()
                continue

            last_action = None
            if last_action_dict and device_name in last_action_dict:
                last_action = last_action_dict[device_name]

            interpolated_chunk = self._interpolate_chunk(chunk, last_action)
            interpolated_chunk_dict[device_name] = interpolated_chunk

            print("original:", chunk)
            print("interpolated:", interpolated_chunk)

        super().put(interpolated_chunk_dict, observation_timestamp)

    def _get_last_executed_action(self) -> Optional[Dict[str, np.ndarray]]:
        if self.current_action_chunk_dict is None:
            return None
        if self.current_step <= 0:
            return None

        last_action_dict: Dict[str, np.ndarray] = {}
        for device_name, chunk in self.current_action_chunk_dict.items():
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            if chunk.shape[0] == 0:
                return None
            step_index = min(max(self.current_step - 1, 0), chunk.shape[0] - 1)
            last_action_dict[device_name] = chunk[step_index]

        return last_action_dict

    def _interpolate_chunk(
        self,
        chunk: np.ndarray,
        last_action: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.interpolation_ratio <= 1:
            return chunk.copy()

        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return chunk.copy()

        action_dim = chunk.shape[1]
        quat_slices = self._get_quaternion_slices(action_dim)

        if last_action is not None:
            last_action = last_action.reshape(-1)
            if last_action.shape[0] != action_dim:
                logger.warning(
                    "InterpolationChunkManager: last_action dim %s != chunk dim %s, skip boundary interpolation",
                    last_action.shape[0],
                    action_dim,
                )
                last_action = None

        if last_action is None:
            interpolated = self._interpolate_sequence(chunk, quat_slices)
        else:
            merged = np.vstack([last_action[None, :], chunk])
            interpolated = self._interpolate_sequence(merged, quat_slices)
            interpolated = interpolated[1:]

        tail_trim = self.interpolation_ratio - 1
        if tail_trim > 0 and interpolated.shape[0] > tail_trim:
            interpolated = interpolated[:-tail_trim]

        return interpolated

    def _get_quaternion_slices(self, action_dim: int) -> List[slice]:
        if action_dim in (8, ):
            return [slice(3, 7)]
        elif action_dim == 16:
            return [slice(3, 7), slice(11, 15)]
        else:
            logger.error(
                "InterpolationChunkManager: Unknown action_dim=%s",
                action_dim,
            )
            raise NotImplementedError

    def _interpolate_sequence(
        self,
        actions: np.ndarray,
        quat_slices: List[slice],
    ) -> np.ndarray:
        ratio = self.interpolation_ratio
        if ratio <= 1:
            return actions.copy()

        num_steps, action_dim = actions.shape
        if num_steps == 0:
            return actions.copy()

        interpolated = np.empty((ratio * num_steps, action_dim), dtype=actions.dtype)
        interpolated[::ratio] = actions

        if num_steps == 1:
            interpolated[1:] = actions[-1]
            return interpolated

        ratios = np.arange(1, ratio, dtype=np.float64) / float(ratio)
        for idx in range(num_steps - 1):
            start = actions[idx].astype(np.float64, copy=False)
            end = actions[idx + 1].astype(np.float64, copy=False)
            deltas = end - start

            segment = start[None, :] + deltas[None, :] * ratios[:, None]
            if quat_slices:
                for quat_slice in quat_slices:
                    slerp_result = self._slerp_quaternion_segment(
                        start[quat_slice],
                        end[quat_slice],
                        ratios,
                    )
                    if slerp_result is not None:
                        segment[:, quat_slice] = slerp_result
                    else:
                        segment[:, quat_slice] = self._normalize_quaternions(
                            segment[:, quat_slice]
                        )

            start_index = idx * ratio + 1
            end_index = start_index + (ratio - 1)
            interpolated[start_index:end_index] = segment.astype(actions.dtype, copy=False)

        interpolated[(num_steps - 1) * ratio + 1 :] = actions[-1]
        return interpolated

    def _slerp_quaternion_segment(
        self,
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        ratios: np.ndarray,
    ) -> Optional[np.ndarray]:
        if ratios.size == 0:
            return np.empty((0, 4), dtype=np.float64)

        start_xyzw = self._to_xyzw(start_quat)
        end_xyzw = self._to_xyzw(end_quat)

        start_xyzw = self._normalize_quaternion(start_xyzw)
        end_xyzw = self._normalize_quaternion(end_xyzw)
        if start_xyzw is None or end_xyzw is None:
            return None

        try:
            rotations = R.from_quat([start_xyzw, end_xyzw])
            slerp = Slerp([0.0, 1.0], rotations)
            interpolated = slerp(ratios).as_quat()
        except Exception as exc:
            logger.warning("InterpolationChunkManager: slerp failed, fallback to lerp: %s", exc)
            interpolated = self._lerp_quaternion(start_xyzw, end_xyzw, ratios)

        if interpolated is None:
            return None

        if self.quaternion_format == "wxyz":
            interpolated = interpolated[:, [3, 0, 1, 2]]
        return interpolated

    def _lerp_quaternion(
        self,
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        ratios: np.ndarray,
    ) -> Optional[np.ndarray]:
        lerped = (1.0 - ratios)[:, None] * start_quat[None, :] + ratios[:, None] * end_quat[None, :]
        return self._normalize_quaternions(quats=lerped)

    def _normalize_quaternion(self, quat: np.ndarray) -> Optional[np.ndarray]:
        if not np.all(np.isfinite(quat)):
            return None
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return None
        return quat / norm

    def _normalize_quaternions(self, quats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return quats / norms

    def _to_xyzw(self, quat: np.ndarray) -> np.ndarray:
        if self.quaternion_format == "xyzw":
            return quat.astype(np.float64, copy=False)
        return quat[[1, 2, 3, 0]].astype(np.float64, copy=False)
