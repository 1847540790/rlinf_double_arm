"""
Standalone pose conversion functions without lib_py dependencies.
Only uses third-party libraries (numpy, transforms3d).
"""

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def pose_to_4x4(pose: np.ndarray) -> np.ndarray:
    """Convert pose into 4x4 format.

    Args:
        pose: numpy array, in 6D (axis angle), 7D or 4x4 format

    Returns:
        pose: numpy array, in 4x4 format
    """
    if pose.size == 16:
        if pose.shape == (4, 4):
            return pose
        else:
            return pose.reshape(4, 4)

    if pose.size == 7:
        pose = pose.reshape(7)
        pos = pose[0:3]
        quat = pose[3:7]
        # Convert quaternion to rotation matrix
        mat = quat2mat(quat)
        # Build 4x4 pose matrix
        new_pose = np.zeros([4, 4])
        new_pose[0:3, 0:3] = mat
        new_pose[0:3, -1] = pos
        new_pose[-1, -1] = 1
        return new_pose

    if pose.size == 6:
        # Convert axis-angle to rotation matrix
        axis_angle = pose[3:]
        rot_mat = _axis_angle_to_rot_mat(axis_angle)
        # Build 4x4 pose matrix
        new_pose = np.zeros([4, 4])
        new_pose[0:3, 0:3] = rot_mat
        new_pose[0:3, -1] = pose[:3]
        new_pose[-1, -1] = 1
        return new_pose

    raise RuntimeError("unknown size: " + str(pose.size))


def pose_to_7D(pose: np.ndarray) -> np.ndarray:
    """Convert pose into 7D format.

    Args:
        pose: numpy array, in 6D (axis angle), 7D or 4x4 format

    Returns:
        pose: numpy array, in 7D format
    """
    if pose.size == 7:
        if pose.shape == (7,):
            return pose
        else:
            return pose.reshape(7)

    # Convert to 4x4 first
    pose_4x4 = pose_to_4x4(pose)
    # Extract position and rotation matrix
    pos = pose_4x4[0:3, -1]
    rot_mat = pose_4x4[0:3, 0:3]
    # Convert rotation matrix to quaternion
    quat = mat2quat(rot_mat)
    # Build 7D pose
    new_pose = np.zeros(7)
    new_pose[0:3] = pos
    new_pose[3:7] = quat
    return new_pose


def _axis_angle_to_rot_mat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix.

    This is a simplified version for single axis-angle (not batch).

    Args:
        axis_angle: numpy array of shape (3,), axis-angle representation

    Returns:
        rot_mat: numpy array of shape (3, 3), rotation matrix
    """
    axis_angle_norm = np.linalg.norm(axis_angle + 1e-8)
    if axis_angle_norm < 1e-8:
        # Zero rotation, return identity
        return np.eye(3)

    angle = axis_angle_norm * 0.5
    axis_angle_normalized = axis_angle / axis_angle_norm

    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.array(
        [
            v_cos,
            v_sin * axis_angle_normalized[0],
            v_sin * axis_angle_normalized[1],
            v_sin * axis_angle_normalized[2],
        ]
    )

    # Convert quaternion to rotation matrix
    return _quat_to_rot_mat(quat)


def _quat_to_rot_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    This is a simplified version for single quaternion (not batch).

    Args:
        quat: numpy array of shape (4,), quaternion [w, x, y, z]

    Returns:
        rot_mat: numpy array of shape (3, 3), rotation matrix
    """
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = np.array(
        [
            [w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz],
            [2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx],
            [2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
        ]
    )
    return rot_mat
