#!/usr/bin/env python3
"""
Rerun Visualization Utilities

This module provides reusable functions and classes for rerun-based 3D visualization,
including trajectory visualization, action data display, and 3D reference grids.

Author: Han Xue
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Literal, Callable
from collections import deque
import rerun as rr
from utils.transform import process_quaternion
from utils.logger_config import logger
import os
import cv2



def _extract_latest_image_frame(data: Any) -> Optional[np.ndarray]:
    if data is None:
        return None

    frame = data
    if isinstance(frame, (list, tuple)):
        if not frame:
            return None
        frame = frame[-1]

    arr = np.asarray(frame)
    if arr.size == 0:
        return None

    # Reduce leading dimensions (e.g., history) down to image tensor
    while arr.ndim > 3 and arr.size > 0:
        arr = arr[-1]

    return arr

def _get_visualizer_app_id() -> str:
    return "policy_connector_visualizer"


def _get_visualizer_world_path() -> str:
    return "trajectory/world"


def _visualize_camera_frames( camera_devices ,obs_dict: Dict[str, Any], timestamp_ns: int) -> None:
    if not camera_devices:
        return

    for camera_name in camera_devices:
        if camera_name not in obs_dict:
            continue

        frame = _extract_latest_image_frame(obs_dict[camera_name])
        if frame is None:
            continue

        camera_path = f"{_get_visualizer_world_path()}/cameras/{camera_name}"
        log_camera_image(
            camera_path,
            frame,
            timestamp_ns,
            center_crop_and_resize=True,
        )


def _extract_robot_pose_history(
    robot_device_name,
    policy_input: Dict[str, Any]
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    if not isinstance(policy_input, dict):
        return None

    robot_device = robot_device_name
    if robot_device is None:
        for key in policy_input.keys():
            lower = key.lower()
            if lower.startswith("robot") or lower.startswith("rizon"):
                robot_device = key
                break

    if robot_device is None or robot_device not in policy_input:
        return None

    raw_sequence = policy_input[robot_device]
    if raw_sequence is None:
        return None

    if isinstance(raw_sequence, np.ndarray):
        robot_obs_array = raw_sequence
    else:
        frames: List[np.ndarray] = []
        for frame in raw_sequence:
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.size == 0:
                continue
            frames.append(arr)
        if not frames:
            return None
        try:
            robot_obs_array = np.stack(frames, axis=0)
        except ValueError:
            return None

    if robot_obs_array.ndim == 1:
        robot_obs_array = robot_obs_array[None, :]

    if robot_obs_array.size == 0:
        return None

    obs_dim = robot_obs_array.shape[-1]
    if obs_dim < 22:
        return None

    pose_history: Dict[str, Dict[str, np.ndarray]] = {}

    def _slice_to_pose(start: int, end: int, arm_name: str) -> None:
        if end > robot_obs_array.shape[1]:
            return
        tcp_segment = robot_obs_array[:, start:end]
        if tcp_segment.shape[1] < 8:
            return
        positions = tcp_segment[:, :3].astype(np.float64, copy=False)
        quats = np.asarray(
            [process_quaternion(tcp_segment[i, 3:7], "f2l") for i in range(len(tcp_segment))],
            dtype=np.float64,
        )
        gripper = tcp_segment[:, 7:8].astype(np.float64, copy=False)
        pose_history[arm_name] = {
            "positions": positions,
            "quaternions": quats,
            "gripper": gripper,
        }

    _slice_to_pose(14, 22, "left_arm")
    if obs_dim >= 44:
        _slice_to_pose(36, 44, "right_arm")

    return pose_history if pose_history else None


@staticmethod
def _get_observation_base_color(arm_name: str) -> List[int]:
    palettes = {
        "left_arm": [90, 170, 255],
        "right_arm": [255, 180, 110],
        "single_arm": [200, 200, 200],
    }
    return palettes.get(arm_name, [200, 200, 200])


@staticmethod
def _build_history_colors(base_color: List[int], length: int) -> List[List[int]]:
    if length <= 0:
        return []
    scales = np.linspace(0.35, 1.0, num=length)
    colors: List[List[int]] = []
    for scale in scales:
        colors.append(
            [
                min(255, int(component * scale))
                for component in base_color
            ]
        )
    return colors


def _visualize_observation_poses(
    robot_device_name, policy_input: Dict[str, Any], timestamp_ns: int
) -> None:

    pose_history = _extract_robot_pose_history(robot_device_name, policy_input)
    if not pose_history:
        return

    try:
        set_time_context(timestamp_ns, "timestamp")
        base_path = f"{_get_visualizer_world_path()}/observation_history"
        for arm_name, pose_data in pose_history.items():
            positions = pose_data.get("positions")
            quaternions = pose_data.get("quaternions")
            if positions is None or quaternions is None:
                continue

            num_points = len(positions)
            if num_points == 0:
                continue

            base_color = _get_observation_base_color(arm_name)
            colors = _build_history_colors(base_color, num_points)
            arm_path = f"{base_path}/{arm_name}"

            if num_points > 1:
                visualize_trajectory_with_rotation(
                    arm_path,
                    positions,
                    quaternions,
                    colors,
                    arrow_scale=0.015,
                    rotation_scale=0.025,
                    show_every_n=num_points,
                )
            else:
                rr.log(
                    f"{arm_path}/pose",
                    rr.Transform3D(
                        translation=positions[0].tolist(),
                        quaternion=quaternions[0].tolist(),
                    ),
                )

            rr.log(
                f"{arm_path}/points",
                rr.Points3D(
                    positions=positions.tolist(),
                    colors=colors,
                ),
            )

            current_pos = positions[-1]
            current_quat = quaternions[-1]
            highlight_color = [min(255, c + 40) for c in base_color]
            rr.log(
                f"{arm_path}/current_pose",
                rr.Transform3D(
                    translation=current_pos.tolist(),
                    quaternion=current_quat.tolist(),
                ),
            )
            rr.log(
                f"{arm_path}/current_pose/point",
                rr.Points3D(
                    positions=[current_pos.tolist()],
                    colors=[highlight_color],
                ),
            )

            gripper = pose_data.get("gripper")
            if gripper is not None and len(gripper) > 0:
                info_text = (
                    f"{arm_name}: {num_points} obs | "
                    f"pos={np.round(current_pos, 4).tolist()} | "
                    f"grip={float(gripper[-1, 0]):.4f}"
                )
            else:
                info_text = f"{arm_name}: {num_points} obs | pos={np.round(current_pos, 4).tolist()}"
            log_text_summary(f"{arm_path}/current_pose/info", info_text)
    except Exception as exc:
        logger.error(f"PolicyConnector: Error visualizing observation poses: {exc}")

def _center_crop_and_resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Center-crop the shorter side and resize to target size."""
    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr

    h, w = arr.shape[:2]
    crop = min(h, w)
    y0 = max(0, (h - crop) // 2)
    x0 = max(0, (w - crop) // 2)
    cropped = arr[y0 : y0 + crop, x0 : x0 + crop]
    resized = cv2.resize(cropped, target_size[::-1], interpolation=cv2.INTER_LINEAR)
    return resized


class RerunVisualizer:
    """
    A reusable class for rerun-based 3D visualization.
    """
    
    def __init__(
        self,
        app_name: str = "visualization",
        spawn: bool = True,
        operating_mode: Optional[Literal["spawn", "connect_grpc", "serve_grpc", "save", "stdout", "none"]] = None,
        connect_uri: Optional[str] = None,
        save_path: Optional[str] = None,
        serve_web_viewer: bool = False
    ) -> None:
        """
        Initialize the rerun visualizer.
        
        Args:
            app_name: Name of the rerun application
            spawn: Whether to spawn a new rerun viewer window (backward compatible)
            operating_mode: One of: "spawn", "connect_grpc", "serve_grpc", "save", "stdout", "none"
            connect_uri: gRPC URI for connect mode, e.g. "grpc://HOST:PORT"
            save_path: Path to .rrd file for save mode
            serve_web_viewer: If True and in serve_grpc mode, also start a web viewer connected to this server
        """
        self.app_name = app_name
        self.is_initialized = False
        self.server_uri: Optional[str] = None
        self.web_viewer_url: Optional[str] = None
        
        try:
            # Allow environment variables to override settings if explicit args not provided
            # RERUN_MODE: spawn|connect_grpc|serve_grpc|save|stdout|none
            env_mode = os.getenv("RERUN_MODE")
            env_connect = os.getenv("RERUN_CONNECT")
            env_save_path = os.getenv("RERUN_SAVE_PATH")
            env_serve_web = os.getenv("RERUN_SERVE_WEB", "0")

            chosen_mode = operating_mode or (env_mode if env_mode else None)
            chosen_connect = connect_uri or (env_connect if env_connect else None)
            chosen_save = save_path or (env_save_path if env_save_path else None)
            chosen_serve_web = serve_web_viewer or (env_serve_web.lower() in ["1", "true", "yes"])

            if chosen_mode is None:
                # Backward-compatible behavior
                rr.init(app_name, spawn=spawn)
                self.is_initialized = True
                logger.info(f"Rerun visualizer '{app_name}' initialized (spawn={spawn})")
            else:
                # Standard operating modes per Rerun documentation
                # https://rerun.io/docs/reference/sdk/operating-modes
                rr.init(app_name)
                self.is_initialized = True
                logger.info(f"Rerun visualizer '{app_name}' initialized (mode={chosen_mode})")

                if chosen_mode == "spawn":
                    rr.spawn()
                    logger.info("Rerun: spawned external viewer via gRPC")
                elif chosen_mode == "connect_grpc":
                    if not chosen_connect:
                        raise ValueError("connect_grpc mode requires connect_uri (e.g. grpc://HOST:PORT)")
                    rr.connect_grpc(chosen_connect)
                    logger.info(f"Rerun: connected to remote viewer at {chosen_connect}")
                elif chosen_mode == "serve_grpc":
                    self.server_uri = rr.serve_grpc()
                    logger.info(f"Rerun: serving gRPC at {self.server_uri}")
                    if chosen_serve_web:
                        # Host a web viewer connected to our gRPC server
                        try:
                            self.web_viewer_url = rr.serve_web_viewer(connect_to=self.server_uri)
                            logger.info(f"Rerun: web viewer served (connected to {self.server_uri})")
                        except Exception as e:
                            logger.error(f"Failed to start web viewer: {e}")
                elif chosen_mode == "save":
                    path = chosen_save or f"{app_name}.rrd"
                    rr.save(path)
                    logger.info(f"Rerun: saving to file '{path}'")
                elif chosen_mode == "stdout":
                    rr.stdout()
                    logger.info("Rerun: streaming to STDOUT")
                elif chosen_mode == "none":
                    logger.info("Rerun: buffering logs in-memory (no sink configured)")
                else:
                    logger.warning(f"Unknown operating mode '{chosen_mode}', falling back to spawn")
                    rr.spawn()
        
        except Exception as e:
            logger.error(f"Failed to initialize rerun visualizer: {e}")

    def start_web_viewer(self) -> None:
        """
        Start a web viewer connected to the current gRPC server if available.
        """
        if not self.is_initialized:
            logger.warning("Rerun visualizer not initialized")
            return
        if not self.server_uri:
            logger.warning("No gRPC server running; cannot start web viewer")
            return
        try:
            self.web_viewer_url = rr.serve_web_viewer(connect_to=self.server_uri)
            logger.info(f"Rerun: web viewer served (connected to {self.server_uri})")
        except Exception as e:
            logger.error(f"Failed to start web viewer: {e}")
    
    def setup_3d_world(self, world_path: str = "world", coordinate_system: str = "z_up") -> None:
        """
        Set up 3D world coordinate system with axes and grid.
        
        Args:
            world_path: Path for the world coordinate system in rerun
            coordinate_system: Either "y_up" or "z_up"
        """
        if not self.is_initialized:
            logger.warning("Rerun visualizer not initialized")
            return
            
        try:
            # Set coordinate system
            if coordinate_system == "z_up":
                rr.log(world_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            else:
                rr.log(world_path, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            
            # Add coordinate axes
            setup_coordinate_axes(f"{world_path}/origin", scale=0.2)
            
            # Add 3D grid
            setup_3d_grid(world_path, grid_size=1.2, grid_step=0.1, coordinate_system=coordinate_system)
            
            logger.info(f"3D world setup complete with {coordinate_system} coordinate system")
            
        except Exception as e:
            logger.error(f"Failed to setup 3D world: {e}")


def setup_coordinate_axes(path: str, scale: float = 0.2, colors: Optional[List[List[int]]] = None) -> None:
    """
    Set up coordinate axes at the specified path.
    
    Args:
        path: Rerun path for the axes
        scale: Scale of the axes arrows
        colors: RGB colors for X, Y, Z axes (default: red, green, blue)
    """
    if colors is None:
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
    
    try:
        rr.log(path, rr.Transform3D(), static=True)
        rr.log(f"{path}/axes", rr.Arrows3D(
            vectors=[[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
            origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            colors=colors,
            labels=["X", "Y", "Z"]
        ), static=True)
    except Exception as e:
        logger.error(f"Failed to setup coordinate axes: {e}")


def log_camera_image(
    path: str,
    image: Optional[np.ndarray],
    timestamp_ns: Optional[int] = None,
    convert_bgr_to_rgb: bool = False,
    center_crop_and_resize: bool = False,
    target_size: Tuple[int, int] = (224, 224),
) -> None:
    """Log a camera image to rerun with optional preprocessing."""
    try:
        if image is None:
            return

        arr = np.asarray(image)
        if arr.size == 0:
            return

        while arr.ndim > 3 and arr.size > 0:
            arr = arr[-1]

        if center_crop_and_resize:
            arr = _center_crop_and_resize_image(arr, target_size)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if convert_bgr_to_rgb and arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., ::-1]

        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        if timestamp_ns is not None:
            set_time_context(timestamp_ns, "timestamp")

        rr.log(path, rr.Image(arr))
    except Exception as e:
        logger.error(f"Failed to log camera image at {path}: {e}")


def setup_3d_grid(world_path: str, grid_size: float = 1.2, grid_step: float = 0.1, 
                  coordinate_system: str = "z_up") -> None:
    """
    Set up 3D reference grid.
    
    Args:
        world_path: Base path for the world
        grid_size: Size of the grid in meters
        grid_step: Step size between grid lines
        coordinate_system: Either "y_up" or "z_up"
    """
    try:
        if coordinate_system == "z_up":
            # Z-up coordinate system
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XY", z_level=0.0, color=[80, 80, 80])    # Floor
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XZ", y_level=0.0, color=[60, 80, 60])    # Front wall
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "YZ", x_level=0.0, color=[80, 60, 60])    # Side wall
            
            # Additional depth perception grids
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XY", z_level=0.6, color=[40, 40, 40])    # Upper grid
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XZ", y_level=-0.6, color=[30, 40, 30])   # Back wall
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "YZ", x_level=-0.6, color=[40, 30, 30])   # Left wall
        else:
            # Y-up coordinate system
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XZ", y_level=0.0, color=[80, 80, 80])    # Floor
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XY", z_level=0.0, color=[60, 80, 60])    # Front wall
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "YZ", x_level=0.0, color=[80, 60, 60])    # Side wall
            
            # Additional depth perception grids
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XZ", y_level=0.6, color=[40, 40, 40])    # Upper grid
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "XY", z_level=-0.6, color=[30, 40, 30])   # Back wall
            create_grid_plane(f"{world_path}/grid", grid_size, grid_step, "YZ", x_level=-0.6, color=[40, 30, 30])   # Left wall
            
    except Exception as e:
        logger.error(f"Failed to setup 3D grid: {e}")


def create_grid_plane(base_path: str, grid_size: float, grid_step: float, plane: str, 
                     x_level: Optional[float] = None, y_level: Optional[float] = None, 
                     z_level: Optional[float] = None, color: List[int] = [100, 100, 100]) -> None:
    """
    Create a grid plane in the specified orientation.
    
    Args:
        base_path: Base path for the grid
        grid_size: Size of the grid
        grid_step: Step size between grid lines
        plane: Plane orientation ("XY", "XZ", or "YZ")
        x_level, y_level, z_level: Fixed coordinate level for the plane
        color: RGB color for the grid lines
    """
    try:
        grid_points: List[List[float]] = []
        
        if plane == "XY" and z_level is not None:
            # XY plane at fixed Z level (horizontal planes - floor/ceiling)
            for i in np.arange(-grid_size, grid_size + grid_step, grid_step):
                # Lines parallel to X axis
                grid_points.extend([[-grid_size, i, z_level], [grid_size, i, z_level]])
                # Lines parallel to Y axis
                grid_points.extend([[i, -grid_size, z_level], [i, grid_size, z_level]])
                
        elif plane == "XZ" and y_level is not None:
            # XZ plane at fixed Y level (vertical wall parallel to X-Z)
            for i in np.arange(-grid_size, grid_size + grid_step, grid_step):
                # Lines parallel to X axis
                grid_points.extend([[-grid_size, y_level, i], [grid_size, y_level, i]])
                # Lines parallel to Z axis
                grid_points.extend([[i, y_level, -grid_size], [i, y_level, grid_size]])
                
        elif plane == "YZ" and x_level is not None:
            # YZ plane at fixed X level (vertical wall parallel to Y-Z)
            for i in np.arange(-grid_size, grid_size + grid_step, grid_step):
                # Lines parallel to Y axis
                grid_points.extend([[x_level, -grid_size, i], [x_level, grid_size, i]])
                # Lines parallel to Z axis
                grid_points.extend([[x_level, i, -grid_size], [x_level, i, grid_size]])
        
        if grid_points:
            plane_name = f"{base_path}_{plane}"
            if x_level is not None:
                plane_name += f"_x{x_level}"
            if y_level is not None:
                plane_name += f"_y{y_level}"
            if z_level is not None:
                plane_name += f"_z{z_level}"
                
            rr.log(plane_name, rr.LineStrips3D(
                strips=[grid_points[i:i+2] for i in range(0, len(grid_points), 2)],
                colors=[color] * (len(grid_points) // 2),
                radii=[0.001]
            ), static=True)
            
    except Exception as e:
        logger.error(f"Failed to create {plane} grid plane: {e}")


def visualize_trajectory_with_colors(path: str, positions: np.ndarray, colors: List[List[int]], 
                                   radii: float = 0.003, episode_boundaries: Optional[List[int]] = None) -> None:
    """
    Visualize a 3D trajectory with color coding.
    
    Args:
        path: Rerun path for the trajectory
        positions: Array of 3D positions [N, 3]
        colors: List of RGB colors for each segment
        radii: Thickness of the trajectory line
        episode_boundaries: List of lengths for each episode to draw separate strips.
    """
    try:
        if len(positions) > 1:
            strips = [positions.tolist()]
            if episode_boundaries and len(episode_boundaries) > 1:
                indices = np.cumsum(episode_boundaries[:-1])
                position_strips = np.split(positions, indices)
                strips = [strip.tolist() for strip in position_strips if len(strip) > 0]

            rr.log(path, rr.LineStrips3D(
                strips=strips,
                colors=colors,
                radii=[radii]
            ))
    except Exception as e:
        logger.error(f"Failed to visualize trajectory: {e}")


def visualize_trajectory_with_arrows(path: str, positions: np.ndarray, colors: List[List[int]], 
                                   arrow_scale: float = 0.02, arrow_step: int = 4) -> None:
    """
    Visualize trajectory with direction arrows.
    
    Args:
        path: Base path for the trajectory
        positions: Array of 3D positions [N, 3]
        colors: List of RGB colors for each point
        arrow_scale: Scale of direction arrows
        arrow_step: Step between arrows (every N points)
    """
    try:
        # Log trajectory line
        visualize_trajectory_with_colors(f"{path}/trajectory", positions, colors)
        
        # Add direction arrows
        if len(positions) > 2:
            arrow_step = max(1, len(positions) // arrow_step)
            
            for i in range(0, len(positions) - 1, arrow_step):
                if i + 1 < len(positions):
                    start_pos = positions[i]
                    end_pos = positions[i + 1]
                    
                    # Calculate direction vector
                    direction = end_pos - start_pos
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.001:  # Avoid zero-length vectors
                        direction = direction / direction_norm * arrow_scale
                        
                        # Arrow color (brighter than trajectory)
                        base_r, base_g, base_b = colors[min(i, len(colors) - 1)]
                        arrow_color = [min(255, int(base_r * 1.5 + 50)),
                                     min(255, int(base_g * 1.5 + 50)),
                                     min(255, int(base_b * 1.5 + 50))]
                        
                        rr.log(f"{path}/direction_arrow_{i}", rr.Arrows3D(
                            vectors=[direction.tolist()],
                            origins=[start_pos.tolist()],
                            colors=[arrow_color],
                            radii=[0.003]
                        ))
                        
    except Exception as e:
        logger.error(f"Failed to visualize trajectory with arrows: {e}")


def visualize_trajectory_with_rotation(path: str, positions: np.ndarray, quaternions: np.ndarray, 
                                     colors: List[List[int]], arrow_scale: float = 0.02, 
                                     rotation_scale: float = 0.03, show_every_n: int = 4,
                                     episode_boundaries: Optional[List[int]] = None) -> None:
    """
    Visualize trajectory with direction arrows and rotation indicators.
    
    Args:
        path: Base path for the trajectory
        positions: Array of 3D positions [N, 3]
        quaternions: Array of quaternions [N, 4] in [x, y, z, w] format
        colors: List of RGB colors for each point
        arrow_scale: Scale of direction arrows
        rotation_scale: Scale of rotation coordinate axes
        show_every_n: Show rotation indicators every N points
        episode_boundaries: List of lengths for each episode to avoid connecting them.
    """
    try:
        # Log trajectory line
        visualize_trajectory_with_colors(f"{path}/trajectory", positions, colors, episode_boundaries=episode_boundaries)
        
        # Add direction arrows and rotation indicators
        if len(positions) > 1:
            episode_end_indices = set(np.cumsum(episode_boundaries) - 1) if episode_boundaries and len(episode_boundaries) > 1 else set()
            step = max(1, len(positions) // show_every_n)
            
            for i in range(0, len(positions), step):
                pos = positions[i]
                quat = quaternions[i] if i < len(quaternions) else quaternions[-1]
                color = colors[min(i, len(colors) - 1)]
                
                # Add direction arrow (if not the last point)
                if i < len(positions) - 1 and i not in episode_end_indices:
                    next_pos = positions[i + 1]
                    direction = next_pos - pos
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.001:
                        direction = direction / direction_norm * arrow_scale
                        
                        # Arrow color (brighter than trajectory)
                        arrow_color = [min(255, int(color[0] * 1.5 + 50)),
                                     min(255, int(color[1] * 1.5 + 50)),
                                     min(255, int(color[2] * 1.5 + 50))]
                        
                        # rr.log(f"{path}/direction_arrow_{i}", rr.Arrows3D(
                        #     vectors=[direction.tolist()],
                        #     origins=[pos.tolist()],
                        #     colors=[arrow_color],
                        #     radii=[0.003]
                        # ))
                
                # Add rotation indicator (local coordinate system)
                rotation_path = f"{path}/rotation_{i}"
                
                # Log the transform at this position with rotation
                rr.log(rotation_path, rr.Transform3D(
                    translation=pos.tolist(),
                    quaternion=quat.tolist()  # [x, y, z, w] format
                ))
                
                # Add small local coordinate axes to show rotation
                axis_colors = [
                    [min(255, int(color[0] * 0.8 + 100)), int(color[1] * 0.3), int(color[2] * 0.3)],  # X-axis (red-ish)
                    [int(color[0] * 0.3), min(255, int(color[1] * 0.8 + 100)), int(color[2] * 0.3)],  # Y-axis (green-ish)
                    [int(color[0] * 0.3), int(color[1] * 0.3), min(255, int(color[2] * 0.8 + 100))]   # Z-axis (blue-ish)
                ]
                
                rr.log(f"{rotation_path}/axes", rr.Arrows3D(
                    vectors=[[rotation_scale, 0.0, 0.0], [0.0, rotation_scale, 0.0], [0.0, 0.0, rotation_scale]],
                    origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    colors=axis_colors,
                    radii=[0.001, 0.001, 0.001]
                ))
                        
    except Exception as e:
        logger.error(f"Failed to visualize trajectory with rotation: {e}")


def gripper_width_to_color(gripper_width: float, min_width: float = 0.01, 
                          max_width: float = 0.08) -> List[int]:
    """
    Convert gripper width to color (red=closed, yellow=mid, green=open).
    
    Args:
        gripper_width: Width of the gripper
        min_width: Minimum gripper width
        max_width: Maximum gripper width
        
    Returns:
        RGB color as [r, g, b]
    """
    try:
        # Normalize gripper width
        normalized_width = np.clip((gripper_width - min_width) / (max_width - min_width), 0.0, 1.0)
        
        # Color interpolation: Red (closed) -> Yellow (mid) -> Green (open)
        if normalized_width < 0.5:
            # Red to Yellow
            r = 255
            g = int(255 * normalized_width * 2)
            b = 0
        else:
            # Yellow to Green
            r = int(255 * (1 - (normalized_width - 0.5) * 2))
            g = 255
            b = 0
        
        return [r, g, b]
        
    except Exception as e:
        logger.error(f"Failed to convert gripper width to color: {e}")
        return [128, 128, 128]  # Default gray

def visualize_action_time_series(
    base_path: str,
    device_name: str,
    arm_name: str,
    positions: np.ndarray,
    grippers: np.ndarray,
    timestamp_ns: int,
    rpy: np.ndarray = None,
    step_offset_ns: int = int(0.04 * 1e9),
    time_timeline: str = "timestamp",
    use_sequence_time: bool = False,
    constant_sequence_timelines: Optional[Dict[str, int]] = None,
    per_step_sequence_timelines: Optional[Dict[str, Callable[[int], int]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    Visualize action data as time series.

    Args:
        base_path: Base path for action trends
        device_name: Name of the device
        arm_name: Name of the arm
        positions: Position data [N, 3]
        grippers: Gripper data [N, 1]
        timestamp_ns: Base timestamp in nanoseconds (ignored when use_sequence_time is True)
        step_offset_ns: Time offset between steps
        time_timeline: Timeline name used for logging the time dimension
        use_sequence_time: Whether to log using sequence indices instead of timestamps
        constant_sequence_timelines: Additional sequence timelines with fixed values per episode (e.g., {'episode': 2})
        per_step_sequence_timelines: Additional sequence timelines whose values depend on the step index
        axis_ranges: Optional axis ranges for rerun blueprint configuration (e.g., {'pos_x': (-0.1, 0.1)})
    """
    try:
        trend_path = f"{base_path}/{device_name}_{arm_name}"
        if constant_sequence_timelines:
            for timeline_name, timeline_value in constant_sequence_timelines.items():
                rr.set_time(timeline_name, sequence=int(timeline_value))

        for step_idx in range(len(positions)):
            if per_step_sequence_timelines:
                for (
                    timeline_name,
                    timeline_generator,
                ) in per_step_sequence_timelines.items():
                    rr.set_time(
                        timeline_name, sequence=int(timeline_generator(step_idx))
                    )

            if use_sequence_time:
                rr.set_time(time_timeline, sequence=step_idx)
            else:
                step_time = int(timestamp_ns + step_idx * step_offset_ns)
                # Use new set_time API (seconds)
                rr.set_time(time_timeline, timestamp=1e-9 * step_time)

            pos = positions[step_idx]
            pos = np.asarray(pos, dtype=float)
            if rpy is not None:
                pose3d = rpy[step_idx]
                pose3d = np.asarray(pose3d, dtype=float) 
            gripper_entry = np.asarray(grippers[step_idx])
            gripper_value = (
                float(gripper_entry.flatten()[0])
                if gripper_entry.size > 0
                else float(grippers[step_idx])
            )
            # Log position components
            
            rr.log(f"{trend_path}/pos_x", rr.Scalars(float(pos[0])))
            rr.log(f"{trend_path}/pos_y", rr.Scalars(float(pos[1])))
            rr.log(f"{trend_path}/pos_z", rr.Scalars(float(pos[2])))
            if rpy is not None:
                rr.log(f"{trend_path}/r", rr.Scalars(float(pose3d[0])))
                rr.log(f"{trend_path}/p", rr.Scalars(float(pose3d[1])))
                rr.log(f"{trend_path}/y", rr.Scalars(float(pose3d[2])))
            rr.log(f"{trend_path}/gripper", rr.Scalars(gripper_value))

        # Note: axis_ranges configuration is no longer supported in current Rerun version
        # The ComponentBatch API has been removed. Rerun now handles axis ranges automatically
        # or through blueprint configuration in the viewer UI.
        if axis_ranges:
            logger.debug(
                f"Axis ranges specified for {trend_path} but automatic scaling will be used instead"
            )

    except Exception as e:
        logger.error(f"Failed to visualize action time series: {e}")


def visualize_poses_with_axes(path: str, positions: np.ndarray, quaternions: np.ndarray, 
                             axis_scale: float = 0.03) -> None:
    """
    Visualize poses with coordinate axes at each position.
    
    Args:
        path: Base path for the poses
        positions: Position data [N, 3]
        quaternions: Quaternion data [N, 4] in [x, y, z, w] format
        axis_scale: Scale of the coordinate axes
    """
    try:
        for i, (pos, quat) in enumerate(zip(positions, quaternions)):
            waypoint_path = f"{path}/waypoint_{i}"
            
            # Log the waypoint transform
            rr.log(waypoint_path, rr.Transform3D(
                translation=pos.tolist(),
                quaternion=quat.tolist()  # Rerun expects [x,y,z,w] format
            ))
            
            # Add coordinate axes for this waypoint
            rr.log(f"{waypoint_path}/axes", rr.Arrows3D(
                vectors=[[axis_scale, 0.0, 0.0], [0.0, axis_scale, 0.0], [0.0, 0.0, axis_scale]],
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]],
                radii=[0.002, 0.002, 0.002]
            ))
            
    except Exception as e:
        logger.error(f"Failed to visualize poses with axes: {e}")


def log_text_summary(path: str, text: str) -> None:
    """
    Log text summary at the specified path.
    
    Args:
        path: Rerun path for the text
        text: Text content to log
    """
    try:
        rr.log(path, rr.TextLog(text))
    except Exception as e:
        logger.error(f"Failed to log text summary: {e}")


def set_time_context(timestamp_ns: int, time_type: str = "timestamp") -> None:
    """
    Set time context for rerun logging.
    
    Args:
        timestamp_ns: Timestamp in nanoseconds
        time_type: Type of time context
    """
    try:
        rr.set_time(time_type, timestamp=1e-9 * timestamp_ns)
    except Exception as e:
        logger.error(f"Failed to set time context: {e}")


class OpenXRTrackerVisualizer:
    """
    A specialized visualizer for OpenXR-based Vive Trackers with real-time 3D visualization.
    """
    
    def __init__(
        self,
        app_name: str = "openxr_tracker_visualization",
        spawn: bool = True,
        operating_mode: Optional[Literal["spawn", "connect_grpc", "serve_grpc", "save", "stdout", "none"]] = None,
        connect_uri: Optional[str] = None,
        save_path: Optional[str] = None,
        serve_web_viewer: bool = False
    ) -> None:
        """
        Initialize the OpenXR tracker visualizer.
        
        Args:
            app_name: Name of the rerun application
            spawn: Whether to spawn a new rerun viewer window
        """
        self.app_name = app_name
        self.is_initialized = False
        self.tracker_trajectories: Dict[str, deque] = {}
        self.max_trajectory_points = 200
        
        self.server_uri: Optional[str] = None
        self.web_viewer_url: Optional[str] = None
        
        try:
            # Environment overrides
            env_mode = os.getenv("RERUN_MODE")
            env_connect = os.getenv("RERUN_CONNECT")
            env_save_path = os.getenv("RERUN_SAVE_PATH")
            env_serve_web = os.getenv("RERUN_SERVE_WEB", "0")

            chosen_mode = operating_mode or (env_mode if env_mode else None)
            chosen_connect = connect_uri or (env_connect if env_connect else None)
            chosen_save = save_path or (env_save_path if env_save_path else None)
            chosen_serve_web = serve_web_viewer or (env_serve_web.lower() in ["1", "true", "yes"])

            if chosen_mode is None:
                rr.init(app_name, spawn=spawn)
                self.is_initialized = True
                logger.info(f"OpenXR tracker visualizer '{app_name}' initialized (spawn={spawn})")
            else:
                rr.init(app_name)
                self.is_initialized = True
                logger.info(f"OpenXR tracker visualizer '{app_name}' initialized (mode={chosen_mode})")

                if chosen_mode == "spawn":
                    rr.spawn()
                elif chosen_mode == "connect_grpc":
                    if not chosen_connect:
                        raise ValueError("connect_grpc mode requires connect_uri (e.g. grpc://HOST:PORT)")
                    rr.connect_grpc(chosen_connect)
                elif chosen_mode == "serve_grpc":
                    self.server_uri = rr.serve_grpc()
                    if chosen_serve_web:
                        try:
                            self.web_viewer_url = rr.serve_web_viewer(connect_to=self.server_uri)
                        except Exception as e:
                            logger.error(f"Failed to start web viewer: {e}")
                elif chosen_mode == "save":
                    path = chosen_save or f"{app_name}.rrd"
                    rr.save(path)
                elif chosen_mode == "stdout":
                    rr.stdout()
                elif chosen_mode == "none":
                    pass
                else:
                    rr.spawn()

            # Setup 3D world coordinate system
            self.setup_openxr_world()
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenXR tracker visualizer: {e}")
    
    def setup_openxr_world(self) -> None:
        """Set up 3D world coordinate system optimized for OpenXR tracking."""
        if not self.is_initialized:
            logger.warning("Visualizer not initialized")
            return
            
        try:
            # Set coordinate system (OpenXR typically uses right-handed Y-up)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            
            # Add coordinate axes at origin
            setup_coordinate_axes("world/origin", scale=0.3)
            
            # Add 3D grid optimized for room-scale tracking
            setup_3d_grid("world", grid_size=2.5, grid_step=0.25, coordinate_system="y_up")
            
            logger.info("OpenXR world setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenXR world: {e}")
    
    def log_tracker_pose(self, tracker_name: str, position: List[float], 
                        quaternion: List[float], timestamp_ns: Optional[int] = None) -> None:
        """
        Log a tracker's pose with position and rotation.
        
        Args:
            tracker_name: Name/ID of the tracker
            position: Position as [x, y, z]
            quaternion: Quaternion as [x, y, z, w] (rerun format)
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if not self.is_initialized:
            return
            
        try:
            # Set time context if provided
            if timestamp_ns is not None:
                rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
            
            tracker_path = f"world/tracker_{tracker_name}"
            
            # Log the tracker's transform
            rr.log(tracker_path, rr.Transform3D(
                translation=position,
                quaternion=quaternion  # [x, y, z, w] format
            ))
            
            # Add coordinate axes for this tracker
            rr.log(f"{tracker_path}/axes", rr.Arrows3D(
                vectors=[[0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]],
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]],
                radii=[0.004, 0.004, 0.004],
                labels=["X", "Y", "Z"]
            ))
            
            # Update trajectory
            self._update_tracker_trajectory(tracker_name, position)
            
        except Exception as e:
            logger.error(f"Failed to log tracker pose for {tracker_name}: {e}")
    
    def _update_tracker_trajectory(self, tracker_name: str, position: List[float]) -> None:
        """Update and visualize tracker trajectory."""
        try:
            # Initialize trajectory storage for new trackers
            if tracker_name not in self.tracker_trajectories:
                self.tracker_trajectories[tracker_name] = deque(maxlen=self.max_trajectory_points)
            
            # Add current position to trajectory
            self.tracker_trajectories[tracker_name].append(position)
            
            # Visualize trajectory if we have multiple points
            if len(self.tracker_trajectories[tracker_name]) > 1:
                trajectory_points = list(self.tracker_trajectories[tracker_name])
                
                # Generate color gradient for trajectory (newer points brighter)
                trajectory_colors = []
                num_points = len(trajectory_points)
                for i in range(num_points):
                    # Fade from dark to bright along trajectory
                    intensity = int(100 + (155 * i / max(1, num_points - 1)))
                    trajectory_colors.append([intensity, intensity, 0])  # Yellow gradient
                
                rr.log(f"world/trajectory_{tracker_name}", rr.LineStrips3D(
                    strips=[trajectory_points],
                    colors=trajectory_colors,
                    radii=[0.003]
                ))
                
        except Exception as e:
            logger.error(f"Failed to update trajectory for {tracker_name}: {e}")
    
    def log_multiple_trackers(self, tracker_data: Dict[str, Dict[str, Any]], 
                             timestamp_ns: Optional[int] = None) -> None:
        """
        Log poses for multiple trackers at once.
        
        Args:
            tracker_data: Dict mapping tracker names to pose data
                         Each pose data should have 'position' and 'quaternion' keys
            timestamp_ns: Optional timestamp in nanoseconds
        """
        if not self.is_initialized:
            return
            
        try:
            # Set time context if provided
            if timestamp_ns is not None:
                rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
            
            for tracker_name, pose_data in tracker_data.items():
                if 'position' in pose_data and 'quaternion' in pose_data:
                    self.log_tracker_pose(
                        tracker_name=tracker_name,
                        position=pose_data['position'],
                        quaternion=pose_data['quaternion'],
                        timestamp_ns=None  # Already set above
                    )
                    
        except Exception as e:
            logger.error(f"Failed to log multiple trackers: {e}")
    
    def clear_trajectories(self) -> None:
        """Clear all tracker trajectories."""
        try:
            for tracker_name in self.tracker_trajectories:
                self.tracker_trajectories[tracker_name].clear()
                # Clear trajectory visualization
                rr.log(f"world/trajectory_{tracker_name}", rr.Clear(recursive=True))
            logger.info("All tracker trajectories cleared")
        except Exception as e:
            logger.error(f"Failed to clear trajectories: {e}")
    
    def log_tracker_info(self, info_text: str) -> None:
        """
        Log tracker information as text.
        
        Args:
            info_text: Information text to display
        """
        try:
            rr.log("info/tracker_status", rr.TextLog(info_text))
        except Exception as e:
            logger.error(f"Failed to log tracker info: {e}")


def openxr_pose_to_rerun_format(openxr_pose) -> Tuple[List[float], List[float]]:
    """
    Convert OpenXR pose format to rerun-compatible format.
    
    Args:
        openxr_pose: OpenXR pose object with position and orientation
        
    Returns:
        tuple: (position [x,y,z], quaternion [x,y,z,w])
    """
    try:
        # Extract position (typically as Vector3f)
        position = [
            float(openxr_pose.position.x),
            float(openxr_pose.position.y), 
            float(openxr_pose.position.z)
        ]
        
        # Extract quaternion (typically as Quaternionf) and convert to [x,y,z,w] format
        quaternion = [
            float(openxr_pose.orientation.x),
            float(openxr_pose.orientation.y),
            float(openxr_pose.orientation.z),
            float(openxr_pose.orientation.w)
        ]
        
        return position, quaternion
        
    except Exception as e:
        logger.error(f"Failed to convert OpenXR pose: {e}")
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
