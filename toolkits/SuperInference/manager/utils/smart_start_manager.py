#!/usr/bin/env python3
"""
Smart Start Chunk Manager - intelligently finds optimal start index for action chunk connection.

This manager:
1. Calculates delay based on observation timestamp and current time
2. Finds optimal start index by searching for closest action to current executing action
3. Separately optimizes robot and gripper start indices for better continuity

Author: Based on chunk_connect branch implementation
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils.logger_config import logger
from utils.transform import calc_action_distance, calc_position_distance, calc_angle_diff, calc_ee_diff
from .action_manager import BaseChunkManager


class SmartStartChunkManager(BaseChunkManager):
    """
    Smart start manager that finds optimal start index for seamless action chunk connection.
    
    This manager eliminates obsolete actions from each new action chunk by:
    1. Calculating total delay (observation + read/write + inference)
    2. Finding current executing action from previous chunk
    3. Searching for closest action in new chunk within tolerance ranges
    4. Using either separate or merged optimization mode
    
    Modes:
        - 'separate': Separately optimize robot and gripper start indices with their own search ranges
        - 'merged': Compute unified distance combining robot and gripper, use single optimal start index
    
    Args:
        dt (float): Duration between action steps in seconds (typically 1/execution_fps)
        execution_fps (float): Execution frequency in Hz (alternative to dt)
        mode (str): Optimization mode - 'separate' or 'merged' (default: 'separate')
        robot_search_tolerance (int): Search range tolerance around delayed start index for robot matching
        gripper_search_tolerance (int): Search range tolerance around delayed start index for gripper matching
        search_tolerance (int): Deprecated - unified search tolerance (kept for backward compatibility)
        pos_weight (float): Weight for position distance in robot action matching
        rot_weight (float): Weight for rotation distance in robot action matching
        ee_weight (float): Weight for end-effector distance in gripper action matching
    """
    
    def __init__(
        self, 
        dt: Optional[float] = None,
        execution_fps: Optional[float] = None,
        mode: str = 'separate',
        robot_search_tolerance: int = 2,
        gripper_search_tolerance: int = 2,
        search_tolerance: Optional[int] = None,  # Deprecated, kept for backward compatibility
        pos_weight: float = 0.5,
        rot_weight: float = 0.5,
        ee_weight: float = 1.0
    ):
        super().__init__()
        
        # Validate and set mode
        if mode not in ['separate', 'merged']:
            logger.warning(f"SmartStartChunkManager: Invalid mode '{mode}', defaulting to 'separate'")
            mode = 'separate'
        self.mode = mode
        
        # Calculate dt from execution_fps if provided, otherwise use dt parameter
        if execution_fps is not None:
            dt_seconds = 1.0 / execution_fps
            self.dt = int(dt_seconds * 1e9)  # Convert seconds to nanoseconds
            logger.info(f"SmartStartChunkManager: Calculated dt from execution_fps={execution_fps}Hz: "
                       f"dt={dt_seconds}s ({self.dt}ns)")
        elif dt is not None:
            self.dt = int(dt * 1e9)  # Convert seconds to nanoseconds
            logger.info(f"SmartStartChunkManager: Using provided dt={dt}s ({self.dt}ns)")
        else:
            # Default fallback
            self.dt = int(0.01 * 1e9)  # Default 10ms
            logger.warning(f"SmartStartChunkManager: Neither dt nor execution_fps provided, "
                          f"using default dt=0.01s ({self.dt}ns)")
        
        # Handle backward compatibility: if search_tolerance is provided, use it for both
        if search_tolerance is not None:
            logger.warning(f"SmartStartChunkManager: 'search_tolerance' is deprecated. "
                          f"Use 'robot_search_tolerance' and 'gripper_search_tolerance' instead.")
            self.robot_search_tolerance = search_tolerance
            self.gripper_search_tolerance = search_tolerance
        else:
            self.robot_search_tolerance = robot_search_tolerance
            self.gripper_search_tolerance = gripper_search_tolerance
        
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.ee_weight = ee_weight
        
        logger.info(f"SmartStartChunkManager initialized: mode={self.mode}, dt={self.dt}ns, "
                   f"robot_search_tolerance={self.robot_search_tolerance}, "
                   f"gripper_search_tolerance={self.gripper_search_tolerance}, "
                   f"pos_weight={pos_weight}, rot_weight={rot_weight}, ee_weight={ee_weight}")
    
    def put(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: Optional[int] = None) -> None:
        """
        Process new action chunk by finding optimal start index and clipping obsolete actions.
        
        Args:
            action_chunk_dict: New action chunk dictionary from policy
            observation_timestamp: Timestamp when observation was taken (nanoseconds)
        """
        if observation_timestamp is None:
            logger.warning("SmartStartChunkManager: observation_timestamp is None, using current time")
            observation_timestamp = time.time_ns()
        
        # Calculate total delay (includes observation, read/write, and inference delays)
        current_time_ns = time.time_ns()
        delay = current_time_ns - observation_timestamp
        delayed_start_idx = delay // self.dt
        
        logger.debug(f"SmartStartChunkManager: delay={delay/1e9:.4f}s, delayed_start_idx={delayed_start_idx}")
        
        # Get chunk length from any device (they should all have the same chunk length)
        device_names = list(action_chunk_dict.keys())
        if not device_names:
            logger.warning("SmartStartChunkManager: Empty action_chunk_dict, skipping")
            return
        
        first_chunk = action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)
        
        # If this is the first chunk, just store it
        if self.current_action_chunk_dict is None:
            logger.info(f"SmartStartChunkManager: First chunk received, length={chunk_length}")
            super().put(action_chunk_dict, observation_timestamp)
            return
        
        # Get current executing action from previous chunk
        action_current = self._get_current_executing_action()
        if action_current is None:
            logger.warning("SmartStartChunkManager: Cannot get current executing action, using delayed_start_idx")
            # Fallback to simple delay-based clipping
            if delayed_start_idx < chunk_length:
                clipped_chunk_dict = {}
                for device_name, chunk in action_chunk_dict.items():
                    clipped_chunk_dict[device_name] = chunk[delayed_start_idx:]
                super().put(clipped_chunk_dict, observation_timestamp)
                logger.info(f"SmartStartChunkManager: Clipped {delayed_start_idx} actions using delay-based method")
            else:
                logger.warning(f"SmartStartChunkManager: Delay too large ({delay/1e9:.4f}s), dropping chunk")
            return
        
        # Process based on mode
        if self.mode == 'merged':
            # Merged mode: use unified search range and combined distance
            optimized_chunk_dict = self._process_merged_mode(
                action_chunk_dict, action_current, delayed_start_idx, chunk_length
            )
        else:
            # Separate mode: use independent search ranges for robot and gripper
            optimized_chunk_dict = self._process_separate_mode(
                action_chunk_dict, action_current, delayed_start_idx, chunk_length
            )
        
        if optimized_chunk_dict:
            super().put(optimized_chunk_dict, observation_timestamp)
            total_clipped = chunk_length - len(list(optimized_chunk_dict.values())[0])
            logger.info(f"SmartStartChunkManager: Successfully processed chunk, clipped {total_clipped} actions on average")
        else:
            logger.warning("SmartStartChunkManager: No optimized chunks, dropping entire chunk")
    
    def _process_merged_mode(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        action_current: Dict[str, np.ndarray],
        delayed_start_idx: int,
        chunk_length: int
    ) -> Dict[str, np.ndarray]:
        """
        Process action chunk using merged mode: unified distance combining robot and gripper.
        
        Args:
            action_chunk_dict: New action chunk dictionary
            action_current: Current executing actions
            delayed_start_idx: Estimated start index based on delay
            chunk_length: Length of action chunks
        
        Returns:
            Optimized action chunk dictionary
        """
        # Use robot_search_tolerance as the unified search tolerance
        search_start = max(0, delayed_start_idx - self.robot_search_tolerance)
        search_end = min(chunk_length - 1, delayed_start_idx + self.robot_search_tolerance)
        
        if search_start > search_end:
            logger.warning(f"SmartStartChunkManager: Invalid search range [{search_start}, {search_end}], using delayed_start_idx")
            search_start = max(0, min(delayed_start_idx, chunk_length - 1))
            search_end = search_start
        
        logger.info(f"SmartStartChunkManager (merged): delayed_start_idx={delayed_start_idx}, chunk_length={chunk_length}")
        logger.info(f"SmartStartChunkManager (merged): Unified search range=[{search_start}, {search_end}]")
        
        # Find optimal start index for each device using combined distance
        device_names = list(action_chunk_dict.keys())
        best_indices = []
        min_distances = []
        
        for device_name in device_names:
            if device_name not in action_current:
                logger.warning(f"SmartStartChunkManager (merged): Device {device_name} not in current action, skipping")
                continue
            
            current_action = action_current[device_name]
            new_chunk = action_chunk_dict[device_name]
            
            # Search for closest action using combined distance
            min_distance = float('inf')
            best_index = search_start
            
            for idx in range(search_start, search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                
                # Calculate combined distance based on action dimension
                if len(candidate_action) == 16:
                    # Dual-arm case: average distance of left and right arms
                    left_dist = calc_action_distance(
                        current_action[:8], candidate_action[:8],
                        pos_weight=self.pos_weight,
                        rot_weight=self.rot_weight,
                        ee_weight=self.ee_weight
                    )
                    right_dist = calc_action_distance(
                        current_action[8:16], candidate_action[8:16],
                        pos_weight=self.pos_weight,
                        rot_weight=self.rot_weight,
                        ee_weight=self.ee_weight
                    )
                    distance = (left_dist + right_dist) / 2.0
                else:
                    # Single arm or robot-only case
                    distance = calc_action_distance(
                        current_action, candidate_action,
                        pos_weight=self.pos_weight,
                        rot_weight=self.rot_weight,
                        ee_weight=self.ee_weight
                    )
                
                if distance < min_distance:
                    min_distance = distance
                    best_index = idx
            
            best_indices.append(best_index)
            min_distances.append(min_distance)
            logger.info(f"SmartStartChunkManager (merged): Device {device_name} - "
                       f"best_index={best_index}, min_distance={min_distance:.6f}")
        
        # Average the best indices from all devices and apply to all
        if not best_indices:
            logger.warning("SmartStartChunkManager (merged): No valid devices found")
            return {}
        
        optimal_index = int(np.round(np.mean(best_indices)))
        avg_distance = float(np.mean(min_distances))
        
        logger.info(f"SmartStartChunkManager (merged): optimal_index={optimal_index} "
                   f"(from indices {best_indices}), avg_distance={avg_distance:.6f}, "
                   f"skipping {optimal_index} obsolete actions")
        
        # Apply optimal index to all devices
        optimized_chunk_dict = {}
        for device_name in device_names:
            chunk = action_chunk_dict[device_name]
            if optimal_index < len(chunk):
                optimized_chunk_dict[device_name] = chunk[optimal_index:]
            else:
                logger.warning(f"SmartStartChunkManager (merged): optimal_index {optimal_index} >= chunk_length for {device_name}")
        
        return optimized_chunk_dict
    
    def _process_separate_mode(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        action_current: Dict[str, np.ndarray],
        delayed_start_idx: int,
        chunk_length: int
    ) -> Dict[str, np.ndarray]:
        """
        Process action chunk using separate mode: independent search for robot and gripper.
        
        Args:
            action_chunk_dict: New action chunk dictionary
            action_current: Current executing actions
            delayed_start_idx: Estimated start index based on delay
            chunk_length: Length of action chunks
        
        Returns:
            Optimized action chunk dictionary
        """
        # Calculate search ranges for robot and gripper separately
        robot_search_start = max(0, delayed_start_idx - self.robot_search_tolerance)
        robot_search_end = min(chunk_length - 1, delayed_start_idx + self.robot_search_tolerance)
        
        gripper_search_start = max(0, delayed_start_idx - self.gripper_search_tolerance)
        gripper_search_end = min(chunk_length - 1, delayed_start_idx + self.gripper_search_tolerance)
        
        if robot_search_start > robot_search_end:
            logger.warning(f"SmartStartChunkManager (separate): Invalid robot search range [{robot_search_start}, {robot_search_end}], using delayed_start_idx")
            robot_search_start = max(0, min(delayed_start_idx, chunk_length - 1))
            robot_search_end = robot_search_start
        
        if gripper_search_start > gripper_search_end:
            logger.warning(f"SmartStartChunkManager (separate): Invalid gripper search range [{gripper_search_start}, {gripper_search_end}], using delayed_start_idx")
            gripper_search_start = max(0, min(delayed_start_idx, chunk_length - 1))
            gripper_search_end = gripper_search_start
        
        logger.info(f"SmartStartChunkManager (separate): delayed_start_idx={delayed_start_idx}, chunk_length={chunk_length}")
        logger.info(f"SmartStartChunkManager (separate): Robot search range=[{robot_search_start}, {robot_search_end}], "
                   f"Gripper search range=[{gripper_search_start}, {gripper_search_end}]")
        
        # Find optimal start indices for each device
        device_names = list(action_chunk_dict.keys())
        optimized_chunk_dict = {}
        
        for device_name in device_names:
            if device_name not in action_current:
                logger.warning(f"SmartStartChunkManager (separate): Device {device_name} not in current action, using delayed_start_idx")
                chunk = action_chunk_dict[device_name]
                if delayed_start_idx < chunk_length:
                    optimized_chunk_dict[device_name] = chunk[delayed_start_idx:]
                else:
                    logger.warning(f"SmartStartChunkManager (separate): Dropping chunk for {device_name} due to large delay")
                continue
            
            current_action = action_current[device_name]
            new_chunk = action_chunk_dict[device_name]
            action_dim = len(current_action)
            
            # Find optimal start indices separately for robot and gripper
            robot_start_idx, gripper_start_idx = self._find_optimal_start_indices(
                current_action=current_action,
                new_chunk=new_chunk,
                robot_search_start=robot_search_start,
                robot_search_end=robot_search_end,
                gripper_search_start=gripper_search_start,
                gripper_search_end=gripper_search_end
            )
            
            logger.info(f"SmartStartChunkManager (separate): Device {device_name} - "
                       f"robot_start_idx={robot_start_idx}, gripper_start_idx={gripper_start_idx}")
            
            # Extract robot and gripper parts separately from their respective start indices
            if robot_start_idx >= chunk_length or gripper_start_idx >= chunk_length:
                logger.warning(f"SmartStartChunkManager (separate): Start indices out of range for {device_name} "
                             f"(robot={robot_start_idx}, gripper={gripper_start_idx}, chunk_length={chunk_length}), "
                             f"dropping chunk")
                continue
            
            # Handle different action dimensions
            if action_dim == 16:
                # Dual-arm case: [left_robot(7), left_gripper(1), right_robot(7), right_gripper(1)]
                # Extract robot parts (left and right) from robot_start_idx
                robot_chunk = new_chunk[robot_start_idx:, :]
                robot_left_part = robot_chunk[:, 0:7]  # Left arm robot
                robot_right_part = robot_chunk[:, 8:15]  # Right arm robot
                
                # Extract gripper parts (left and right) from gripper_start_idx
                gripper_chunk = new_chunk[gripper_start_idx:, :]
                gripper_left_part = gripper_chunk[:, 7:8]  # Left arm gripper
                gripper_right_part = gripper_chunk[:, 15:16]  # Right arm gripper
                
                # Calculate lengths and use minimum
                robot_length = len(robot_chunk)
                gripper_length = len(gripper_chunk)
                final_length = min(robot_length, gripper_length)
                
                # Truncate to final_length and concatenate
                robot_left_final = robot_left_part[:final_length, :]
                robot_right_final = robot_right_part[:final_length, :]
                gripper_left_final = gripper_left_part[:final_length, :]
                gripper_right_final = gripper_right_part[:final_length, :]
                
                # Concatenate: [left_robot, left_gripper, right_robot, right_gripper]
                optimized_chunk = np.concatenate([
                    robot_left_final, gripper_left_final,
                    robot_right_final, gripper_right_final
                ], axis=1)
                
                logger.debug(f"SmartStartChunkManager (separate): Dual-arm - robot_length={robot_length}, "
                           f"gripper_length={gripper_length}, final_length={final_length}")
            
            elif action_dim >= 8:
                # Single arm with gripper: [robot(7), gripper(1+)]
                # Extract robot part (first 7 dims) from robot_start_idx
                robot_chunk = new_chunk[robot_start_idx:, 0:7]
                
                # Extract gripper part (remaining dims) from gripper_start_idx
                gripper_chunk = new_chunk[gripper_start_idx:, 7:]
                
                # Calculate lengths and use minimum
                robot_length = len(robot_chunk)
                gripper_length = len(gripper_chunk)
                final_length = min(robot_length, gripper_length)
                
                # Truncate to final_length and concatenate
                robot_final = robot_chunk[:final_length, :]
                gripper_final = gripper_chunk[:final_length, :]
                
                # Concatenate robot and gripper
                optimized_chunk = np.concatenate([robot_final, gripper_final], axis=1)
                
                logger.debug(f"SmartStartChunkManager (separate): Single-arm - robot_length={robot_length}, "
                           f"gripper_length={gripper_length}, final_length={final_length}")
            
            else:
                # 7D case: only robot, no gripper
                optimized_chunk = new_chunk[robot_start_idx:, :]
                final_length = len(optimized_chunk)
                logger.debug(f"SmartStartChunkManager (separate): 7D action - final_length={final_length}")
            
            optimized_chunk_dict[device_name] = optimized_chunk
            logger.info(f"SmartStartChunkManager (separate): Device {device_name} - "
                       f"optimized chunk length={final_length}, "
                       f"robot_start_idx={robot_start_idx}, gripper_start_idx={gripper_start_idx}")
        
        return optimized_chunk_dict
    
    def _get_current_executing_action(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current executing action from previous chunk based on current_step.
        
        Returns:
            Dictionary of current actions for each device, or None if not available
        """
        if self.current_action_chunk_dict is None:
            return None
        
        if self.current_step < 0:
            return None
        
        # Get chunk length to validate current_step
        device_names = list(self.current_action_chunk_dict.keys())
        if not device_names:
            return None
        
        first_chunk = self.current_action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)
        
        # If current_step is beyond chunk length, use the last action
        step_index = min(self.current_step, chunk_length - 1)
        
        action_current = {}
        for device_name, chunk in self.current_action_chunk_dict.items():
            if step_index < len(chunk):
                action_current[device_name] = chunk[step_index]
            else:
                logger.warning(f"SmartStartChunkManager: step_index {step_index} out of range for {device_name}")
                return None
        
        logger.debug(f"SmartStartChunkManager: Current executing action at step {step_index}/{chunk_length}")
        return action_current
    
    def _find_optimal_start_indices(
        self,
        current_action: np.ndarray,
        new_chunk: np.ndarray,
        robot_search_start: int,
        robot_search_end: int,
        gripper_search_start: int,
        gripper_search_end: int
    ) -> Tuple[int, int]:
        """
        Find optimal start indices separately for robot and gripper with independent search ranges.
        
        Supports:
        - 7D: [x, y, z, qx, qy, qz, qw] (position + quaternion, no gripper)
        - 8D: [x, y, z, qx, qy, qz, qw, gripper] (single arm with gripper)
        - 16D: [left_robot(7), left_gripper(1), right_robot(7), right_gripper(1)] (dual arm)
        
        Args:
            current_action: Current executing action
            new_chunk: New action chunk array
            robot_search_start: Start of robot search range
            robot_search_end: End of robot search range
            gripper_search_start: Start of gripper search range
            gripper_search_end: End of gripper search range
        
        Returns:
            Tuple of (robot_start_idx, gripper_start_idx)
        """
        action_dim = len(current_action)
        
        # Handle dual-arm case (16 dimensions)
        if action_dim == 16:
            # Left arm: robot [0:7], gripper [7]
            # Right arm: robot [8:15], gripper [15]
            left_robot_current = current_action[0:7]
            left_gripper_current = current_action[7:8]
            right_robot_current = current_action[8:15]
            right_gripper_current = current_action[15:16]
            
            # Search for robot optimal index (using position + rotation for both arms)
            robot_min_distance = float('inf')
            robot_best_idx = robot_search_start
            
            # Search for gripper optimal index (using end-effector for both arms)
            gripper_min_distance = float('inf')
            gripper_best_idx = gripper_search_start
            
            # Search for optimal robot index
            for idx in range(robot_search_start, robot_search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                
                # Left arm robot distance
                left_robot_candidate = candidate_action[0:7]
                left_pos_dist = calc_position_distance(left_robot_current[:3], left_robot_candidate[:3])
                left_rot_dist = calc_angle_diff(left_robot_current[3:7], left_robot_candidate[3:7], scalar_last=True)
                left_robot_dist = (self.pos_weight * left_pos_dist + self.rot_weight * left_rot_dist) / (self.pos_weight + self.rot_weight)
                
                # Right arm robot distance
                right_robot_candidate = candidate_action[8:15]
                right_pos_dist = calc_position_distance(right_robot_current[:3], right_robot_candidate[:3])
                right_rot_dist = calc_angle_diff(right_robot_current[3:7], right_robot_candidate[3:7], scalar_last=True)
                right_robot_dist = (self.pos_weight * right_pos_dist + self.rot_weight * right_rot_dist) / (self.pos_weight + self.rot_weight)
                
                # Average robot distance for both arms
                robot_dist = (left_robot_dist + right_robot_dist) / 2.0
                
                if robot_dist < robot_min_distance:
                    robot_min_distance = robot_dist
                    robot_best_idx = idx
            
            # Search for optimal gripper index (separate search range)
            for idx in range(gripper_search_start, gripper_search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                
                # Left arm gripper distance
                left_gripper_candidate = candidate_action[7:8]
                left_gripper_dist = calc_ee_diff(left_gripper_current, left_gripper_candidate)
                
                # Right arm gripper distance
                right_gripper_candidate = candidate_action[15:16]
                right_gripper_dist = calc_ee_diff(right_gripper_current, right_gripper_candidate)
                
                # Average gripper distance for both arms
                gripper_dist = (left_gripper_dist + right_gripper_dist) / 2.0
                
                if gripper_dist < gripper_min_distance:
                    gripper_min_distance = gripper_dist
                    gripper_best_idx = idx
            
            logger.debug(f"SmartStartChunkManager: Dual-arm - Robot best_idx={robot_best_idx} (dist={robot_min_distance:.6f}), "
                        f"Gripper best_idx={gripper_best_idx} (dist={gripper_min_distance:.6f})")
            
            return robot_best_idx, gripper_best_idx
        
        # For actions with gripper (8D+), separate robot and gripper
        elif action_dim >= 8:
            # Robot part: [x, y, z, qx, qy, qz, qw]
            robot_current = current_action[:7]
            # Gripper part: [gripper, ...]
            gripper_current = current_action[7:]
            
            # Search for robot optimal index (using position + rotation)
            robot_min_distance = float('inf')
            robot_best_idx = robot_search_start
            
            # Search for gripper optimal index (using end-effector only)
            gripper_min_distance = float('inf')
            gripper_best_idx = gripper_search_start
            
            # Search for optimal robot index
            for idx in range(robot_search_start, robot_search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                
                # Robot distance: position + rotation only
                robot_candidate = candidate_action[:7]
                pos_dist = calc_position_distance(robot_current[:3], robot_candidate[:3])
                rot_dist = calc_angle_diff(robot_current[3:7], robot_candidate[3:7], scalar_last=True)
                robot_dist = (self.pos_weight * pos_dist + self.rot_weight * rot_dist) / (self.pos_weight + self.rot_weight)
                
                if robot_dist < robot_min_distance:
                    robot_min_distance = robot_dist
                    robot_best_idx = idx
            
            # Search for optimal gripper index (separate search range)
            for idx in range(gripper_search_start, gripper_search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                
                # Gripper distance: end-effector only
                gripper_candidate = candidate_action[7:]
                gripper_dist = calc_ee_diff(gripper_current, gripper_candidate)
                
                if gripper_dist < gripper_min_distance:
                    gripper_min_distance = gripper_dist
                    gripper_best_idx = idx
            
            logger.debug(f"SmartStartChunkManager: Robot best_idx={robot_best_idx} (dist={robot_min_distance:.6f}), "
                        f"Gripper best_idx={gripper_best_idx} (dist={gripper_min_distance:.6f})")
            
            return robot_best_idx, gripper_best_idx
        
        else:
            # For 7D actions (no gripper), only optimize robot part
            robot_min_distance = float('inf')
            robot_best_idx = robot_search_start
            
            for idx in range(robot_search_start, robot_search_end + 1):
                if idx >= len(new_chunk):
                    break
                
                candidate_action = new_chunk[idx]
                pos_dist = calc_position_distance(current_action[:3], candidate_action[:3])
                rot_dist = calc_angle_diff(current_action[3:7], candidate_action[3:7], scalar_last=True)
                robot_dist = (self.pos_weight * pos_dist + self.rot_weight * rot_dist) / (self.pos_weight + self.rot_weight)
                
                if robot_dist < robot_min_distance:
                    robot_min_distance = robot_dist
                    robot_best_idx = idx
            
            logger.debug(f"SmartStartChunkManager: Robot best_idx={robot_best_idx} (dist={robot_min_distance:.6f}), "
                        f"no gripper data")
            
            # For 7D actions, return same index for both robot and gripper
            return robot_best_idx, robot_best_idx

