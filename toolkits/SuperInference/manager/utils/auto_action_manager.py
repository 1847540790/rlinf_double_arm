#!/usr/bin/env python3
"""
Auto Action Manager - automatically finds optimal start index for action chunk connection.

This manager:
1. Calculates elapsed steps based on observation timestamp and current time
2. Retrieves current executing action from previous chunk
3. Searches for optimal start index by finding closest action in new chunk
4. Uses the optimal start index as the initial current_step for new chunk

The logic is migrated from PolicyConnector's smart start detection method.

Author: Assistant, Based on PolicyConnector implementation
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils.logger_config import logger
from utils.transform import calc_action_distance
from .action_manager import BaseChunkManager


class AutoActionManager(BaseChunkManager):
    """
    Auto action manager that automatically finds optimal start index for seamless chunk connection.
    
    This manager eliminates obsolete actions from each new action chunk by:
    1. Calculating elapsed time from observation timestamp to current time
    2. Estimating elapsed steps using execution FPS
    3. Searching for closest action to current executing action within tolerance range
    4. Setting optimal start index as initial current_step
    
    Args:
        execution_fps (float): Execution frequency in Hz for calculating elapsed steps
        search_tolerance (int): Tolerance for search range around estimated start index (default: 2)
        pos_weight (float): Weight for position distance in action matching (default: 0.4)
        rot_weight (float): Weight for rotation distance in action matching (default: 0.4)
        ee_weight (float): Weight for end-effector distance in action matching (default: 0.2)
        discount_factor (float): Discount factor for penalizing indices far from estimated_steps_elapsed.
                                 At estimated_steps_elapsed, discount=1.0; at ±1, discount=discount_factor;
                                 at ±2, discount=discount_factor^2, etc. (default: 1.0, no discount)
    """
    
    def __init__(
        self,
        execution_fps: float = 100.0,
        search_tolerance: int = 2,
        pos_weight: float = 0.4,
        rot_weight: float = 0.4,
        ee_weight: float = 0.2,
        discount_factor: float = 1.0
    ):
        super().__init__()
        
        self.execution_fps = execution_fps
        self.search_tolerance = search_tolerance
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.ee_weight = ee_weight
        self.discount_factor = discount_factor
        
        # Calculate dt (time per step) in seconds
        self.dt_seconds = 1.0 / execution_fps
        
        logger.info(f"AutoActionManager initialized: execution_fps={execution_fps}Hz, "
                   f"dt={self.dt_seconds}s, search_tolerance={search_tolerance}, "
                   f"discount_factor={discount_factor}, "
                   f"weights: pos={pos_weight}, rot={rot_weight}, ee={ee_weight}")
    
    def put(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: Optional[int] = None,
        last_action_dict: Optional[Dict[str, np.ndarray]] = None,
        steps_elapsed: Optional[int] = None,
        search_tolerance: Optional[int] = None,
        pos_weight: Optional[float] = None,
        rot_weight: Optional[float] = None,
        ee_weight: Optional[float] = None,
        discount_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Process new action chunk by finding optimal start index automatically.
        
        Args:
            action_chunk_dict: New action chunk dictionary from policy
            observation_timestamp: Timestamp when observation was taken (nanoseconds)
            last_action_dict: Not used in this manager (kept for compatibility)
            steps_elapsed: Not used in this manager (kept for compatibility)
            search_tolerance: Optional override for search tolerance
            pos_weight: Optional override for position distance weight
            rot_weight: Optional override for rotation distance weight
            ee_weight: Optional override for end-effector distance weight
            discount_factor: Optional override for discount factor
            kwargs: Additional keyword arguments kept for forward compatibility
        """
        if kwargs:
            logger.debug(
                "AutoActionManager: Received additional kwargs in put call: %s",
                list(kwargs.keys()),
            )
        
        # Use override values if provided, otherwise use configured defaults
        search_tolerance_value = search_tolerance if search_tolerance is not None else self.search_tolerance
        pos_weight_value = pos_weight if pos_weight is not None else self.pos_weight
        rot_weight_value = rot_weight if rot_weight is not None else self.rot_weight
        ee_weight_value = ee_weight if ee_weight is not None else self.ee_weight
        discount_factor_value = discount_factor if discount_factor is not None else self.discount_factor

        # Get device names and chunk length
        device_names = list(action_chunk_dict.keys())
        if not device_names:
            logger.warning("AutoActionManager: Empty action_chunk_dict, skipping")
            return
        
        first_chunk = action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)
        
        # If this is the first chunk, just store it and start from beginning
        if self.current_action_chunk_dict is None:
            logger.info(f"AutoActionManager: First chunk received, length={chunk_length}, starting from index 0")
            # Deep copy to avoid shared memory reference issues
            self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
            self.last_read_time = observation_timestamp
            self.current_chunk_timestamp = observation_timestamp if observation_timestamp is not None else 0
            self.current_step = 0
            return
        
        # Calculate elapsed steps from observation timestamp to current time
        if observation_timestamp is None:
            logger.warning("AutoActionManager: observation_timestamp is None, using current time")
            observation_timestamp = time.time_ns()
        
        current_time_ns = time.time_ns()
        elapsed_time_seconds = (current_time_ns - observation_timestamp) / 1e9
        estimated_steps_elapsed = int(elapsed_time_seconds / self.dt_seconds)
        
        logger.info(f"AutoActionManager: obs_timestamp={observation_timestamp}, "
                   f"current_time={current_time_ns}, "
                   f"elapsed_time={elapsed_time_seconds:.4f}s, "
                   f"estimated_steps_elapsed={estimated_steps_elapsed}")
        
        # Get current executing action from previous chunk
        current_executing_action = self._get_current_executing_action()

        device_names = list(action_chunk_dict.keys())

        logger.info(f"action_chunk_dict[0:10]{action_chunk_dict[device_names[0]][0:10]}")
        
        if current_executing_action is None:
            logger.warning("AutoActionManager: Cannot get current executing action, using estimated_steps_elapsed as start index")
            # Fallback to simple elapsed-step-based start index
            if estimated_steps_elapsed < chunk_length:
                optimal_start_index = max(0, estimated_steps_elapsed)
            else:
                logger.warning(f"AutoActionManager: Estimated steps {estimated_steps_elapsed} >= chunk_length {chunk_length}, "
                              f"using start index 0")
                optimal_start_index = 0
        else:
            # Search for optimal start index
            optimal_start_index = self._find_optimal_start_index(
                action_chunk_dict=action_chunk_dict,
                current_executing_action=current_executing_action,
                estimated_steps_elapsed=estimated_steps_elapsed,
                search_tolerance=search_tolerance_value,
                pos_weight=pos_weight_value,
                rot_weight=rot_weight_value,
                ee_weight=ee_weight_value,
                discount_factor=discount_factor_value
            )
        
        # Store new chunk and set current_step to optimal_start_index
        # Deep copy to avoid shared memory reference issues
        self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
        self.last_read_time = observation_timestamp
        self.current_chunk_timestamp = observation_timestamp if observation_timestamp is not None else 0
        self.current_step = optimal_start_index
        
        logger.info(f"AutoActionManager: New chunk stored with optimal start index={optimal_start_index}, "
                   f"chunk_length={chunk_length}, devices={device_names}")
    
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
        logger.info(f"shm_now_action_chunk[0:10]{first_chunk[0:10]}")
        chunk_length = len(first_chunk)
        
        # If current_step is beyond chunk length, use the last action
        step_index = min(self.current_step, chunk_length - 1)
        
        current_executing_action = {}
        for device_name, chunk in self.current_action_chunk_dict.items():
            if step_index < len(chunk):
                current_executing_action[device_name] = chunk[step_index]
            else:
                logger.warning(f"AutoActionManager: step_index {step_index} out of range for {device_name}")
                return None
        
        logger.debug(f"AutoActionManager: Retrieved current executing action at step {step_index}/{chunk_length}")
        return current_executing_action
    
    def _find_optimal_start_index(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        current_executing_action: Dict[str, np.ndarray],
        estimated_steps_elapsed: int,
        search_tolerance: int,
        pos_weight: float,
        rot_weight: float,
        ee_weight: float,
        discount_factor: float
    ) -> int:
        """
        Find optimal start index by searching for closest action to current executing action.
        
        Applies a discount factor to penalize indices far from estimated_steps_elapsed:
        - At estimated_steps_elapsed: discount = 1.0
        - At estimated_steps_elapsed ± n: discount = discount_factor^n
        - Weighted distance = raw_distance / discount (larger penalty for distant indices)
        
        Args:
            action_chunk_dict: New action chunk dict
            current_executing_action: Current action being executed
            estimated_steps_elapsed: Estimated steps elapsed during observation + prediction
            search_tolerance: Tolerance for search range
            pos_weight: Weight for position distance
            rot_weight: Weight for rotation distance
            ee_weight: Weight for end-effector distance
            discount_factor: Discount factor for penalizing distant indices (1.0 = no discount)
        
        Returns:
            Optimal start index (0-based)
        """
        try:
            # Get chunk length from first device
            device_names = list(action_chunk_dict.keys())
            if not device_names:
                return 0
            
            first_device = device_names[0]
            chunk_length = len(action_chunk_dict[first_device])
            
            # Calculate search range [estimated_steps_elapsed - tolerance, estimated_steps_elapsed + tolerance]
            search_start = max(0, estimated_steps_elapsed - search_tolerance)
            search_end = min(chunk_length - 1, estimated_steps_elapsed + search_tolerance)
            
            if search_start > search_end:
                logger.warning(f"AutoActionManager: Invalid search range [{search_start}, {search_end}], using index 0")
                return 0
            
            logger.info(f"AutoActionManager: Searching optimal start index in range [{search_start}, {search_end}] "
                       f"(estimated_steps_elapsed={estimated_steps_elapsed}, chunk_length={chunk_length})")
            
            # For each device, find the closest action
            best_indices = []
            min_distances = []
            
            for device_name in device_names:
                if device_name not in current_executing_action:
                    logger.warning(f"AutoActionManager: Device {device_name} not in current_executing_action, skipping")
                    continue
                
                current_action = current_executing_action[device_name]
                new_chunk = action_chunk_dict[device_name]
                
                # Search for closest action
                min_distance = float('inf')
                best_index = search_start
                
                for idx in range(search_start, search_end + 1):
                    candidate_action = new_chunk[idx]
                    
                    # For dual-arm data (16 dimensions)
                    if len(candidate_action) == 16:
                        # Left arm: [0:8], Right arm: [8:16]
                        left_dist = calc_action_distance(
                            current_action[:8], candidate_action[:8],
                            pos_weight=pos_weight,
                            rot_weight=rot_weight,
                            ee_weight=ee_weight
                        )
                        right_dist = calc_action_distance(
                            current_action[8:16], candidate_action[8:16],
                            pos_weight=pos_weight,
                            rot_weight=rot_weight,
                            ee_weight=ee_weight
                        )
                        distance = (left_dist + right_dist) / 2.0
                    else:
                        # Single arm or other format
                        distance = calc_action_distance(
                            current_action, candidate_action,
                            pos_weight=pos_weight,
                            rot_weight=rot_weight,
                            ee_weight=ee_weight
                        )
                    
                    # Apply discount factor: penalize indices far from estimated_steps_elapsed
                    # Distance from estimated index
                    offset = abs(idx - estimated_steps_elapsed)
                    # Discount coefficient: discount_factor^offset
                    # At offset=0: discount=1.0; at offset=1: discount=discount_factor; etc.
                    discount = discount_factor ** offset
                    # Weighted distance: divide by discount to penalize distant indices
                    # (smaller discount -> larger weighted distance -> less likely to be selected)
                    weighted_distance = distance / discount if discount > 0 else distance
                    
                    if weighted_distance < min_distance:
                        min_distance = weighted_distance
                        best_index = idx
                
                best_indices.append(best_index)
                min_distances.append(min_distance)
                logger.info(f"AutoActionManager: Device {device_name} - "
                           f"best_index={best_index}, min_distance={min_distance:.6f}, "
                           f"search_range=[{search_start}, {search_end}]")
            
            # Average the best indices from all devices
            if best_indices:
                optimal_index = int(np.round(np.mean(best_indices)))
                avg_min_distance = np.mean(min_distances) if min_distances else 0.0
                logger.info(f"AutoActionManager: optimal_index={optimal_index} (from indices {best_indices}), "
                           f"avg_min_distance={avg_min_distance:.6f}, "
                           f"search_range=[{search_start}, {search_end}], "
                           f"skipping {optimal_index} obsolete actions")
                return optimal_index
            else:
                logger.warning("AutoActionManager: No valid devices found for optimal index calculation, using search_start")
                return search_start
        
        except Exception as e:
            logger.error(f"AutoActionManager: Error finding optimal start index: {e}, falling back to 0")
            return 0



