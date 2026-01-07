#!/usr/bin/env python3
"""
Ensemble Auto Action Manager - combines automatic optimal start index search with temporal ensembling.

This manager:
1. Automatically finds optimal start index for action chunk connection (from AutoActionManager)
2. Uses temporal ensembling to smooth transitions between chunks (from ACTTemporalEnsembler)
3. Applies exponential weighting to blend predictions from multiple chunks

The temporal ensembling reduces action jumps between chunk transitions by maintaining
a weighted average of overlapping predictions from different inference steps.

Author: Assistant
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils.logger_config import logger
from utils.transform import calc_action_distance
from .action_manager import BaseChunkManager


class EnsembleAutoActionManager(BaseChunkManager):
    """
    Auto action manager with temporal ensembling for smooth chunk transitions.
    
    This manager combines:
    - AutoActionManager: Automatically finds optimal start index by searching for 
      closest action match and skipping obsolete actions
    - ACTTemporalEnsembler: Uses exponential weighting to ensemble predictions from 
      multiple chunks, reducing discontinuities at chunk boundaries
    
    Temporal Ensemble Weights:
        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ 
        is the oldest action. They are normalized to sum to 1 by dividing by Σwᵢ.
        
        - Setting coeff to 0: uniformly weighs all actions
        - Setting coeff positive: gives more weight to older actions  
        - Setting coeff negative: gives more weight to newer actions
        
        Default coeff=0.01 (as in original ACT work) weighs older actions more highly.
    
    Args:
        execution_fps (float): Execution frequency in Hz for calculating elapsed steps
        search_tolerance (int): Tolerance for search range around estimated start index (default: 2)
        pos_weight (float): Weight for position distance in action matching (default: 0.4)
        rot_weight (float): Weight for rotation distance in action matching (default: 0.4)
        ee_weight (float): Weight for end-effector distance in action matching (default: 0.2)
        discount_factor (float): Discount factor for penalizing indices far from estimated_steps_elapsed (default: 1.0)
        temporal_ensemble_coeff (float): Coefficient for temporal ensemble exponential weights (default: 0.01)
        chunk_size (int): Expected chunk size for temporal ensembling (default: 100)
    """
    
    def __init__(
        self,
        execution_fps: float = 100.0,
        search_tolerance: int = 2,
        pos_weight: float = 0.4,
        rot_weight: float = 0.4,
        ee_weight: float = 0.2,
        discount_factor: float = 1.0,
        temporal_ensemble_coeff: float = 0.01,
        chunk_size: int = 100
    ):
        super().__init__()
        
        self.execution_fps = execution_fps
        self.search_tolerance = search_tolerance
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.ee_weight = ee_weight
        self.discount_factor = discount_factor
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.chunk_size = chunk_size
        
        # Calculate dt (time per step) in seconds
        self.dt_seconds = 1.0 / execution_fps
        
        # Temporal ensemble state - stores the ensembled actions remaining to execute
        self.ensembled_actions_dict = None  # Dict[device_name, np.ndarray] of shape (remaining_steps, action_dim)
        self.ensembled_actions_count = None  # np.ndarray of shape (remaining_steps, 1) counting ensemble depth
        
        # Precompute exponential weights and cumulative sum for efficiency
        self.ensemble_weights = np.exp(-temporal_ensemble_coeff * np.arange(chunk_size))
        self.ensemble_weights_cumsum = np.cumsum(self.ensemble_weights)
        
        logger.info(f"EnsembleAutoActionManager initialized: execution_fps={execution_fps}Hz, "
                   f"dt={self.dt_seconds}s, search_tolerance={search_tolerance}, "
                   f"discount_factor={discount_factor}, "
                   f"temporal_ensemble_coeff={temporal_ensemble_coeff}, "
                   f"chunk_size={chunk_size}, "
                   f"weights: pos={pos_weight}, rot={rot_weight}, ee={ee_weight}")
    
    def reset(self) -> None:
        """Reset the ensemble state. Should be called when environment is reset."""
        self.ensembled_actions_dict = None
        self.ensembled_actions_count = None
        self.current_action_chunk_dict = None
        self.current_step = 0
        self.last_read_time = 0
        logger.info("EnsembleAutoActionManager: Ensemble state reset")
    
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
        Process new action chunk by finding optimal start index and updating temporal ensemble.
        
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
                "EnsembleAutoActionManager: Received additional kwargs in put call: %s",
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
            logger.warning("EnsembleAutoActionManager: Empty action_chunk_dict, skipping")
            return
        
        first_chunk = action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)
        
        # If this is the first chunk, initialize ensemble with it
        if self.current_action_chunk_dict is None:
            logger.info(f"EnsembleAutoActionManager: First chunk received, length={chunk_length}, "
                       f"initializing ensemble from index 0")
            # Deep copy to avoid shared memory reference issues
            self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
            self.last_read_time = observation_timestamp
            self.current_chunk_timestamp = observation_timestamp if observation_timestamp is not None else 0
            self.current_step = 0
            
            # Initialize ensemble with the first chunk
            self.ensembled_actions_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
            self.ensembled_actions_count = np.ones((chunk_length, 1), dtype=np.int64)
            
            logger.info(f"EnsembleAutoActionManager: Ensemble initialized with shape "
                       f"{self.ensembled_actions_dict[device_names[0]].shape}")
            return
        
        # Calculate elapsed steps from observation timestamp to current time
        if observation_timestamp is None:
            logger.warning("EnsembleAutoActionManager: observation_timestamp is None, using current time")
            observation_timestamp = time.time_ns()
        
        current_time_ns = time.time_ns()
        elapsed_time_seconds = (current_time_ns - observation_timestamp) / 1e9
        estimated_steps_elapsed = int(elapsed_time_seconds / self.dt_seconds)
        
        logger.info(f"EnsembleAutoActionManager: obs_timestamp={observation_timestamp}, "
                   f"current_time={current_time_ns}, "
                   f"elapsed_time={elapsed_time_seconds:.4f}s, "
                   f"estimated_steps_elapsed={estimated_steps_elapsed}")
        
        # Get current executing action from previous chunk
        current_executing_action = self._get_current_executing_action()

        logger.info(f"action_chunk_dict[0:10]: {action_chunk_dict[device_names[0]][0:10]}")
        
        # Find optimal start index
        if current_executing_action is None:
            logger.warning("EnsembleAutoActionManager: Cannot get current executing action, "
                          "using estimated_steps_elapsed as start index")
            # Fallback to simple elapsed-step-based start index
            if estimated_steps_elapsed < chunk_length:
                optimal_start_index = max(0, estimated_steps_elapsed)
            else:
                logger.warning(f"EnsembleAutoActionManager: Estimated steps {estimated_steps_elapsed} >= "
                              f"chunk_length {chunk_length}, using start index 0")
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
        
        # Update temporal ensemble with new chunk starting from optimal_start_index
        self._update_temporal_ensemble(action_chunk_dict, optimal_start_index)
        
        # Store new chunk and set current_step to optimal_start_index (like AutoActionManager)
        # Deep copy to avoid shared memory reference issues
        self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
        self.last_read_time = observation_timestamp
        self.current_chunk_timestamp = observation_timestamp if observation_timestamp is not None else 0
        self.current_step = optimal_start_index  # Start from optimal index, like AutoActionManager
        
        logger.info(f"EnsembleAutoActionManager: New chunk stored with optimal start index={optimal_start_index}, "
                   f"chunk_length={chunk_length}, ensemble updated")
    
    def get(self, query_timestamp: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get next ensembled action dict.
        
        This consumes the first action from the ensemble and updates the ensemble state.
        Following ACT's temporal ensembling approach.
        
        Args:
            query_timestamp: Optional timestamp for the query (not used currently)
            
        Returns:
            Dictionary of ensembled actions for each device, or None if no actions available
        """
        try:
            if self.ensembled_actions_dict is None:
                return None
            
            # Get device names
            device_names = list(self.ensembled_actions_dict.keys())
            if not device_names:
                return None
            
            # Check if we have actions remaining in ensemble
            first_device = device_names[0]
            remaining_steps = len(self.ensembled_actions_dict[first_device])
            
            if remaining_steps == 0:
                logger.debug("EnsembleAutoActionManager: No more actions in ensemble")
                return None
            
            # Extract first action from ensemble for all devices (ACT style)
            action_dict = {}
            for device_name in device_names:
                action_dict[device_name] = self.ensembled_actions_dict[device_name][0].copy()
            
            # "Consume" the first action: shift ensemble forward (ACT style)
            for device_name in device_names:
                self.ensembled_actions_dict[device_name] = self.ensembled_actions_dict[device_name][1:]
            
            self.ensembled_actions_count = self.ensembled_actions_count[1:]
            
            # Increment current_step for tracking (like AutoActionManager)
            self.current_step += 1
            
            remaining_after = len(self.ensembled_actions_dict[first_device])
            logger.debug(f"EnsembleAutoActionManager: Dispatching ensembled action step {self.current_step}, "
                        f"remaining_steps={remaining_after}")
            
            return action_dict
                
        except Exception as e:
            logger.error(f"EnsembleAutoActionManager: Error getting next ensembled action: {e}")
            return None
    
    def _get_current_executing_action(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current executing action from ensemble.
        
        This returns the first action in the ensemble (what we're currently executing or about to execute).
        
        Returns:
            Dictionary of current actions for each device, or None if not available
        """
        if self.ensembled_actions_dict is None:
            return None
        
        # Get device names
        device_names = list(self.ensembled_actions_dict.keys())
        if not device_names:
            return None
        
        first_device = device_names[0]
        
        # Check if ensemble is empty
        if len(self.ensembled_actions_dict[first_device]) == 0:
            logger.debug("EnsembleAutoActionManager: Ensemble is empty, no current executing action")
            return None
        
        remaining_steps = len(self.ensembled_actions_dict[first_device])
        
        logger.info(f"ensemble_actions[0:10]: {self.ensembled_actions_dict[first_device][0:min(10, remaining_steps)]}")
        
        # Get the first action from ensemble (this is what we're currently executing)
        current_executing_action = {}
        for device_name in device_names:
            # Double check each device has actions
            if len(self.ensembled_actions_dict[device_name]) == 0:
                logger.warning(f"EnsembleAutoActionManager: Device {device_name} has empty ensemble")
                return None
            current_executing_action[device_name] = self.ensembled_actions_dict[device_name][0].copy()
        
        logger.debug(f"EnsembleAutoActionManager: Retrieved current executing action from ensemble, "
                    f"remaining_steps={remaining_steps}")
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
        
        This is the same logic as AutoActionManager._find_optimal_start_index.
        
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
                logger.warning(f"EnsembleAutoActionManager: Invalid search range [{search_start}, {search_end}], "
                              f"using index 0")
                return 0
            
            logger.info(f"EnsembleAutoActionManager: Searching optimal start index in range "
                       f"[{search_start}, {search_end}] "
                       f"(estimated_steps_elapsed={estimated_steps_elapsed}, chunk_length={chunk_length})")
            
            # For each device, find the closest action
            best_indices = []
            min_distances = []
            
            for device_name in device_names:
                if device_name not in current_executing_action:
                    logger.warning(f"EnsembleAutoActionManager: Device {device_name} not in "
                                  f"current_executing_action, skipping")
                    continue
                
                current_action = current_executing_action[device_name]
                new_chunk = action_chunk_dict[device_name]
                
                # Validate current_action
                if len(current_action) == 0:
                    logger.warning(f"EnsembleAutoActionManager: Device {device_name} has empty current_action, skipping")
                    continue
                
                # Validate new_chunk
                if len(new_chunk) == 0:
                    logger.warning(f"EnsembleAutoActionManager: Device {device_name} has empty new_chunk, skipping")
                    continue
                
                # Search for closest action
                min_distance = float('inf')
                best_index = search_start
                
                for idx in range(search_start, search_end + 1):
                    # Boundary check
                    if idx >= len(new_chunk):
                        logger.warning(f"EnsembleAutoActionManager: idx {idx} >= chunk length {len(new_chunk)}, breaking")
                        break
                    
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
                    offset = abs(idx - estimated_steps_elapsed)
                    discount = discount_factor ** offset
                    weighted_distance = distance / discount if discount > 0 else distance
                    
                    if weighted_distance < min_distance:
                        min_distance = weighted_distance
                        best_index = idx
                
                best_indices.append(best_index)
                min_distances.append(min_distance)
                logger.info(f"EnsembleAutoActionManager: Device {device_name} - "
                           f"best_index={best_index}, min_distance={min_distance:.6f}, "
                           f"search_range=[{search_start}, {search_end}]")
            
            # Average the best indices from all devices
            if best_indices:
                optimal_index = int(np.round(np.mean(best_indices)))
                avg_min_distance = np.mean(min_distances) if min_distances else 0.0
                logger.info(f"EnsembleAutoActionManager: optimal_index={optimal_index} "
                           f"(from indices {best_indices}), "
                           f"avg_min_distance={avg_min_distance:.6f}, "
                           f"search_range=[{search_start}, {search_end}], "
                           f"skipping {optimal_index} obsolete actions")
                return optimal_index
            else:
                logger.warning("EnsembleAutoActionManager: No valid devices found for optimal index "
                              "calculation, using search_start")
                return search_start
        
        except Exception as e:
            logger.error(f"EnsembleAutoActionManager: Error finding optimal start index: {e}, "
                        f"falling back to 0")
            return 0
    
    def _update_temporal_ensemble(
        self,
        new_action_chunk_dict: Dict[str, np.ndarray],
        start_index: int
    ) -> None:
        """
        Update temporal ensemble with new action chunk using ACT's algorithm.
        
        This implements the temporal ensembling algorithm from ACT (Algorithm 2 in the paper):
        1. For overlapping actions in ensemble, compute online weighted average with new predictions
        2. Append new actions that extend beyond current ensemble
        
        The key insight from ACT:
        - When a new chunk arrives, its actions from index `start_index` onwards should be 
          ensembled with the existing ensemble
        - The first action in new chunk (at start_index) should be ensembled with the first 
          action in current ensemble
        - This creates smooth transitions at chunk boundaries
        
        The weights follow exponential decay: wᵢ = exp(-temporal_ensemble_coeff * i)
        
        Args:
            new_action_chunk_dict: New action chunk from policy
            start_index: The index in new chunk to start from (obsolete actions before this are skipped)
        """
        device_names = list(new_action_chunk_dict.keys())
        if not device_names:
            return
        
        # Extract the valid portion of new chunk (from start_index onwards)
        new_actions_dict = {}
        for device_name in device_names:
            new_actions_dict[device_name] = new_action_chunk_dict[device_name][start_index:].copy()
        
        first_device = device_names[0]
        new_chunk_length = len(new_actions_dict[first_device])
        
        logger.info(f"EnsembleAutoActionManager: Updating ensemble with new chunk, "
                   f"new_chunk_length={new_chunk_length}, start_index={start_index}")
        
        # If ensemble is empty or somehow corrupted, initialize it
        if self.ensembled_actions_dict is None or len(self.ensembled_actions_dict[first_device]) == 0:
            logger.warning("EnsembleAutoActionManager: Ensemble is empty, initializing with new chunk")
            self.ensembled_actions_dict = {k: v.copy() for k, v in new_actions_dict.items()}
            self.ensembled_actions_count = np.ones((new_chunk_length, 1), dtype=np.int64)
            return
        
        current_ensemble_length = len(self.ensembled_actions_dict[first_device])
        
        # ACT algorithm: The new chunk's actions should be ensembled with current ensemble
        # overlap_length is how many actions we can ensemble (limited by shorter sequence)
        overlap_length = min(current_ensemble_length, new_chunk_length)
        
        logger.info(f"EnsembleAutoActionManager: current_ensemble_length={current_ensemble_length}, "
                   f"new_chunk_length={new_chunk_length}, overlap_length={overlap_length}")
        
        # Update overlapping portion with weighted average (ACT's online ensembling)
        if overlap_length > 0:
            for i in range(overlap_length):
                # Get current ensemble count for this position
                count = self.ensembled_actions_count[i, 0]
                
                # Clamp count to not exceed chunk_size (for weight indexing)
                count_clamped = min(count, self.chunk_size - 1)
                
                # ACT's online weighted average formula:
                # ensembled_avg = (ensembled_avg * weight_sum_old + new_action * weight_new) / weight_sum_new
                weight_sum_old = self.ensemble_weights_cumsum[count_clamped]
                weight_new = self.ensemble_weights[min(count, self.chunk_size - 1)]
                weight_sum_new = self.ensemble_weights_cumsum[min(count_clamped + 1, self.chunk_size - 1)]
                
                for device_name in device_names:
                    old_action = self.ensembled_actions_dict[device_name][i]
                    new_action = new_actions_dict[device_name][i]
                    
                    # Weighted average update (ACT formula)
                    self.ensembled_actions_dict[device_name][i] = (
                        old_action * weight_sum_old + new_action * weight_new
                    ) / weight_sum_new
            
            # Update count for overlapping portion (increment but clamp to chunk_size)
            self.ensembled_actions_count[:overlap_length] = np.minimum(
                self.ensembled_actions_count[:overlap_length] + 1,
                self.chunk_size
            )
        
        # If new chunk extends beyond current ensemble, append the extra actions
        if new_chunk_length > overlap_length:
            extra_length = new_chunk_length - overlap_length
            logger.info(f"EnsembleAutoActionManager: Appending {extra_length} new actions to ensemble")
            
            for device_name in device_names:
                extra_actions = new_actions_dict[device_name][overlap_length:].copy()
                self.ensembled_actions_dict[device_name] = np.concatenate([
                    self.ensembled_actions_dict[device_name],
                    extra_actions
                ], axis=0)
            
            # Append count of 1 for new actions (first time being ensembled)
            extra_count = np.ones((extra_length, 1), dtype=np.int64)
            self.ensembled_actions_count = np.concatenate([
                self.ensembled_actions_count,
                extra_count
            ], axis=0)
        
        final_ensemble_length = len(self.ensembled_actions_dict[first_device])
        logger.info(f"EnsembleAutoActionManager: Ensemble updated, final_length={final_ensemble_length}")
