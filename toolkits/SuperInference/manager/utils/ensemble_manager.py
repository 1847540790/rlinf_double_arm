#!/usr/bin/env python3
"""
Ensemble Chunk Manager - combines overlapping action chunks using temporal ensembles.

This manager leverages the EnsembleBuffer implementation from the UMI diffusion policy
to fuse multiple predicted trajectories into a single smooth command stream.

Author: 
"""

from typing import Dict, Optional, Any
import numpy as np

from .action_manager import BaseChunkManager
from utils.logger_config import logger
from third_party.umi_base.diffusion_policy.common.ensemble import EnsembleBuffer


class EnsembleChunkManager(BaseChunkManager):
    """
    Maintains a temporal ensemble for each device and outputs ensembled actions step-by-step.

    Args:
        ensemble_mode (str): Mode passed to EnsembleBuffer ("new", "old", "avg", "act", "hato").
        k (float): Decay factor used by "act" mode.
        tau (float): Discount factor used by "hato" mode.
    """

    def __init__(
        self,
        ensemble_mode: str = "new",
        k: float = 0.01,
        tau: float = 0.9,
    ) -> None:
        super().__init__()
        self.ensemble_mode = ensemble_mode
        self.k = k
        self.tau = tau
        self.device_buffers: Dict[str, EnsembleBuffer] = {}

    def _get_buffer(self, device_name: str) -> EnsembleBuffer:
        """Lazy-initialize an EnsembleBuffer for the specified device."""
        if device_name not in self.device_buffers:
            self.device_buffers[device_name] = EnsembleBuffer(
                ensemble_mode=self.ensemble_mode,
                k=self.k,
                tau=self.tau,
            )
        return self.device_buffers[device_name]

    def _buffer_has_ready_action(self, buffer: EnsembleBuffer) -> bool:
        """Check whether the buffer has at least one action ready to be consumed."""
        delta = buffer.timestep - buffer.actions_start_timestep
        if delta < 0:
            logger.warning("EnsembleBuffer timestep misalignment detected.")
            return False
        if delta >= len(buffer.actions):
            return False

        # Actions are stored per timestep; account for stale entries that would be pruned.
        actions_idx = delta
        actions_available = list(buffer.actions)
        timesteps_available = list(buffer.actions_timestep)

        # Skip any stale entries that would be removed by get_action.
        if actions_idx > 0:
            actions_available = actions_available[actions_idx:]
            timesteps_available = timesteps_available[actions_idx:]

        if not actions_available:
            return False

        return len(actions_available[0]) > 0

    def _consume_buffer_action(self, buffer: EnsembleBuffer) -> Optional[np.ndarray]:
        """
        Consume the next action from the buffer using the same logic as EnsembleBuffer.get_action.
        """
        # Direct translation of EnsembleBuffer.get_action with explicit control over data flow.
        if buffer.timestep - buffer.actions_start_timestep >= len(buffer.actions):
            return None

        while buffer.actions_start_timestep < buffer.timestep:
            if not buffer.actions:
                return None
            buffer.actions.popleft()
            buffer.actions_timestep.popleft()
            buffer.actions_start_timestep += 1

        if not buffer.actions or not buffer.actions_timestep:
            return None

        actions = buffer.actions[0]
        actions_timestep = buffer.actions_timestep[0]
        if not actions:
            return None

        sorted_actions = sorted(zip(actions_timestep, actions))
        all_actions = np.array([x for _, x in sorted_actions])

        if self.ensemble_mode == "new":
            action = all_actions[-1]
        elif self.ensemble_mode == "old":
            action = all_actions[0]
        else:
            raise AttributeError(f"Ensemble mode {self.ensemble_mode} not supported.")

        buffer.timestep += 1
        return action

    def is_empty(self) -> bool:
        """Override to check ensemble buffers."""
        if not self.device_buffers:
            return True
        return not any(self._buffer_has_ready_action(buffer) for buffer in self.device_buffers.values())

    def put(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: Optional[int] = None) -> None:
        """Add a new action chunk to the ensemble buffers."""
        self.last_read_time = observation_timestamp
        for device_name, action_chunk in action_chunk_dict.items():
            buffer = self._get_buffer(device_name)
            try:
                start_timestep = buffer.timestep
                buffer.add_action(action_chunk, start_timestep)
                logger.debug(
                    "EnsembleChunkManager: Added chunk for %s starting at timestep %d (len=%d)",
                    device_name,
                    start_timestep,
                    len(action_chunk),
                )
            except Exception as exc:
                logger.error("EnsembleChunkManager: Failed to add action chunk for %s: %s", device_name, exc)

    def get(self, query_timestamp: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """Get the next ensembled action for all devices."""
        if not self.device_buffers:
            return None

        # Ensure every device has an action ready before consuming any of them.
        ready = all(self._buffer_has_ready_action(buffer) for buffer in self.device_buffers.values())
        if not ready:
            return None

        action_dict: Dict[str, np.ndarray] = {}
        for device_name, buffer in self.device_buffers.items():
            action = self._consume_buffer_action(buffer)
            if action is None:
                logger.debug("EnsembleChunkManager: Buffer for %s had no action despite readiness check.", device_name)
                return None
            action_dict[device_name] = action

        return action_dict

    def clear(self) -> None:
        """Reset all ensemble buffers."""
        for buffer in self.device_buffers.values():
            buffer.clear()
        self.device_buffers.clear()
        self.current_action_chunk_dict = None
        self.current_step = 0
        self.last_read_time = 0
