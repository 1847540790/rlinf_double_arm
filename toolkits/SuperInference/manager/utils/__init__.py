#!/usr/bin/env python3
"""
Action Manager Package - Contains different types of action manager

This package provides various manager implementations for deciding 
the action for execution from time-varying action chunks

Author: Zheng Wang
"""

from typing import Optional
from .action_manager import BaseChunkManager
from .temporal_manager import TemporalChunkManager
from .nodelay_manager import NoDelayChunkManager
from .smart_start_manager import SmartStartChunkManager
from .auto_action_manager import AutoActionManager
from .ensemble_manager_v2 import AutoActionManagerV2
from .ensemble_auto_action_manager import EnsembleAutoActionManager
from .interpolation_manager import InterpolationChunkManager

__all__ = [
    'BaseChunkManager',
    'TemporalChunkManager',
    'NoDelayChunkManager',
    'SmartStartChunkManager',
    'AutoActionManager',
    'EnsembleAutoActionManager',
    'AutoActionManagerV2',
    'InterpolationChunkManager',
]

CHUNK_MANAGERS = {
    'base_manager': BaseChunkManager, 
    'temporal_manager': TemporalChunkManager,
    'nodelay_manager': NoDelayChunkManager,
    'smart_start_manager': SmartStartChunkManager,
    'auto_action_manager': AutoActionManager,
    'ensemble_auto_action_manager': EnsembleAutoActionManager,
    'auto_manager_v2': AutoActionManagerV2,
    'interpolation_manager': InterpolationChunkManager,
}

def load_chunk_manager(manager_config: dict = {}, execution_fps: Optional[float] = None) -> BaseChunkManager:
    """
    Load chunk manager from configuration dict.
    
    Args:
        manager_config: Configuration dictionary for the chunk manager
        execution_fps: Execution FPS from action_executor (used for SmartStartChunkManager and AutoActionManager)
    """
    manager_name = manager_config.get('type', 'base_manager')
    assert manager_name in CHUNK_MANAGERS, f"type `{manager_name}` not in {CHUNK_MANAGERS}"
    MANAGER_CLASS = CHUNK_MANAGERS.get(manager_name)
    manager_para = {k:v for k,v in manager_config.items() if k!='type'}
    
    # If execution_fps is provided and manager supports it, add it to parameters
    # This allows dt to be calculated from execution_fps instead of being configured directly
    if execution_fps is not None:
        if manager_name in ['smart_start_manager', 'auto_action_manager', 'ensemble_auto_action_manager', 'auto_manager_v2']:
            manager_para['execution_fps'] = execution_fps
    
    chunk_manager = MANAGER_CLASS(**manager_para)
    return chunk_manager
