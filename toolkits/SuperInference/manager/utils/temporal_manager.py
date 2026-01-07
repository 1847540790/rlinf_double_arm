from . import BaseChunkManager
import time
import numpy as np
from utils.logger_config import logger
from typing import Dict, List, Any, Optional, Tuple
from utils.logger_config import logger

class TemporalChunkManager(BaseChunkManager):
    """
    The temporal chunk manager expotienally averages the actions of
    the older chunks and the latest chunk for better smoothness. 

    Args:
        averaging_factor (float): the final action chunk is computed by: averaging_factor * chunk_past + (1 - averaging_factor) * chunk_current
    """
    def __init__(self, averaging_factor: float=0.1):
        super().__init__()
        self.averaging_factor = averaging_factor

    def put(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: Optional[int]=None) -> None:
        """
        Put the action chunk dict to local chunk cache and compute 
        the expotienal averaging of the older one and the new one.
        """
        if self.current_action_chunk_dict is None:
            super().put(action_chunk_dict, observation_timestamp)
            return 
        previous_step = self.current_step
        device_names = list(self.current_action_chunk_dict.keys())
        first_chunk = self.current_action_chunk_dict[device_names[0]]
        previous_length =  len(first_chunk)
        remaining_length = previous_length - previous_step
        
        if remaining_length > 0:
            # Accumulate remaining steps of the previous chunk from all devices to the current action chunk
            for device_name, chunk in self.current_action_chunk_dict.items():
                if device_name in action_chunk_dict:
                    action_chunk_dict[device_name][:remaining_length] = self.averaging_factor * chunk[previous_step:] + (1-self.averaging_factor) * action_chunk_dict[device_name][:remaining_length]
        super().put(action_chunk_dict, observation_timestamp)