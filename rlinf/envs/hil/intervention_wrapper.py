# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human-in-the-Loop intervention wrapper for real robot environments.

This wrapper intercepts actions before they are sent to the environment,
allowing a human operator to take over control when needed. It tracks
intervention statistics and stores intervention data for learning.

Usage:
    env = YourRobotEnv()
    hil_env = HILInterventionWrapper(env, input_device="keyboard")
    
    obs = hil_env.reset()
    for step in range(max_steps):
        action = policy.get_action(obs)
        obs, reward, done, truncated, info = hil_env.step(action)
        
        if info.get("is_intervention"):
            # Human took over - use info["intervene_action"] for learning
            pass
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from rlinf.envs.hil.input_devices import (
    BaseInputDevice,
    InputDeviceConfig,
    get_input_device,
)


@dataclass
class HILStats:
    """Statistics for Human-in-the-Loop training."""
    total_steps: int = 0
    intervention_steps: int = 0
    intervention_count: int = 0  # Number of intervention episodes
    episode_count: int = 0
    
    # Per-episode tracking
    current_episode_interventions: int = 0
    current_episode_intervention_steps: int = 0
    is_currently_intervening: bool = False
    
    def step(self, is_intervention: bool):
        """Update stats for a single step."""
        self.total_steps += 1
        
        if is_intervention:
            self.intervention_steps += 1
            self.current_episode_intervention_steps += 1
            
            if not self.is_currently_intervening:
                # New intervention episode started
                self.intervention_count += 1
                self.current_episode_interventions += 1
                self.is_currently_intervening = True
        else:
            self.is_currently_intervening = False
            
    def end_episode(self):
        """Called at end of episode to reset per-episode stats."""
        self.episode_count += 1
        episode_stats = {
            "episode_interventions": self.current_episode_interventions,
            "episode_intervention_steps": self.current_episode_intervention_steps,
        }
        self.current_episode_interventions = 0
        self.current_episode_intervention_steps = 0
        self.is_currently_intervening = False
        return episode_stats
    
    @property
    def intervention_rate(self) -> float:
        """Fraction of steps with human intervention."""
        if self.total_steps == 0:
            return 0.0
        return self.intervention_steps / self.total_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "hil/total_steps": self.total_steps,
            "hil/intervention_steps": self.intervention_steps,
            "hil/intervention_count": self.intervention_count,
            "hil/intervention_rate": self.intervention_rate,
            "hil/episode_count": self.episode_count,
        }


class HILInterventionWrapper(gym.Wrapper):
    """Wrapper that enables human intervention during robot control.
    
    This wrapper:
        1. Checks for human input at each step
        2. If human is intervening, replaces policy action with human action
        3. Records intervention data in the info dict
        4. Tracks intervention statistics
        
    The key output is `info["intervene_action"]` which contains the human
    action when intervention occurs. This should be used for:
        - Storing in replay buffer (use human action, not policy action)
        - Computing intervention-based rewards
        - Demo buffer for RLPD-style training
    
    Args:
        env: The environment to wrap
        input_device: Name of input device ("mock", "keyboard", "vivetracker")
                     or an instance of BaseInputDevice
        input_device_config: Configuration for the input device
        blend_actions: If True, blend human and policy actions instead of replacing
        blend_alpha: Blending factor when blend_actions=True (1.0 = full human)
    """
    
    def __init__(
        self,
        env: gym.Env,
        input_device: Union[str, BaseInputDevice] = "mock",
        input_device_config: Optional[InputDeviceConfig] = None,
        blend_actions: bool = False,
        blend_alpha: float = 1.0,
        **device_kwargs,
    ):
        super().__init__(env)
        
        # Set up input device
        if isinstance(input_device, str):
            config = input_device_config or InputDeviceConfig(
                action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 7
            )
            self.input_device = get_input_device(input_device, config=config, **device_kwargs)
        else:
            self.input_device = input_device
            
        self.blend_actions = blend_actions
        self.blend_alpha = blend_alpha
        
        # Statistics
        self.stats = HILStats()
        
        # Start input device
        self.input_device.start()
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action, potentially with human intervention.
        
        Args:
            action: Policy action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        The info dict will contain:
            - "is_intervention": bool, True if human intervened
            - "intervene_action": np.ndarray, the human action (if intervening)
            - "policy_action": np.ndarray, the original policy action
            - "executed_action": np.ndarray, the action that was actually executed
        """
        # Get human input
        human_action, is_intervening = self.input_device.get_action()
        
        # Determine action to execute
        if is_intervening:
            if self.blend_actions:
                # Blend human and policy actions
                executed_action = (
                    self.blend_alpha * human_action + 
                    (1 - self.blend_alpha) * action
                )
            else:
                # Replace with human action
                executed_action = human_action
        else:
            executed_action = action
            
        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        
        # Update stats
        self.stats.step(is_intervening)
        
        # Add intervention info
        info["is_intervention"] = is_intervening
        info["policy_action"] = action.copy()
        info["executed_action"] = executed_action.copy()
        
        if is_intervening:
            info["intervene_action"] = human_action.copy()
            
        # Check for episode end
        if terminated or truncated:
            episode_stats = self.stats.end_episode()
            info["episode_intervention_count"] = episode_stats["episode_interventions"]
            info["episode_intervention_steps"] = episode_stats["episode_intervention_steps"]
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and input device state."""
        self.input_device.reset()
        obs, info = self.env.reset(**kwargs)
        info["is_intervention"] = False
        return obs, info
    
    def close(self):
        """Clean up resources."""
        self.input_device.stop()
        super().close()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get intervention statistics."""
        return self.stats.to_dict()
    
    def set_intervention_mode(self, enabled: bool):
        """Enable or disable intervention (for testing)."""
        if hasattr(self.input_device, 'set_intervention'):
            if enabled:
                # Force intervention with zero action
                self.input_device.set_intervention(
                    np.zeros(self.input_device.config.action_dim)
                )
            else:
                self.input_device.clear_intervention()


class ActionRecordingWrapper(gym.Wrapper):
    """Wrapper that records all actions for later playback or analysis.
    
    Useful for:
        - Recording demonstrations
        - Debugging action sequences
        - Analyzing intervention patterns
    """
    
    def __init__(self, env: gym.Env, max_steps: int = 10000):
        super().__init__(env)
        self.max_steps = max_steps
        self.recorded_data = []
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record step data
        record = {
            "action": action.copy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        
        # Include intervention info if available
        if "is_intervention" in info:
            record["is_intervention"] = info["is_intervention"]
            if info["is_intervention"]:
                record["intervene_action"] = info.get("intervene_action", action).copy()
                
        self.recorded_data.append(record)
        
        # Truncate if too long
        if len(self.recorded_data) > self.max_steps:
            self.recorded_data = self.recorded_data[-self.max_steps:]
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        # Don't clear recorded data on reset - only clear manually
        return self.env.reset(**kwargs)
    
    def get_recorded_data(self) -> list:
        """Get all recorded data."""
        return self.recorded_data
    
    def clear_recorded_data(self):
        """Clear recorded data."""
        self.recorded_data = []
    
    def save_recorded_data(self, filepath: str):
        """Save recorded data to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.recorded_data, f)
            
    def load_recorded_data(self, filepath: str):
        """Load recorded data from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.recorded_data = pickle.load(f)

