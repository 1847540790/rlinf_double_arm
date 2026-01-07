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

"""Replay buffer implementations for off-policy reinforcement learning.

This module provides replay buffer classes for storing and sampling
transitions for off-policy algorithms like SAC.

Key classes:
    - ReplayBuffer: Basic replay buffer for state-based observations
    - ImageReplayBuffer: Memory-efficient buffer for image observations
    - MixedReplayBuffer: Combines multiple buffers for RLPD-style sampling
"""

import copy
import pickle
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class Transition:
    """A single transition in the replay buffer."""
    observation: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: Dict[str, np.ndarray]
    done: bool
    # Optional fields for HIL
    is_intervention: bool = False
    info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "observations": self.observation,
            "actions": self.action,
            "rewards": self.reward,
            "next_observations": self.next_observation,
            "dones": self.done,
            "masks": 1.0 - float(self.done),
            "is_intervention": self.is_intervention,
        }


class ReplayBuffer:
    """Basic replay buffer for off-policy learning.
    
    Stores transitions as (obs, action, reward, next_obs, done) tuples
    and provides efficient random sampling.
    
    Args:
        capacity: Maximum number of transitions to store
        observation_shape: Shape of observation arrays (can be dict of shapes)
        action_shape: Shape of action arrays
        seed: Random seed for sampling
        
    Example:
        buffer = ReplayBuffer(
            capacity=100000,
            observation_shape={"state": (7,), "images": (3, 128, 128)},
            action_shape=(7,),
        )
        
        buffer.insert({
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "next_observations": next_obs,
            "dones": done,
        })
        
        batch = buffer.sample(batch_size=256)
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Optional[Union[tuple, Dict[str, tuple]]] = None,
        action_shape: Optional[tuple] = None,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        self._rng = np.random.RandomState(seed)
        self._lock = threading.Lock()
        
        # Storage - lazy initialization on first insert
        self._storage: Dict[str, np.ndarray] = {}
        self._size = 0
        self._insert_index = 0
        self._initialized = False
        
    def _initialize_storage(self, sample: Dict[str, Any]):
        """Initialize storage arrays based on first sample."""
        if self._initialized:
            return
            
        # Infer shapes from sample
        obs = sample.get("observations", {})
        if isinstance(obs, dict):
            self._storage["observations"] = {
                k: np.empty((self.capacity, *np.array(v).shape), dtype=np.float32)
                for k, v in obs.items()
            }
            self._storage["next_observations"] = {
                k: np.empty((self.capacity, *np.array(v).shape), dtype=np.float32)
                for k, v in obs.items()
            }
        else:
            obs = np.array(obs)
            self._storage["observations"] = np.empty(
                (self.capacity, *obs.shape), dtype=np.float32
            )
            self._storage["next_observations"] = np.empty(
                (self.capacity, *obs.shape), dtype=np.float32
            )
            
        action = np.array(sample.get("actions", []))
        self._storage["actions"] = np.empty(
            (self.capacity, *action.shape), dtype=np.float32
        )
        self._storage["rewards"] = np.empty((self.capacity,), dtype=np.float32)
        self._storage["dones"] = np.empty((self.capacity,), dtype=bool)
        self._storage["masks"] = np.empty((self.capacity,), dtype=np.float32)
        
        # Optional fields
        if "is_intervention" in sample:
            self._storage["is_intervention"] = np.empty((self.capacity,), dtype=bool)
            
        self._initialized = True
        
    def _insert_recursive(
        self, 
        storage: Union[Dict, np.ndarray], 
        data: Union[Dict, np.ndarray],
        index: int
    ):
        """Recursively insert data into storage."""
        if isinstance(storage, dict):
            for key in storage:
                if key in data:
                    self._insert_recursive(storage[key], data[key], index)
        else:
            storage[index] = np.array(data)
            
    def insert(self, transition: Dict[str, Any]):
        """Insert a transition into the buffer.
        
        Args:
            transition: Dictionary with keys:
                - observations: Current observation
                - actions: Action taken
                - rewards: Reward received
                - next_observations: Next observation
                - dones: Whether episode ended
                - masks: 1 - done (optional, computed if not provided)
                - is_intervention: Whether this was a human intervention (optional)
        """
        with self._lock:
            # Initialize storage on first insert
            if not self._initialized:
                self._initialize_storage(transition)
                
            # Compute mask if not provided
            if "masks" not in transition:
                transition["masks"] = 1.0 - float(transition["dones"])
                
            # Insert data
            for key, storage in self._storage.items():
                if key in transition:
                    self._insert_recursive(storage, transition[key], self._insert_index)
                    
            # Update indices
            self._insert_index = (self._insert_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            
    def _sample_recursive(
        self,
        storage: Union[Dict, np.ndarray],
        indices: np.ndarray,
    ) -> Union[Dict, np.ndarray]:
        """Recursively sample from storage."""
        if isinstance(storage, dict):
            return {k: self._sample_recursive(v, indices) for k, v in storage.items()}
        else:
            return storage[indices]
            
    def sample(
        self,
        batch_size: int,
        indices: Optional[np.ndarray] = None,
        to_torch: bool = False,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            indices: Optional specific indices to sample
            to_torch: If True, convert arrays to PyTorch tensors
            device: Device to put tensors on (if to_torch=True)
            
        Returns:
            Dictionary with sampled batch data
        """
        with self._lock:
            if indices is None:
                indices = self._rng.randint(0, self._size, size=batch_size)
                
            batch = {}
            for key, storage in self._storage.items():
                batch[key] = self._sample_recursive(storage, indices)
                
            if to_torch:
                batch = self._to_torch(batch, device)
                
            return batch
            
    def _to_torch(
        self, 
        data: Union[Dict, np.ndarray],
        device: Optional[str] = None,
    ) -> Union[Dict, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        if isinstance(data, dict):
            return {k: self._to_torch(v, device) for k, v in data.items()}
        else:
            tensor = torch.from_numpy(data)
            if device:
                tensor = tensor.to(device)
            return tensor
            
    def get_iterator(
        self,
        batch_size: int,
        num_batches: Optional[int] = None,
        to_torch: bool = False,
        device: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Get an iterator over batches.
        
        Args:
            batch_size: Size of each batch
            num_batches: Number of batches to yield (None = infinite)
            to_torch: Convert to PyTorch tensors
            device: Device for tensors
            
        Yields:
            Batch dictionaries
        """
        count = 0
        while num_batches is None or count < num_batches:
            yield self.sample(batch_size, to_torch=to_torch, device=device)
            count += 1
            
    def __len__(self) -> int:
        return self._size
        
    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity
        
    def save(self, filepath: Union[str, Path]):
        """Save buffer to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            data = {
                "storage": self._storage,
                "size": self._size,
                "insert_index": self._insert_index,
                "capacity": self.capacity,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
    def load(self, filepath: Union[str, Path]):
        """Load buffer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        with self._lock:
            self._storage = data["storage"]
            self._size = data["size"]
            self._insert_index = data["insert_index"]
            self.capacity = data["capacity"]
            self._initialized = True


class ImageReplayBuffer(ReplayBuffer):
    """Memory-efficient replay buffer for image observations.
    
    Stores images in uint8 format to save memory and only converts
    to float32 when sampling. Also supports frame stacking without
    duplicating frames.
    
    Args:
        capacity: Maximum number of transitions
        image_keys: Keys in observation dict that contain images
        num_stack: Number of frames to stack (if using frame stacking)
        seed: Random seed
    """
    
    def __init__(
        self,
        capacity: int,
        image_keys: Tuple[str, ...] = ("images",),
        num_stack: int = 1,
        seed: int = 42,
    ):
        super().__init__(capacity, seed=seed)
        self.image_keys = image_keys
        self.num_stack = num_stack
        
        # Track valid indices for frame stacking
        self._is_valid = np.zeros(capacity, dtype=bool)
        
    def _initialize_storage(self, sample: Dict[str, Any]):
        """Initialize storage with uint8 for images."""
        if self._initialized:
            return
            
        obs = sample.get("observations", {})
        if isinstance(obs, dict):
            self._storage["observations"] = {}
            self._storage["next_observations"] = {}
            
            for k, v in obs.items():
                v = np.array(v)
                if k in self.image_keys:
                    # Store images as uint8
                    # For frame stacking, only store single frame
                    if self.num_stack > 1 and len(v.shape) == 4:
                        # v shape: (num_stack, H, W, C) -> store (H, W, C)
                        frame_shape = v.shape[1:]
                    else:
                        frame_shape = v.shape
                    self._storage["observations"][k] = np.empty(
                        (self.capacity, *frame_shape), dtype=np.uint8
                    )
                    # For next_obs, only store the newest frame
                    self._storage["next_observations"][k] = np.empty(
                        (self.capacity, *frame_shape), dtype=np.uint8
                    )
                else:
                    # Non-image data
                    self._storage["observations"][k] = np.empty(
                        (self.capacity, *v.shape), dtype=np.float32
                    )
                    self._storage["next_observations"][k] = np.empty(
                        (self.capacity, *v.shape), dtype=np.float32
                    )
        else:
            super()._initialize_storage(sample)
            return
            
        action = np.array(sample.get("actions", []))
        self._storage["actions"] = np.empty(
            (self.capacity, *action.shape), dtype=np.float32
        )
        self._storage["rewards"] = np.empty((self.capacity,), dtype=np.float32)
        self._storage["dones"] = np.empty((self.capacity,), dtype=bool)
        self._storage["masks"] = np.empty((self.capacity,), dtype=np.float32)
        
        if "is_intervention" in sample:
            self._storage["is_intervention"] = np.empty((self.capacity,), dtype=bool)
            
        self._initialized = True
        
    def insert(self, transition: Dict[str, Any]):
        """Insert transition, storing only newest frame for images."""
        with self._lock:
            if not self._initialized:
                self._initialize_storage(transition)
                
            # Handle frame stacking - only store newest frame
            transition = copy.deepcopy(transition)
            obs = transition.get("observations", {})
            next_obs = transition.get("next_observations", {})
            
            if isinstance(obs, dict):
                for k in self.image_keys:
                    if k in obs:
                        img = np.array(obs[k])
                        if self.num_stack > 1 and len(img.shape) == 4:
                            obs[k] = img[-1]  # Only newest frame
                        next_img = np.array(next_obs[k])
                        if self.num_stack > 1 and len(next_img.shape) == 4:
                            next_obs[k] = next_img[-1]
                            
                transition["observations"] = obs
                transition["next_observations"] = next_obs
                
            # Mark as valid
            self._is_valid[self._insert_index] = True
            
            # Invalidate indices that would break frame stacking
            if self.num_stack > 1:
                for i in range(1, self.num_stack):
                    idx = (self._insert_index + i) % self.capacity
                    self._is_valid[idx] = False
                    
            # Call parent insert
            super().insert(transition)
            
    def sample(
        self,
        batch_size: int,
        indices: Optional[np.ndarray] = None,
        to_torch: bool = False,
        device: Optional[str] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> Dict[str, Any]:
        """Sample with frame stacking support."""
        with self._lock:
            if indices is None:
                # Sample only from valid indices
                valid_indices = np.where(self._is_valid[:self._size])[0]
                if len(valid_indices) < batch_size:
                    # Fall back to any indices if not enough valid
                    indices = self._rng.randint(0, self._size, size=batch_size)
                else:
                    indices = self._rng.choice(valid_indices, size=batch_size, replace=False)
                    
            batch = {}
            for key, storage in self._storage.items():
                if key in ("observations", "next_observations") and isinstance(storage, dict):
                    batch[key] = {}
                    for k, v in storage.items():
                        if k in self.image_keys and self.num_stack > 1:
                            # Reconstruct frame stack
                            frames = []
                            for i in range(self.num_stack):
                                frame_idx = (indices - self.num_stack + 1 + i) % self.capacity
                                frames.append(v[frame_idx])
                            stacked = np.stack(frames, axis=1)  # (B, T, H, W, C)
                            batch[key][k] = stacked.astype(np.float32) / 255.0
                        else:
                            batch[key][k] = v[indices]
                else:
                    batch[key] = self._sample_recursive(storage, indices)
                    
            if to_torch:
                batch = self._to_torch(batch, device)
                
            return batch


class MixedReplayBuffer:
    """Buffer that samples from multiple sources with configurable ratios.
    
    Useful for RLPD-style training where we sample 50/50 from:
        - Online replay buffer (robot experience)
        - Demo buffer (human demonstrations/interventions)
        
    Args:
        buffers: Dictionary mapping buffer names to ReplayBuffer instances
        sample_ratios: Dictionary mapping buffer names to sampling ratios
                      (will be normalized to sum to 1)
        seed: Random seed
        
    Example:
        mixed_buffer = MixedReplayBuffer(
            buffers={
                "online": online_buffer,
                "demo": demo_buffer,
            },
            sample_ratios={"online": 0.5, "demo": 0.5},
        )
        
        batch = mixed_buffer.sample(batch_size=256)
    """
    
    def __init__(
        self,
        buffers: Dict[str, ReplayBuffer],
        sample_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.buffers = buffers
        self._rng = np.random.RandomState(seed)
        
        # Default to equal ratios
        if sample_ratios is None:
            sample_ratios = {name: 1.0 / len(buffers) for name in buffers}
            
        # Normalize ratios
        total = sum(sample_ratios.values())
        self.sample_ratios = {k: v / total for k, v in sample_ratios.items()}
        
    def sample(
        self,
        batch_size: int,
        to_torch: bool = False,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sample from all buffers according to ratios.
        
        Args:
            batch_size: Total batch size (split across buffers by ratio)
            to_torch: Convert to PyTorch tensors
            device: Device for tensors
            
        Returns:
            Combined batch from all buffers
        """
        batches = []
        remaining = batch_size
        
        # Calculate samples per buffer
        buffer_names = list(self.buffers.keys())
        for i, name in enumerate(buffer_names):
            if i == len(buffer_names) - 1:
                # Last buffer gets remaining to handle rounding
                n_samples = remaining
            else:
                n_samples = int(batch_size * self.sample_ratios[name])
                remaining -= n_samples
                
            if n_samples > 0 and len(self.buffers[name]) > 0:
                batch = self.buffers[name].sample(
                    n_samples, to_torch=to_torch, device=device
                )
                batches.append(batch)
                
        # Concatenate batches
        if len(batches) == 0:
            raise ValueError("All buffers are empty")
        elif len(batches) == 1:
            return batches[0]
        else:
            return self._concat_batches(batches)
            
    def _concat_batches(self, batches: List[Dict]) -> Dict:
        """Concatenate multiple batch dictionaries."""
        result = {}
        for key in batches[0]:
            values = [b[key] for b in batches if key in b]
            if isinstance(values[0], dict):
                result[key] = self._concat_batches(values)
            elif isinstance(values[0], np.ndarray):
                result[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = values
        return result
        
    def get_iterator(
        self,
        batch_size: int,
        num_batches: Optional[int] = None,
        to_torch: bool = False,
        device: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Get iterator over mixed batches."""
        count = 0
        while num_batches is None or count < num_batches:
            yield self.sample(batch_size, to_torch=to_torch, device=device)
            count += 1
            
    def __len__(self) -> int:
        return sum(len(b) for b in self.buffers.values())
        
    @property
    def buffer_sizes(self) -> Dict[str, int]:
        """Get size of each buffer."""
        return {name: len(buf) for name, buf in self.buffers.items()}

