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

"""Human-in-the-Loop SAC Runner.

This runner implements SAC training with human interventions for real robot
learning, following the RLPD/SERL paradigm:
    - Actor interacts with environment, human can intervene
    - Intervention data stored in separate demo buffer
    - Learner samples 50/50 from online and demo buffers
    - High UTD (update-to-data ratio) for sample efficiency

Key features:
    - Human intervention detection and recording
    - Mixed replay buffer (online + demonstrations)
    - SAC with automatic temperature tuning
    - Checkpoint saving and resuming
    - Integration with RLinf's distributed training

Usage:
    runner = HILSACRunner(
        env=robot_env,
        config=config,
        input_device="vivetracker",
    )
    runner.train(max_steps=100000)
"""

import copy
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.algorithms.sac import SACLosses, soft_target_update
from rlinf.data.replay_buffer import (
    ImageReplayBuffer,
    MixedReplayBuffer,
    ReplayBuffer,
)
from rlinf.envs.hil.intervention_wrapper import HILInterventionWrapper
from rlinf.envs.hil.input_devices import InputDeviceConfig, get_input_device
from rlinf.models.sac.actor import SquashedGaussianActor, create_actor
from rlinf.models.sac.critic import TwinQCritic, create_critic
from rlinf.models.sac.encoders import MultiModalEncoder
from rlinf.utils.metric_logger import MetricLogger


@dataclass
class HILSACConfig:
    """Configuration for HIL-SAC/RLPD training.
    
    Default values match HIL-SERL exactly.
    """
    
    # Environment
    max_episode_steps: int = 100  # max_traj_length in HIL-SERL
    
    # Replay buffers
    replay_buffer_capacity: int = 200000  # HIL-SERL default
    demo_buffer_capacity: int = 200000    # Same as replay buffer
    training_starts: int = 100            # HIL-SERL default
    
    # Training
    batch_size: int = 256          # HIL-SERL default
    learning_rate: float = 3e-4    # HIL-SERL default
    discount: float = 0.97         # HIL-SERL default (gamma)
    tau: float = 0.005             # soft_target_update_rate in HIL-SERL
    
    # SAC
    init_temperature: float = 0.01  # temperature_init=1e-2 in HIL-SERL
    target_entropy: Optional[float] = None  # Default: -action_dim / 2
    backup_entropy: bool = False    # HIL-SERL default
    critic_ensemble_size: int = 2   # HIL-SERL default
    critic_subsample_size: Optional[int] = None  # HIL-SERL default
    
    # Update schedule (HIL-SERL style)
    # cta_ratio means: (cta_ratio - 1) critic-only updates, then 1 full update
    cta_ratio: int = 2              # Critic-to-Actor ratio
    steps_per_update: int = 50      # Network publish frequency
    random_steps: int = 0           # Random action steps at start
    
    # RLPD-style sampling (50/50 from online and demo buffers)
    demo_ratio: float = 0.5  # batch_size // 2 from each buffer
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    encoder_type: str = "resnet-pretrained"  # HIL-SERL default
    image_keys: Tuple[str, ...] = ("image",)  # HIL-SERL default key
    use_proprio: bool = True
    
    # Policy config (matching HIL-SERL)
    tanh_squash_distribution: bool = True
    std_parameterization: str = "exp"
    std_min: float = 1e-5
    std_max: float = 5.0
    
    # Network config (matching HIL-SERL)
    use_layer_norm: bool = True
    activations: str = "tanh"
    
    # Input device for HIL
    input_device: str = "mock"  # "mock", "keyboard", "vivetracker"
    
    # Logging (matching HIL-SERL)
    log_period: int = 10
    eval_period: int = 2000
    checkpoint_period: int = 0   # 0 = disabled
    buffer_period: int = 0       # 0 = disabled
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    buffer_dir: str = "./buffers"
    
    # Deprecated aliases for backward compatibility
    @property
    def gamma(self) -> float:
        return self.discount
    
    @property
    def min_buffer_size(self) -> int:
        return self.training_starts


class HILSACRunner:
    """Runner for Human-in-the-Loop SAC training.
    
    This implements the RLPD/SERL training loop:
        1. Actor collects experience, human can intervene
        2. All transitions go to online buffer
        3. Intervention transitions also go to demo buffer
        4. Learner samples from both buffers with configurable ratio
        5. High UTD training for sample efficiency
        
    Args:
        env: Robot environment (will be wrapped with HIL wrapper)
        config: Training configuration
        demo_paths: Optional paths to pre-collected demonstrations
        resume_path: Optional path to resume from checkpoint
    """
    
    def __init__(
        self,
        env,
        config: Union[HILSACConfig, DictConfig] = None,
        demo_paths: Optional[List[str]] = None,
        resume_path: Optional[str] = None,
    ):
        if config is None:
            config = HILSACConfig()
        elif isinstance(config, DictConfig):
            config = self._dictconfig_to_dataclass(config)
            
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Wrap environment with HIL wrapper
        input_config = InputDeviceConfig(
            action_dim=env.action_space.shape[0],
        )
        self.env = HILInterventionWrapper(
            env,
            input_device=config.input_device,
            input_device_config=input_config,
        )
        
        # Get dimensions
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]
        
        # Create models
        self._create_models()
        
        # Create replay buffers
        self._create_buffers()
        
        # Load pre-collected demos if provided
        if demo_paths:
            self._load_demonstrations(demo_paths)
            
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_return = float('-inf')
        
        # Resume from checkpoint if provided
        if resume_path:
            self.load_checkpoint(resume_path)
            
        # Logging
        self.metric_logger = MetricLogger(
            DictConfig({"runner": {"logger": {"log_path": config.checkpoint_dir}}})
        )
        
    def _dictconfig_to_dataclass(self, cfg: DictConfig) -> HILSACConfig:
        """Convert OmegaConf DictConfig to HILSACConfig."""
        return HILSACConfig(
            **{k: v for k, v in cfg.items() if hasattr(HILSACConfig, k)}
        )
        
    def _create_models(self):
        """Create actor, critic, and target networks."""
        config = self.config
        
        # Determine observation dimension
        if isinstance(self.obs_space, dict):
            # Use encoder for dict observations
            obs_dim = 256  # Encoder output dim
            
            # Get image shape
            if config.image_keys[0] in self.obs_space:
                img_space = self.obs_space[config.image_keys[0]]
                image_shape = img_space.shape
            else:
                image_shape = (3, 128, 128)
                
            # Get state dim
            state_dim = 0
            if "state" in self.obs_space and config.use_proprio:
                state_space = self.obs_space["state"]
                if hasattr(state_space, 'shape'):
                    state_dim = state_space.shape[0]
                elif isinstance(state_space, dict):
                    state_dim = sum(
                        s.shape[0] for s in state_space.values() 
                        if hasattr(s, 'shape')
                    )
                    
            encoder = MultiModalEncoder(
                image_keys=config.image_keys,
                image_shape=image_shape,
                state_dim=state_dim,
                output_dim=obs_dim,
                encoder_type=config.encoder_type,
                use_proprio=config.use_proprio,
            )
        else:
            obs_dim = self.obs_space.shape[0]
            encoder = None
            
        # Create actor
        self.actor = create_actor(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dims=config.hidden_dims,
            encoder=encoder,
            squashed=True,
        ).to(self.device)
        
        # Create critic (with separate encoder)
        critic_encoder = copy.deepcopy(encoder) if encoder else None
        self.critic = create_critic(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dims=config.hidden_dims,
            encoder=critic_encoder,
            critic_type="twin",
        ).to(self.device)
        
        # Create target critic
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
            
        # Temperature (log_alpha)
        target_entropy = config.target_entropy
        if target_entropy is None:
            target_entropy = -self.action_dim
        self.target_entropy = target_entropy
        
        self.log_alpha = torch.tensor(
            np.log(config.init_temperature),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=config.learning_rate
        )
        
        # SAC losses helper
        self.sac_losses = SACLosses(
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            backup_entropy=config.backup_entropy,
        )
        
    def _create_buffers(self):
        """Create replay buffers."""
        config = self.config
        
        # Check if we need image buffer
        use_images = len(config.image_keys) > 0
        
        if use_images:
            BufferClass = ImageReplayBuffer
            buffer_kwargs = {"image_keys": config.image_keys}
        else:
            BufferClass = ReplayBuffer
            buffer_kwargs = {}
            
        # Online replay buffer
        self.online_buffer = BufferClass(
            capacity=config.replay_buffer_capacity,
            **buffer_kwargs,
        )
        
        # Demo buffer for interventions
        self.demo_buffer = BufferClass(
            capacity=config.demo_buffer_capacity,
            **buffer_kwargs,
        )
        
        # Mixed buffer for sampling
        self.mixed_buffer = MixedReplayBuffer(
            buffers={
                "online": self.online_buffer,
                "demo": self.demo_buffer,
            },
            sample_ratios={
                "online": 1 - config.demo_ratio,
                "demo": config.demo_ratio,
            },
        )
        
    def _load_demonstrations(self, demo_paths: List[str]):
        """Load pre-collected demonstrations into demo buffer."""
        for path in demo_paths:
            print(f"Loading demonstrations from: {path}")
            with open(path, 'rb') as f:
                demos = pickle.load(f)
                
            for transition in demos:
                self.demo_buffer.insert(transition)
                
        print(f"Loaded {len(self.demo_buffer)} demonstration transitions")
        
    def _process_observation(
        self, 
        obs: Union[np.ndarray, Dict],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process observation for network input."""
        if isinstance(obs, dict):
            processed = {}
            for key, value in obs.items():
                if isinstance(value, dict):
                    processed[key] = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                        for k, v in value.items()
                    }
                else:
                    processed[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            return processed
        else:
            return torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
    @torch.no_grad()
    def select_action(
        self,
        obs: Union[np.ndarray, Dict],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action using current policy.
        
        Args:
            obs: Current observation
            deterministic: If True, return mean action
            
        Returns:
            Action array
        """
        obs_tensor = self._process_observation(obs)
        action = self.actor.sample(obs_tensor, deterministic=deterministic)
        return action.cpu().numpy()[0]
        
    def train_step(self) -> Dict[str, float]:
        """Perform one training step (multiple gradient updates).
        
        Returns:
            Dictionary of training metrics
        """
        config = self.config
        
        if len(self.online_buffer) < config.min_buffer_size:
            return {}
            
        metrics = defaultdict(list)
        
        # Multiple updates per step (high UTD / CTA ratio)
        # cta_ratio means: (cta_ratio - 1) critic-only updates, then 1 full update
        for update_idx in range(config.cta_ratio):
            # Sample batch
            batch = self.mixed_buffer.sample(
                config.batch_size,
                to_torch=True,
                device=str(self.device),
            )
            
            # HIL-SERL style: update actor only on the last iteration
            # cta_ratio=2 means 1 critic-only update, then 1 full update
            update_actor = (update_idx == config.cta_ratio - 1)
            
            # Extract batch components
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_observations = batch["next_observations"]
            dones = batch["dones"]
            
            # Critic update (always)
            critic_loss, critic_metrics = self.sac_losses.compute_critic_loss(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
                critic=self.critic,
                target_critic=self.target_critic,
                actor=self.actor,
                log_alpha=self.log_alpha,
            )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            step_metrics = {f"critic/{k}": v for k, v in critic_metrics.items()}
            
            # Actor and temperature update (only on last iteration)
            if update_actor:
                # Actor update
                actor_loss, actor_metrics = self.sac_losses.compute_actor_loss(
                    observations=observations,
                    actor=self.actor,
                    critic=self.critic,
                    log_alpha=self.log_alpha,
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                step_metrics.update({f"actor/{k}": v for k, v in actor_metrics.items()})
                
                # Temperature update
                temp_loss, temp_metrics = self.sac_losses.compute_temperature_loss(
                    observations=observations,
                    actor=self.actor,
                    log_alpha=self.log_alpha,
                    action_dim=self.action_dim,
                )
                self.alpha_optimizer.zero_grad()
                temp_loss.backward()
                self.alpha_optimizer.step()
                step_metrics.update({f"temperature/{k}": v for k, v in temp_metrics.items()})
                
            # Soft target update
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_critic.parameters(),
                    self.critic.parameters()
                ):
                    target_param.data.copy_(
                        config.tau * param.data + 
                        (1 - config.tau) * target_param.data
                    )
                    
            # Collect metrics
            for key, value in step_metrics.items():
                metrics[key].append(value)
                
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
        
    def collect_episode(self) -> Dict[str, float]:
        """Collect one episode of experience.
        
        Returns:
            Episode statistics
        """
        obs, _ = self.env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        intervention_steps = 0
        
        while not done and episode_length < self.config.max_episode_steps:
            # Select action
            action = self.select_action(obs, deterministic=False)
            
            # Step environment (HIL wrapper handles intervention)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Get executed action (may be overridden by human)
            executed_action = info.get("executed_action", action)
            is_intervention = info.get("is_intervention", False)
            
            # Create transition
            transition = {
                "observations": obs,
                "actions": executed_action,
                "rewards": reward,
                "next_observations": next_obs,
                "dones": done,
                "is_intervention": is_intervention,
            }
            
            # Store in buffers
            self.online_buffer.insert(transition)
            
            if is_intervention:
                self.demo_buffer.insert(transition)
                intervention_steps += 1
                
            # Update
            obs = next_obs
            episode_return += reward
            episode_length += 1
            self.global_step += 1
            
            # Train
            if self.global_step >= self.config.min_buffer_size:
                self.train_step()
                
        self.episode_count += 1
        
        return {
            "episode_return": episode_return,
            "episode_length": episode_length,
            "intervention_steps": intervention_steps,
            "intervention_rate": intervention_steps / episode_length if episode_length > 0 else 0,
        }
        
    def train(
        self,
        max_steps: int = 100000,
        max_episodes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Main training loop.
        
        Args:
            max_steps: Maximum number of environment steps
            max_episodes: Maximum number of episodes (optional)
            
        Returns:
            Final training statistics
        """
        config = self.config
        
        pbar = tqdm(total=max_steps, desc="Training", initial=self.global_step)
        
        try:
            while self.global_step < max_steps:
                if max_episodes and self.episode_count >= max_episodes:
                    break
                    
                # Collect episode
                episode_stats = self.collect_episode()
                
                # Update progress bar
                pbar.update(episode_stats["episode_length"])
                pbar.set_postfix({
                    "ep": self.episode_count,
                    "ret": f"{episode_stats['episode_return']:.2f}",
                    "int": f"{episode_stats['intervention_rate']:.2%}",
                })
                
                # Log
                if self.episode_count % config.log_interval == 0:
                    self.metric_logger.log(
                        {f"train/{k}": v for k, v in episode_stats.items()},
                        step=self.global_step,
                    )
                    self.metric_logger.log(
                        {
                            "buffer/online_size": len(self.online_buffer),
                            "buffer/demo_size": len(self.demo_buffer),
                        },
                        step=self.global_step,
                    )
                    
                # Save checkpoint
                if self.global_step % config.save_interval == 0:
                    self.save_checkpoint()
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
        finally:
            pbar.close()
            self.save_checkpoint()
            
        return {"final_step": self.global_step, "episodes": self.episode_count}
        
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint.
        
        Args:
            path: Optional path to save checkpoint
        """
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_{self.global_step}.pt"
            )
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
        
        # Also save buffers
        buffer_path = os.path.join(
            self.config.buffer_dir, 
            f"buffers_{self.global_step}.pkl"
        )
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        self.online_buffer.save(buffer_path.replace(".pkl", "_online.pkl"))
        self.demo_buffer.save(buffer_path.replace(".pkl", "_demo.pkl"))
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.episode_count = checkpoint["episode_count"]
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.log_alpha = checkpoint["log_alpha"].to(self.device).requires_grad_(True)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        
        print(f"Resumed from step {self.global_step}, episode {self.episode_count}")
        
    @torch.no_grad()
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            
        Returns:
            Evaluation statistics
        """
        returns = []
        lengths = []
        successes = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done and episode_length < self.config.max_episode_steps:
                action = self.select_action(obs, deterministic=deterministic)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                obs = next_obs
                episode_return += reward
                episode_length += 1
                
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(info.get("succeed", reward > 0))
            
        return {
            "eval/mean_return": np.mean(returns),
            "eval/std_return": np.std(returns),
            "eval/mean_length": np.mean(lengths),
            "eval/success_rate": np.mean(successes),
        }

