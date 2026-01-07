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

"""Unit tests for Human-in-the-Loop intervention wrapper.

This module tests:
    - HILStats: Statistics tracking for intervention
    - HILInterventionWrapper: Core intervention wrapper functionality
    - ActionRecordingWrapper: Action recording for playback/analysis
    - Integration of both wrappers together

Run with: pytest tests/unit_tests/test_intervention_wrapper.py -v
"""

import tempfile
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rlinf.envs.hil.input_devices import (
    BaseInputDevice,
    InputDeviceConfig,
    MockInputDevice,
)
from rlinf.envs.hil.intervention_wrapper import (
    ActionRecordingWrapper,
    HILInterventionWrapper,
    HILStats,
)


# =============================================================================
# Mock Environment for Testing
# =============================================================================


class MockRobotEnv(gym.Env):
    """A simple mock robot environment for testing intervention wrappers.
    
    This simulates a 7-DoF robot arm with:
        - Observation: 14D (position + velocity)
        - Action: 7D (xyz + rpy + gripper)
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        obs_dim: int = 14,
        max_episode_steps: int = 100,
        reward_type: str = "sparse",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        
        # Define spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State tracking
        self._step_count = 0
        self._state = None
        self._last_action = None
        
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._state = np.zeros(self.obs_dim, dtype=np.float32)
        self._last_action = None
        return self._state.copy(), {"reset": True}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        self._last_action = action.copy()
        
        # Simple dynamics: state changes based on action
        self._state[:self.action_dim] += action * 0.1
        
        # Reward based on action magnitude
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(action) > 0.5 else 0.0
        else:
            reward = -np.linalg.norm(action)
        
        # Termination
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        
        info = {
            "step_count": self._step_count,
            "action_norm": float(np.linalg.norm(action)),
        }
        
        return self._state.copy(), reward, terminated, truncated, info
    
    def get_last_action(self) -> np.ndarray:
        """Get the last action that was executed."""
        return self._last_action


class ControlledMockInputDevice(BaseInputDevice):
    """A mock input device with explicit control over intervention state.
    
    Use set_next_action() to specify what the next get_action() call returns.
    """
    
    def __init__(self, config: InputDeviceConfig = None):
        super().__init__(config)
        self._next_action = None
        self._next_is_intervening = False
        self._action_sequence = []
        self._sequence_index = 0
        
    def get_action(self) -> Tuple[np.ndarray, bool]:
        # If we have a sequence, use it
        if self._action_sequence and self._sequence_index < len(self._action_sequence):
            action, is_intervening = self._action_sequence[self._sequence_index]
            self._sequence_index += 1
            return action, is_intervening
            
        # Otherwise use next_action or default
        if self._next_action is not None:
            action = self._next_action
            is_intervening = self._next_is_intervening
        else:
            action = np.zeros(self.config.action_dim)
            is_intervening = False
            
        return action, is_intervening
    
    def set_next_action(self, action: np.ndarray, is_intervening: bool = True):
        """Set what the next get_action() call should return."""
        self._next_action = action
        self._next_is_intervening = is_intervening
        
    def set_action_sequence(self, sequence: list):
        """Set a sequence of (action, is_intervening) tuples to return."""
        self._action_sequence = sequence
        self._sequence_index = 0
        
    def clear(self):
        """Clear next action and sequence."""
        self._next_action = None
        self._next_is_intervening = False
        self._action_sequence = []
        self._sequence_index = 0


# =============================================================================
# Tests for HILStats
# =============================================================================


class TestHILStats:
    """Tests for the HILStats dataclass."""
    
    def test_initial_state(self):
        """Test that HILStats initializes with correct default values."""
        stats = HILStats()
        assert stats.total_steps == 0
        assert stats.intervention_steps == 0
        assert stats.intervention_count == 0
        assert stats.episode_count == 0
        assert stats.intervention_rate == 0.0
        
    def test_step_without_intervention(self):
        """Test stepping without intervention updates stats correctly."""
        stats = HILStats()
        stats.step(is_intervention=False)
        
        assert stats.total_steps == 1
        assert stats.intervention_steps == 0
        assert stats.intervention_count == 0
        assert stats.intervention_rate == 0.0
        
    def test_step_with_intervention(self):
        """Test stepping with intervention updates stats correctly."""
        stats = HILStats()
        stats.step(is_intervention=True)
        
        assert stats.total_steps == 1
        assert stats.intervention_steps == 1
        assert stats.intervention_count == 1
        assert stats.intervention_rate == 1.0
        
    def test_consecutive_interventions_count_as_one(self):
        """Test that consecutive intervention steps count as one intervention episode."""
        stats = HILStats()
        
        # Three consecutive intervention steps should be one intervention episode
        stats.step(is_intervention=True)
        stats.step(is_intervention=True)
        stats.step(is_intervention=True)
        
        assert stats.intervention_steps == 3
        assert stats.intervention_count == 1
        
    def test_separate_intervention_episodes(self):
        """Test that non-consecutive interventions count as separate episodes."""
        stats = HILStats()
        
        # First intervention episode
        stats.step(is_intervention=True)
        stats.step(is_intervention=True)
        
        # Break in intervention
        stats.step(is_intervention=False)
        stats.step(is_intervention=False)
        
        # Second intervention episode
        stats.step(is_intervention=True)
        
        assert stats.intervention_steps == 3
        assert stats.intervention_count == 2
        assert stats.total_steps == 5
        
    def test_intervention_rate_calculation(self):
        """Test that intervention rate is calculated correctly."""
        stats = HILStats()
        
        # 30% intervention rate
        for _ in range(7):
            stats.step(is_intervention=False)
        for _ in range(3):
            stats.step(is_intervention=True)
            
        assert stats.intervention_rate == pytest.approx(0.3, rel=1e-3)
        
    def test_end_episode(self):
        """Test that end_episode returns correct stats and resets counters."""
        stats = HILStats()
        
        # Do some steps with intervention
        stats.step(is_intervention=True)
        stats.step(is_intervention=True)
        stats.step(is_intervention=False)
        stats.step(is_intervention=True)
        
        episode_stats = stats.end_episode()
        
        # Check returned episode stats
        assert episode_stats["episode_interventions"] == 2  # Two separate episodes
        assert episode_stats["episode_intervention_steps"] == 3
        
        # Check global stats still preserved
        assert stats.episode_count == 1
        assert stats.intervention_steps == 3
        
        # Check per-episode counters reset
        assert stats.current_episode_interventions == 0
        assert stats.current_episode_intervention_steps == 0
        
    def test_to_dict(self):
        """Test that to_dict returns correctly formatted dictionary."""
        stats = HILStats()
        stats.step(is_intervention=True)
        stats.step(is_intervention=False)
        stats.end_episode()
        
        d = stats.to_dict()
        
        assert "hil/total_steps" in d
        assert "hil/intervention_steps" in d
        assert "hil/intervention_count" in d
        assert "hil/intervention_rate" in d
        assert "hil/episode_count" in d
        
        assert d["hil/total_steps"] == 2
        assert d["hil/intervention_steps"] == 1
        assert d["hil/episode_count"] == 1


# =============================================================================
# Tests for HILInterventionWrapper
# =============================================================================


class TestHILInterventionWrapper:
    """Tests for the HILInterventionWrapper class."""
    
    @pytest.fixture
    def env(self):
        """Create a mock robot environment."""
        return MockRobotEnv()
    
    @pytest.fixture
    def controlled_device(self):
        """Create a controlled mock input device."""
        config = InputDeviceConfig(action_dim=7)
        return ControlledMockInputDevice(config)
    
    def test_wrapper_initialization_with_mock_device(self, env):
        """Test wrapper initialization with 'mock' device string."""
        hil_env = HILInterventionWrapper(env, input_device="mock")
        
        assert hil_env.input_device is not None
        assert isinstance(hil_env.input_device, MockInputDevice)
        assert hil_env.input_device.is_running
        
        hil_env.close()
        
    def test_wrapper_initialization_with_device_instance(self, env, controlled_device):
        """Test wrapper initialization with BaseInputDevice instance."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        
        assert hil_env.input_device is controlled_device
        hil_env.close()
        
    def test_step_without_intervention(self, env, controlled_device):
        """Test step execution without human intervention."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        hil_env.reset()
        
        # Set device to not intervene
        controlled_device.set_next_action(np.zeros(7), is_intervening=False)
        
        policy_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        # Check info fields
        assert info["is_intervention"] is False
        assert "intervene_action" not in info
        np.testing.assert_array_almost_equal(info["policy_action"], policy_action)
        np.testing.assert_array_almost_equal(info["executed_action"], policy_action)
        
        # Verify that policy action was executed
        np.testing.assert_array_almost_equal(env.get_last_action(), policy_action)
        
        hil_env.close()
        
    def test_step_with_intervention(self, env, controlled_device):
        """Test step execution with human intervention replacing policy action."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        hil_env.reset()
        
        # Set device to intervene with specific action
        human_action = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0])
        controlled_device.set_next_action(human_action, is_intervening=True)
        
        policy_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        # Check info fields
        assert info["is_intervention"] is True
        np.testing.assert_array_almost_equal(info["intervene_action"], human_action)
        np.testing.assert_array_almost_equal(info["policy_action"], policy_action)
        np.testing.assert_array_almost_equal(info["executed_action"], human_action)
        
        # Verify that human action was executed (not policy action)
        np.testing.assert_array_almost_equal(env.get_last_action(), human_action)
        
        hil_env.close()
        
    def test_blended_actions(self, env, controlled_device):
        """Test blended action mode where human and policy actions are combined."""
        hil_env = HILInterventionWrapper(
            env, 
            input_device=controlled_device,
            blend_actions=True,
            blend_alpha=0.5,
        )
        hil_env.reset()
        
        # Set device to intervene
        human_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        controlled_device.set_next_action(human_action, is_intervening=True)
        
        policy_action = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        # Expected blended action: 0.5 * human + 0.5 * policy
        expected_action = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(info["executed_action"], expected_action)
        np.testing.assert_array_almost_equal(env.get_last_action(), expected_action)
        
        hil_env.close()
        
    def test_stats_tracking(self, env, controlled_device):
        """Test that intervention statistics are tracked correctly."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        hil_env.reset()
        
        # Set up a sequence: intervene, intervene, no, no, intervene
        sequence = [
            (np.zeros(7), True),   # Intervention episode 1, step 1
            (np.zeros(7), True),   # Intervention episode 1, step 2
            (np.zeros(7), False),  # No intervention
            (np.zeros(7), False),  # No intervention
            (np.zeros(7), True),   # Intervention episode 2, step 1
        ]
        controlled_device.set_action_sequence(sequence)
        
        policy_action = np.zeros(7)
        for _ in range(5):
            hil_env.step(policy_action)
            
        stats = hil_env.get_stats()
        
        assert stats["hil/total_steps"] == 5
        assert stats["hil/intervention_steps"] == 3
        assert stats["hil/intervention_count"] == 2
        assert stats["hil/intervention_rate"] == pytest.approx(0.6, rel=1e-3)
        
        hil_env.close()
        
    def test_episode_end_stats(self, env, controlled_device):
        """Test that episode end returns intervention statistics in info."""
        env = MockRobotEnv(max_episode_steps=3)  # Short episode
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        hil_env.reset()
        
        # Set up interventions
        sequence = [
            (np.zeros(7), True),   # Intervention
            (np.zeros(7), False),  # No intervention
            (np.zeros(7), True),   # Intervention
        ]
        controlled_device.set_action_sequence(sequence)
        
        policy_action = np.zeros(7)
        for _ in range(2):
            hil_env.step(policy_action)
            
        # Third step ends episode
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        assert truncated is True
        assert "episode_intervention_count" in info
        assert "episode_intervention_steps" in info
        assert info["episode_intervention_count"] == 2
        assert info["episode_intervention_steps"] == 2
        
        hil_env.close()
        
    def test_reset_clears_input_device_state(self, env, controlled_device):
        """Test that reset() properly resets the input device."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        
        # Do some steps
        hil_env.reset()
        controlled_device.set_next_action(np.ones(7), is_intervening=True)
        hil_env.step(np.zeros(7))
        
        # Reset and verify
        obs, info = hil_env.reset()
        assert info["is_intervention"] is False
        
        hil_env.close()
        
    def test_set_intervention_mode(self, env):
        """Test programmatically setting intervention mode."""
        hil_env = HILInterventionWrapper(env, input_device="mock")
        hil_env.reset()
        
        # Enable intervention mode
        hil_env.set_intervention_mode(enabled=True)
        
        obs, reward, terminated, truncated, info = hil_env.step(np.ones(7))
        assert info["is_intervention"] is True
        
        # Disable intervention mode
        hil_env.set_intervention_mode(enabled=False)
        
        obs, reward, terminated, truncated, info = hil_env.step(np.ones(7))
        assert info["is_intervention"] is False
        
        hil_env.close()
        
    def test_action_copies_are_independent(self, env, controlled_device):
        """Test that returned action arrays are copies, not references."""
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        hil_env.reset()
        
        human_action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        controlled_device.set_next_action(human_action, is_intervening=True)
        
        policy_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        
        # Modify original arrays
        policy_action[:] = 0
        human_action[:] = 0
        
        # Returned info should not be affected
        assert not np.allclose(info["policy_action"], np.zeros(7))
        assert not np.allclose(info["intervene_action"], np.zeros(7))
        
        hil_env.close()


# =============================================================================
# Tests for ActionRecordingWrapper
# =============================================================================


class TestActionRecordingWrapper:
    """Tests for the ActionRecordingWrapper class."""
    
    @pytest.fixture
    def env(self):
        """Create a mock robot environment."""
        return MockRobotEnv()
    
    def test_basic_recording(self, env):
        """Test basic action recording functionality."""
        wrapped_env = ActionRecordingWrapper(env)
        wrapped_env.reset()
        
        actions = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        ]
        
        for action in actions:
            wrapped_env.step(action)
            
        recorded = wrapped_env.get_recorded_data()
        
        assert len(recorded) == 3
        for i, record in enumerate(recorded):
            np.testing.assert_array_almost_equal(record["action"], actions[i])
            assert "reward" in record
            assert "terminated" in record
            assert "truncated" in record
            
    def test_recording_with_intervention_info(self, env):
        """Test that intervention info is recorded when present."""
        controlled_device = ControlledMockInputDevice(InputDeviceConfig(action_dim=7))
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        wrapped_env = ActionRecordingWrapper(hil_env)
        wrapped_env.reset()
        
        # Step with intervention
        human_action = np.ones(7)
        controlled_device.set_next_action(human_action, is_intervening=True)
        wrapped_env.step(np.zeros(7))
        
        # Step without intervention
        controlled_device.set_next_action(np.zeros(7), is_intervening=False)
        wrapped_env.step(np.zeros(7))
        
        recorded = wrapped_env.get_recorded_data()
        
        assert len(recorded) == 2
        assert recorded[0]["is_intervention"] is True
        np.testing.assert_array_almost_equal(recorded[0]["intervene_action"], human_action)
        assert recorded[1]["is_intervention"] is False
        assert "intervene_action" not in recorded[1]
        
        wrapped_env.close()
        
    def test_clear_recorded_data(self, env):
        """Test clearing recorded data."""
        wrapped_env = ActionRecordingWrapper(env)
        wrapped_env.reset()
        
        wrapped_env.step(np.zeros(7))
        wrapped_env.step(np.zeros(7))
        
        assert len(wrapped_env.get_recorded_data()) == 2
        
        wrapped_env.clear_recorded_data()
        
        assert len(wrapped_env.get_recorded_data()) == 0
        
    def test_max_steps_truncation(self, env):
        """Test that recorded data is truncated at max_steps."""
        wrapped_env = ActionRecordingWrapper(env, max_steps=5)
        wrapped_env.reset()
        
        # Record 10 steps
        for i in range(10):
            wrapped_env.step(np.ones(7) * i)
            
        recorded = wrapped_env.get_recorded_data()
        
        # Should only keep last 5 steps
        assert len(recorded) == 5
        # Check that it kept the most recent ones
        np.testing.assert_array_almost_equal(recorded[0]["action"], np.ones(7) * 5)
        np.testing.assert_array_almost_equal(recorded[-1]["action"], np.ones(7) * 9)
        
    def test_reset_does_not_clear_data(self, env):
        """Test that reset() does not clear recorded data."""
        wrapped_env = ActionRecordingWrapper(env)
        wrapped_env.reset()
        
        wrapped_env.step(np.zeros(7))
        wrapped_env.step(np.zeros(7))
        
        # Reset should not clear
        wrapped_env.reset()
        
        assert len(wrapped_env.get_recorded_data()) == 2
        
    def test_save_and_load_recorded_data(self, env):
        """Test saving and loading recorded data."""
        wrapped_env = ActionRecordingWrapper(env)
        wrapped_env.reset()
        
        actions = [np.random.randn(7) for _ in range(5)]
        for action in actions:
            wrapped_env.step(action)
            
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
            
        wrapped_env.save_recorded_data(filepath)
        
        # Clear and reload
        wrapped_env.clear_recorded_data()
        assert len(wrapped_env.get_recorded_data()) == 0
        
        wrapped_env.load_recorded_data(filepath)
        
        loaded = wrapped_env.get_recorded_data()
        assert len(loaded) == 5
        for i, record in enumerate(loaded):
            np.testing.assert_array_almost_equal(record["action"], actions[i])
            
        # Clean up
        import os
        os.unlink(filepath)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple wrappers."""
    
    def test_full_episode_with_mixed_intervention(self):
        """Test a full episode with mixed intervention patterns."""
        env = MockRobotEnv(max_episode_steps=10)
        controlled_device = ControlledMockInputDevice(InputDeviceConfig(action_dim=7))
        
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        recording_env = ActionRecordingWrapper(hil_env)
        
        recording_env.reset()
        
        # Simulate a mixed intervention pattern
        # Steps 0-2: No intervention
        # Steps 3-5: Human intervention
        # Steps 6-9: No intervention
        intervention_pattern = [False] * 3 + [True] * 3 + [False] * 4
        
        for i, should_intervene in enumerate(intervention_pattern):
            if should_intervene:
                human_action = np.ones(7) * 0.5
                controlled_device.set_next_action(human_action, is_intervening=True)
            else:
                controlled_device.set_next_action(np.zeros(7), is_intervening=False)
                
            policy_action = np.ones(7) * 0.1
            obs, reward, terminated, truncated, info = recording_env.step(policy_action)
            
        # Verify stats
        stats = hil_env.get_stats()
        assert stats["hil/intervention_steps"] == 3
        assert stats["hil/intervention_count"] == 1  # One continuous episode
        
        # Verify recording
        recorded = recording_env.get_recorded_data()
        assert len(recorded) == 10
        
        # Check intervention patterns in recording
        for i, record in enumerate(recorded):
            expected_intervention = i in [3, 4, 5]
            assert record["is_intervention"] == expected_intervention
            
        recording_env.close()
        
    def test_multiple_episodes(self):
        """Test intervention tracking across multiple episodes."""
        env = MockRobotEnv(max_episode_steps=5)
        controlled_device = ControlledMockInputDevice(InputDeviceConfig(action_dim=7))
        
        hil_env = HILInterventionWrapper(env, input_device=controlled_device)
        
        # Episode 1: 2 intervention steps
        hil_env.reset()
        controlled_device.set_action_sequence([
            (np.zeros(7), True),
            (np.zeros(7), True),
            (np.zeros(7), False),
            (np.zeros(7), False),
            (np.zeros(7), False),
        ])
        for _ in range(5):
            hil_env.step(np.zeros(7))
            
        # Episode 2: 3 intervention steps (2 separate episodes)
        hil_env.reset()
        controlled_device.set_action_sequence([
            (np.zeros(7), True),
            (np.zeros(7), False),
            (np.zeros(7), True),
            (np.zeros(7), True),
            (np.zeros(7), False),
        ])
        for _ in range(5):
            hil_env.step(np.zeros(7))
            
        stats = hil_env.get_stats()
        
        assert stats["hil/episode_count"] == 2
        assert stats["hil/total_steps"] == 10
        assert stats["hil/intervention_steps"] == 5
        assert stats["hil/intervention_count"] == 3  # 1 from ep1, 2 from ep2
        
        hil_env.close()
        
    def test_intervention_with_random_policy(self):
        """Test intervention wrapper with random policy actions."""
        np.random.seed(42)
        
        env = MockRobotEnv(max_episode_steps=50)
        
        # Use mock device with 20% random intervention probability
        hil_env = HILInterventionWrapper(
            env, 
            input_device="mock",
            input_device_config=InputDeviceConfig(action_dim=7),
            intervention_prob=0.2,
        )
        
        hil_env.reset()
        
        intervention_count = 0
        for _ in range(50):
            policy_action = np.random.randn(7) * 0.5
            obs, reward, terminated, truncated, info = hil_env.step(policy_action)
            
            if info["is_intervention"]:
                intervention_count += 1
                # Verify that executed action differs from policy action
                assert not np.allclose(info["executed_action"], info["policy_action"])
                
        # With 20% probability, we expect roughly 10 interventions
        # Allow some variance due to randomness
        assert intervention_count > 0, "Expected at least some interventions"
        
        hil_env.close()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_action_space(self):
        """Test with minimal action dimensions."""
        env = MockRobotEnv(action_dim=1)
        hil_env = HILInterventionWrapper(
            env, 
            input_device="mock",
            input_device_config=InputDeviceConfig(action_dim=1),
        )
        hil_env.reset()
        
        obs, reward, terminated, truncated, info = hil_env.step(np.array([0.5]))
        
        assert "is_intervention" in info
        hil_env.close()
        
    def test_high_dimensional_action_space(self):
        """Test with high-dimensional action space."""
        env = MockRobotEnv(action_dim=50, obs_dim=100)
        hil_env = HILInterventionWrapper(
            env,
            input_device="mock",
            input_device_config=InputDeviceConfig(action_dim=50),
        )
        hil_env.reset()
        
        obs, reward, terminated, truncated, info = hil_env.step(np.zeros(50))
        
        assert "is_intervention" in info
        hil_env.close()
        
    def test_blend_alpha_extremes(self):
        """Test blending with extreme alpha values."""
        env = MockRobotEnv()
        controlled_device = ControlledMockInputDevice(InputDeviceConfig(action_dim=7))
        
        # Alpha = 1.0 (full human action)
        hil_env = HILInterventionWrapper(
            env,
            input_device=controlled_device,
            blend_actions=True,
            blend_alpha=1.0,
        )
        hil_env.reset()
        
        human_action = np.ones(7)
        controlled_device.set_next_action(human_action, is_intervening=True)
        
        obs, reward, terminated, truncated, info = hil_env.step(np.zeros(7))
        np.testing.assert_array_almost_equal(info["executed_action"], human_action)
        
        hil_env.close()
        
        # Alpha = 0.0 (full policy action)
        hil_env = HILInterventionWrapper(
            env,
            input_device=controlled_device,
            blend_actions=True,
            blend_alpha=0.0,
        )
        hil_env.reset()
        
        controlled_device.set_next_action(human_action, is_intervening=True)
        policy_action = np.ones(7) * 0.5
        
        obs, reward, terminated, truncated, info = hil_env.step(policy_action)
        np.testing.assert_array_almost_equal(info["executed_action"], policy_action)
        
        hil_env.close()
        
    def test_stats_after_many_episodes(self):
        """Test that stats remain accurate after many episodes."""
        env = MockRobotEnv(max_episode_steps=3)
        hil_env = HILInterventionWrapper(env, input_device="mock")
        
        for episode in range(100):
            hil_env.reset()
            for _ in range(3):
                hil_env.step(np.zeros(7))
                
        stats = hil_env.get_stats()
        assert stats["hil/episode_count"] == 100
        assert stats["hil/total_steps"] == 300
        
        hil_env.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

