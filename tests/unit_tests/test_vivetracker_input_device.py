#!/usr/bin/env python3
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

"""Unit tests for ViveTracker input device.

These tests run without real ViveTracker hardware using mock mode.

Run with:
    pytest tests/unit_tests/test_vivetracker_input_device.py -v
    
Or run directly:
    python tests/unit_tests/test_vivetracker_input_device.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add rlinf to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rlinf.envs.hil.input_devices import (
    ViveTrackerConfig,
    ViveTrackerInputDevice,
    DualViveTrackerInputDevice,
    InputDeviceConfig,
    get_input_device,
    INPUT_DEVICE_REGISTRY,
)


class TestViveTrackerConfig(unittest.TestCase):
    """Tests for ViveTrackerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ViveTrackerConfig()
        
        self.assertEqual(config.serials, [])
        self.assertEqual(config.init_robot_poses, [])
        self.assertEqual(config.calib_path, "vive_calib.json")
        self.assertTrue(config.use_delta_actions)
        self.assertEqual(config.translation_scale, 1.0)
        self.assertEqual(config.rotation_scale, 1.0)
        self.assertEqual(config.deadzone, 0.005)
        self.assertEqual(config.max_velocity, 1.0)
        self.assertEqual(config.max_angular_velocity, 2.0)
        self.assertEqual(config.control_freq, 20.0)
        self.assertTrue(config.interactive_setup)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        serials = ["LHR-123", "LHR-456"]
        poses = [np.array([1, 2, 3, 0, 0, 0, 1]), np.array([4, 5, 6, 0, 0, 0, 1])]
        
        config = ViveTrackerConfig(
            serials=serials,
            init_robot_poses=poses,
            calib_path="custom_calib.json",
            use_delta_actions=False,
            translation_scale=2.0,
            control_freq=30.0,
        )
        
        self.assertEqual(config.serials, serials)
        self.assertEqual(len(config.init_robot_poses), 2)
        self.assertEqual(config.calib_path, "custom_calib.json")
        self.assertFalse(config.use_delta_actions)
        self.assertEqual(config.translation_scale, 2.0)
        self.assertEqual(config.control_freq, 30.0)
        
    def test_get_default_config_template(self):
        """Test default config template generation."""
        template = ViveTrackerConfig.get_default_config_template()
        
        self.assertIn("vivetracker", template)
        self.assertIn("robot", template)
        self.assertIn("serials", template["vivetracker"])
        self.assertIn("init_poses", template["robot"])
        
    def test_create_default_config_file_json(self):
        """Test creating default config file in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            
            created_path = ViveTrackerConfig.create_default_config_file(config_path)
            
            self.assertTrue(os.path.exists(created_path))
            self.assertTrue(created_path.endswith(".json"))
            
            # Verify content
            with open(created_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("vivetracker", data)
            self.assertIn("robot", data)
            
    def test_create_default_config_file_yaml(self):
        """Test creating default config file in YAML format."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
            
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")
            
            created_path = ViveTrackerConfig.create_default_config_file(config_path)
            
            self.assertTrue(os.path.exists(created_path))
            
            # Verify content
            with open(created_path, 'r') as f:
                data = yaml.safe_load(f)
            
            self.assertIn("vivetracker", data)
            self.assertIn("robot", data)
            
    def test_from_file_json(self):
        """Test loading config from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            
            # Create config file
            config_data = {
                "vivetracker": {
                    "serials": {"left": "LHR-AAA", "right": "LHR-BBB"},
                    "calibration_path": "test_calib.json",
                    "translation_scale": 1.5,
                    "control_freq": 25.0,
                },
                "robot": {
                    "init_poses": {
                        "left": [0.5, 0.1, 0, 0, 0, 1, 0],
                        "right": [0.5, -0.1, 0, 0, 0, 1, 0],
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            # Load config
            config = ViveTrackerConfig.from_file(config_path)
            
            self.assertEqual(config.serials, ["LHR-AAA", "LHR-BBB"])
            self.assertEqual(config.calib_path, "test_calib.json")
            self.assertEqual(config.translation_scale, 1.5)
            self.assertEqual(config.control_freq, 25.0)
            self.assertEqual(len(config.init_robot_poses), 2)
            
    def test_from_file_not_found(self):
        """Test loading config from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            ViveTrackerConfig.from_file("nonexistent_config.json")
            
    def test_to_file(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "saved_config.json")
            
            config = ViveTrackerConfig(
                serials=["LHR-111", "LHR-222"],
                init_robot_poses=[
                    np.array([1, 2, 3, 0, 0, 0, 1]),
                    np.array([4, 5, 6, 0, 0, 0, 1]),
                ],
                translation_scale=2.5,
            )
            
            config.to_file(config_path)
            
            self.assertTrue(os.path.exists(config_path))
            
            # Verify content
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            self.assertEqual(data["vivetracker"]["serials"]["left"], "LHR-111")
            self.assertEqual(data["vivetracker"]["serials"]["right"], "LHR-222")
            self.assertEqual(data["vivetracker"]["translation_scale"], 2.5)


class TestViveTrackerInputDevice(unittest.TestCase):
    """Tests for ViveTrackerInputDevice class."""
    
    def test_registration(self):
        """Test that ViveTrackerInputDevice is registered."""
        self.assertIn("vivetracker", INPUT_DEVICE_REGISTRY)
        
    def test_get_input_device(self):
        """Test getting device via registry."""
        device = get_input_device("vivetracker")
        self.assertIsInstance(device, ViveTrackerInputDevice)
        
    def test_default_initialization(self):
        """Test default initialization."""
        device = ViveTrackerInputDevice()
        
        self.assertIsNotNone(device.config)
        self.assertIsNotNone(device.tracker_config)
        self.assertFalse(device.is_running)
        self.assertFalse(device.is_calibrated)
        
    def test_custom_initialization(self):
        """Test initialization with custom config."""
        input_config = InputDeviceConfig(action_dim=14)
        tracker_config = ViveTrackerConfig(
            serials=["LHR-TEST"],
            control_freq=30.0,
        )
        
        device = ViveTrackerInputDevice(
            config=input_config,
            tracker_config=tracker_config,
        )
        
        self.assertEqual(device.config.action_dim, 14)
        self.assertEqual(device.tracker_config.control_freq, 30.0)
        
    def test_start_without_serials_mock_mode(self):
        """Test start with no serials falls back to mock mode."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        self.assertTrue(device.is_running)
        self.assertIsNone(device._tracker)  # Mock mode
        
    def test_get_action_mock_mode(self):
        """Test get_action in mock mode returns zeros."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        action, is_intervening = device.get_action()
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), device.config.action_dim)
        self.assertFalse(is_intervening)
        np.testing.assert_array_equal(action, np.zeros(device.config.action_dim))
        
    def test_get_action_paused(self):
        """Test get_action when paused returns zeros."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        device.pause()
        
        action, is_intervening = device.get_action()
        
        np.testing.assert_array_equal(action, np.zeros(device.config.action_dim))
        self.assertFalse(is_intervening)
        
    def test_pause_resume(self):
        """Test pause and resume functionality."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        self.assertFalse(device.is_paused)
        
        device.pause()
        self.assertTrue(device.is_paused)
        
        device.resume()
        self.assertFalse(device.is_paused)
        
    def test_reset(self):
        """Test reset clears state."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                interactive_setup=False,
            )
        )
        
        # Modify state
        device._gripper_states = [-1.0, -1.0]
        device._prev_tracker_poses = [np.zeros(7)]
        
        device.reset()
        
        self.assertEqual(device._gripper_states, [1.0, 1.0])
        self.assertIsNone(device._prev_tracker_poses)
        
    def test_stop(self):
        """Test stop functionality."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        self.assertTrue(device.is_running)
        
        device.stop()
        self.assertFalse(device.is_running)
        
    def test_set_gripper(self):
        """Test set_gripper method."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                interactive_setup=False,
            )
        )
        
        device.set_gripper(0, -1.0)
        self.assertEqual(device._gripper_states[0], -1.0)
        
        device.set_gripper(1, 0.5)
        self.assertEqual(device._gripper_states[1], 0.5)
        
    def test_toggle_gripper(self):
        """Test toggle_gripper method."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                interactive_setup=False,
            )
        )
        
        # Initially open (1.0)
        self.assertEqual(device._gripper_states[0], 1.0)
        
        device.toggle_gripper(0)
        self.assertEqual(device._gripper_states[0], -1.0)
        
        device.toggle_gripper(0)
        self.assertEqual(device._gripper_states[0], 1.0)
        
    def test_from_config_file_creates_default(self):
        """Test from_config_file creates default config when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "new_config.yaml")
            
            # Should create default config
            device = ViveTrackerInputDevice.from_config_file(
                config_path=config_path,
                create_if_missing=True,
                interactive=False,  # Non-interactive for testing
            )
            
            # Config file should be created
            self.assertTrue(os.path.exists(config_path) or 
                          os.path.exists(config_path.replace('.yaml', '.json')))
            self.assertIsNotNone(device)
            
    def test_from_config_file_not_found_no_create(self):
        """Test from_config_file raises error when create_if_missing=False."""
        with self.assertRaises(FileNotFoundError):
            ViveTrackerInputDevice.from_config_file(
                config_path="nonexistent.yaml",
                create_if_missing=False,
                interactive=False,
            )
            
    def test_quat_multiply(self):
        """Test quaternion multiplication."""
        device = ViveTrackerInputDevice()
        
        # Test identity: q * identity = q
        q = np.array([0.5, 0.5, 0.5, 0.5])  # Normalized quaternion
        identity = np.array([0, 0, 0, 1])
        
        result = device._quat_multiply(q, identity)
        np.testing.assert_array_almost_equal(result, q, decimal=5)
        
    def test_quat_diff_to_axis_angle_identity(self):
        """Test quaternion diff for identical quaternions."""
        device = ViveTrackerInputDevice()
        
        q = np.array([0, 0, 0, 1])
        
        result = device._quat_diff_to_axis_angle(q, q)
        np.testing.assert_array_almost_equal(result, np.zeros(3), decimal=5)


class TestDualViveTrackerInputDevice(unittest.TestCase):
    """Tests for DualViveTrackerInputDevice class."""
    
    def test_registration(self):
        """Test that DualViveTrackerInputDevice is registered."""
        self.assertIn("dual_vivetracker", INPUT_DEVICE_REGISTRY)
        
    def test_get_input_device(self):
        """Test getting device via registry."""
        device = get_input_device("dual_vivetracker")
        self.assertIsInstance(device, DualViveTrackerInputDevice)
        
    def test_default_action_dim(self):
        """Test that default action_dim is 14 for dual arms."""
        device = DualViveTrackerInputDevice()
        self.assertEqual(device.config.action_dim, 14)
        
    def test_action_dim_override(self):
        """Test that action_dim is forced to >= 14."""
        input_config = InputDeviceConfig(action_dim=7)  # Too small
        device = DualViveTrackerInputDevice(config=input_config)
        
        self.assertEqual(device.config.action_dim, 14)
        
    def test_get_action_shape(self):
        """Test get_action returns correct shape for dual arm."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        action, is_intervening = device.get_action()
        
        self.assertEqual(len(action), 14)
        
    def test_get_left_action(self):
        """Test get_left_action returns first 7 dims."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        action, is_intervening = device.get_left_action()
        
        self.assertEqual(len(action), 7)
        
    def test_get_right_action(self):
        """Test get_right_action returns last 7 dims."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        action, is_intervening = device.get_right_action()
        
        self.assertEqual(len(action), 7)
        
    def test_warning_for_wrong_serial_count(self):
        """Test warning is printed for non-dual serial count."""
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            device = DualViveTrackerInputDevice(
                tracker_config=ViveTrackerConfig(
                    serials=["LHR-ONLY-ONE"],
                    interactive_setup=False,
                )
            )
        finally:
            sys.stdout = sys.__stdout__
            
        output = captured_output.getvalue()
        self.assertIn("Warning", output)
        self.assertIn("Expected 2 tracker serials", output)
        
    def test_from_config_file(self):
        """Test from_config_file creates DualViveTrackerInputDevice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "dual_config.json")
            
            # Create config
            ViveTrackerConfig.create_default_config_file(config_path)
            
            device = DualViveTrackerInputDevice.from_config_file(
                config_path=config_path,
                interactive=False,
            )
            
            self.assertIsInstance(device, DualViveTrackerInputDevice)
            self.assertEqual(device.config.action_dim, 14)


class TestViveTrackerWithMockTracker(unittest.TestCase):
    """Tests with mocked vive_tracker module."""
    
    def setUp(self):
        """Set up mock vive_tracker module."""
        self.mock_tracker = MagicMock()
        self.mock_tracker.get_aligned_robot_pose.return_value = [
            np.array([0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([0.7, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ]
        
        # Create mock module
        self.mock_module = MagicMock()
        self.mock_module.HTCViveTracker.return_value = self.mock_tracker
        
    def test_start_with_mock_tracker(self):
        """Test start with mocked vive_tracker module."""
        with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
            config = ViveTrackerConfig(
                serials=["LHR-LEFT", "LHR-RIGHT"],
                init_robot_poses=[
                    np.array([0.7, 0.1, 0, 0, 0, 1, 0]),
                    np.array([0.7, -0.1, 0, 0, 0, 1, 0]),
                ],
                interactive_setup=False,
            )
            
            device = ViveTrackerInputDevice(tracker_config=config)
            device.start()
            
            self.mock_module.HTCViveTracker.assert_called_once_with(
                serials=["LHR-LEFT", "LHR-RIGHT"],
                auto_calib_tracker=True,
            )
            
    def test_calibrate_with_mock(self):
        """Test calibration with mocked tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_path = os.path.join(tmpdir, "calib.json")
            
            with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
                config = ViveTrackerConfig(
                    serials=["LHR-LEFT", "LHR-RIGHT"],
                    init_robot_poses=[
                        np.array([0.7, 0.1, 0, 0, 0, 1, 0]),
                        np.array([0.7, -0.1, 0, 0, 0, 1, 0]),
                    ],
                    calib_path=calib_path,
                    interactive_setup=False,
                )
                
                device = ViveTrackerInputDevice(tracker_config=config)
                device._tracker = self.mock_tracker
                
                result = device.calibrate(save=True)
                
                self.assertTrue(result)
                self.mock_tracker.calibrate_robot_tracker_transform.assert_called_once()
                
    def test_get_action_with_mock_tracker(self):
        """Test get_action with mocked tracker."""
        with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
            config = ViveTrackerConfig(
                serials=["LHR-LEFT", "LHR-RIGHT"],
                use_delta_actions=True,
                interactive_setup=False,
            )
            
            device = DualViveTrackerInputDevice(
                tracker_config=config,
            )
            device._tracker = self.mock_tracker
            device._is_calibrated = True
            device._is_running = True
            
            # First call - establishes baseline
            action1, is_int1 = device.get_action()
            
            # Update mock to return different pose
            self.mock_tracker.get_aligned_robot_pose.return_value = [
                np.array([0.71, 0.11, 0.01, 0.0, 0.0, 0.0, 1.0]),  # Small delta
                np.array([0.71, -0.09, 0.01, 0.0, 0.0, 0.0, 1.0]),
            ]
            
            # Second call - should detect intervention due to motion
            action2, is_int2 = device.get_action()
            
            self.assertEqual(len(action2), 14)
            # Should detect motion as intervention
            self.assertTrue(is_int2)


class TestComputeDeltaAction(unittest.TestCase):
    """Tests for delta action computation."""
    
    def test_compute_delta_no_motion(self):
        """Test delta computation with no motion."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                interactive_setup=False,
            )
        )
        
        poses = [
            np.array([0.7, 0.1, 0, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0, 0, 0, 0, 1]),
        ]
        
        # First call establishes baseline
        device._compute_delta_action(poses)
        
        # Second call with same poses
        action, is_intervening = device._compute_delta_action(poses)
        
        self.assertEqual(len(action), 14)
        # No motion = no intervention
        self.assertFalse(is_intervening)
        # Deltas should be near zero
        np.testing.assert_array_almost_equal(action[:6], np.zeros(6), decimal=5)
        
    def test_compute_delta_with_motion(self):
        """Test delta computation with motion."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                deadzone=0.001,  # Very small deadzone
                interactive_setup=False,
            )
        )
        
        poses_initial = [
            np.array([0.7, 0.1, 0, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0, 0, 0, 0, 1]),
        ]
        
        poses_moved = [
            np.array([0.71, 0.11, 0.01, 0, 0, 0, 1]),  # 1cm motion
            np.array([0.71, -0.09, 0.01, 0, 0, 0, 1]),
        ]
        
        # Establish baseline
        device._compute_delta_action(poses_initial)
        
        # Motion
        action, is_intervening = device._compute_delta_action(poses_moved)
        
        self.assertTrue(is_intervening)
        # Left arm delta
        self.assertAlmostEqual(action[0], 0.01, places=4)  # dx
        self.assertAlmostEqual(action[1], 0.01, places=4)  # dy
        self.assertAlmostEqual(action[2], 0.01, places=4)  # dz


class TestComputeAbsoluteAction(unittest.TestCase):
    """Tests for absolute action computation."""
    
    def test_compute_absolute_action(self):
        """Test absolute pose action computation."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                use_delta_actions=False,
                interactive_setup=False,
            )
        )
        
        poses = [
            np.array([0.7, 0.1, 0.2, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0.2, 0, 0, 0, 1]),
        ]
        
        action, is_intervening = device._compute_absolute_action(poses)
        
        # 8 values per arm (pose + gripper)
        self.assertEqual(len(action), 16)
        
        # Check left arm pose
        np.testing.assert_array_almost_equal(action[:7], poses[0], decimal=5)
        # Check right arm pose
        np.testing.assert_array_almost_equal(action[8:15], poses[1], decimal=5)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

