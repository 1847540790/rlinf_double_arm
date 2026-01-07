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

"""Integration tests for ViveTracker input device.

This module tests the ViveTracker wrapper functionality with or without
real hardware. Run with --mock for testing without trackers.

Run with:
    pytest examples/hil_vivetracker/test_vivetracker.py -v
    
Or run directly:
    python examples/hil_vivetracker/test_vivetracker.py --mock
    python examples/hil_vivetracker/test_vivetracker.py --all
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


# Initial poses used in teleop_demo_dual_flexiv_mock.py
INIT_ROBOT_POSE_LEFT = np.array([0.7, 0.1, 0, 0, 0, 1, 0])
INIT_ROBOT_POSE_RIGHT = np.array([0.7, -0.1, -0.02, 0, 0, 1, 0])
TEST_SERIALS = ["LHR-EFDCEC3F", "LHR-146E8F33"]


class TestViveTrackerConfigCreation(unittest.TestCase):
    """Tests for ViveTrackerConfig creation and file I/O."""
    
    def test_default_config_creation(self):
        """Test creating default ViveTrackerConfig."""
        config = ViveTrackerConfig()
        
        self.assertEqual(config.control_freq, 20.0)
        self.assertTrue(config.use_delta_actions)
        self.assertEqual(config.translation_scale, 1.0)
        self.assertEqual(config.rotation_scale, 1.0)
        self.assertEqual(config.deadzone, 0.005)
        
    def test_custom_config_creation(self):
        """Test creating custom ViveTrackerConfig with specific values."""
        config = ViveTrackerConfig(
            serials=["LHR-123", "LHR-456"],
            translation_scale=2.0,
            control_freq=30.0,
            init_robot_poses=[INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT],
        )
        
        self.assertEqual(config.serials, ["LHR-123", "LHR-456"])
        self.assertEqual(config.translation_scale, 2.0)
        self.assertEqual(config.control_freq, 30.0)
        self.assertEqual(len(config.init_robot_poses), 2)
        
    def test_create_default_config_file(self):
        """Test creating default config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            created_path = ViveTrackerConfig.create_default_config_file(config_path)
            
            self.assertTrue(os.path.exists(created_path))
            
            with open(created_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("vivetracker", data)
            self.assertIn("robot", data)
            
    def test_load_config_from_file(self):
        """Test loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            ViveTrackerConfig.create_default_config_file(config_path)
            
            loaded_config = ViveTrackerConfig.from_file(config_path)
            
            self.assertEqual(loaded_config.control_freq, 20.0)
            
    def test_save_config_to_file(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "saved_config.json")
            config = ViveTrackerConfig(
                serials=["LHR-AAA", "LHR-BBB"],
                translation_scale=1.5,
            )
            
            config.to_file(config_path)
            
            self.assertTrue(os.path.exists(config_path))
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data["vivetracker"]["translation_scale"], 1.5)


class TestViveTrackerDeviceLifecycle(unittest.TestCase):
    """Tests for ViveTrackerInputDevice lifecycle management."""
    
    def test_device_creation(self):
        """Test device creation."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        
        self.assertIsNotNone(device)
        self.assertFalse(device.is_running)
        
    def test_start_device_mock_mode(self):
        """Test starting device in mock mode (no hardware)."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        self.assertTrue(device.is_running)
        self.assertIsNone(device._tracker)  # Mock mode
        
        device.stop()
        
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
        
        device.stop()
        
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
        device.reset()
        
        self.assertEqual(device._gripper_states, [1.0, 1.0])
        
    def test_stop_device(self):
        """Test stopping device."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        self.assertTrue(device.is_running)
        
        device.stop()
        self.assertFalse(device.is_running)


class TestDualViveTrackerDevice(unittest.TestCase):
    """Tests for DualViveTrackerInputDevice (dual-arm setup)."""
    
    def test_dual_device_creation(self):
        """Test creating dual tracker device."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-LEFT", "LHR-RIGHT"],
                interactive_setup=False,
            )
        )
        
        self.assertEqual(device.config.action_dim, 14)
        
    def test_get_action_shape(self):
        """Test that get_action returns correct shape for dual arm."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-LEFT", "LHR-RIGHT"],
                interactive_setup=False,
            )
        )
        device.start()
        
        action, is_int = device.get_action()
        
        self.assertEqual(len(action), 14)
        
        device.stop()
        
    def test_get_left_action(self):
        """Test get_left_action returns first 7 dims."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        left_action, is_int = device.get_left_action()
        
        self.assertEqual(len(left_action), 7)
        
        device.stop()
        
    def test_get_right_action(self):
        """Test get_right_action returns last 7 dims."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        device.start()
        
        right_action, is_int = device.get_right_action()
        
        self.assertEqual(len(right_action), 7)
        
        device.stop()
        
    def test_gripper_control(self):
        """Test gripper set and toggle."""
        device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                interactive_setup=False,
            )
        )
        
        # Test set gripper
        device.set_gripper(0, -1.0)
        self.assertEqual(device._gripper_states[0], -1.0)
        
        # Test toggle gripper
        device.toggle_gripper(1)
        self.assertEqual(device._gripper_states[1], -1.0)
        
        device.toggle_gripper(1)
        self.assertEqual(device._gripper_states[1], 1.0)


class TestFromConfigFile(unittest.TestCase):
    """Tests for from_config_file factory method."""
    
    def test_create_with_new_config_file(self):
        """Test creating device with new config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "new_config.json")
            
            device = DualViveTrackerInputDevice.from_config_file(
                config_path=config_path,
                create_if_missing=True,
                interactive=False,
            )
            
            self.assertIsNotNone(device)
            self.assertTrue(os.path.exists(config_path))
            
    def test_load_from_existing_config_file(self):
        """Test loading device from existing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "existing_config.json")
            
            # Create config file first
            ViveTrackerConfig.create_default_config_file(config_path)
            
            device = DualViveTrackerInputDevice.from_config_file(
                config_path=config_path,
                create_if_missing=False,
                interactive=False,
            )
            
            self.assertIsNotNone(device)
            
    def test_file_not_found_no_create(self):
        """Test error when config file not found and create_if_missing=False."""
        with self.assertRaises(FileNotFoundError):
            ViveTrackerInputDevice.from_config_file(
                config_path="nonexistent.yaml",
                create_if_missing=False,
                interactive=False,
            )


class TestActionComputation(unittest.TestCase):
    """Tests for action computation methods."""
    
    def setUp(self):
        """Set up device for action computation tests."""
        self.device = DualViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(
                serials=["LHR-1", "LHR-2"],
                deadzone=0.001,
                interactive_setup=False,
            )
        )
        
    def test_delta_action_no_motion(self):
        """Test delta action computation with no motion."""
        poses = [
            np.array([0.7, 0.1, 0, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0, 0, 0, 0, 1]),
        ]
        
        # Establish baseline
        self.device._compute_delta_action(poses)
        
        # Same poses
        action, is_int = self.device._compute_delta_action(poses)
        
        self.assertFalse(is_int)
        np.testing.assert_array_almost_equal(action[:6], np.zeros(6), decimal=5)
        
    def test_delta_action_with_motion(self):
        """Test delta action computation with motion."""
        self.device._prev_tracker_poses = None  # Reset
        
        poses_initial = [
            np.array([0.7, 0.1, 0, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0, 0, 0, 0, 1]),
        ]
        poses_moved = [
            np.array([0.71, 0.11, 0.01, 0, 0, 0, 1]),
            np.array([0.71, -0.09, 0.01, 0, 0, 0, 1]),
        ]
        
        # Establish baseline
        self.device._compute_delta_action(poses_initial)
        
        # Motion
        action, is_int = self.device._compute_delta_action(poses_moved)
        
        self.assertTrue(is_int)
        self.assertAlmostEqual(action[0], 0.01, places=3)
        
    def test_absolute_action_computation(self):
        """Test absolute pose action computation."""
        poses = [
            np.array([0.7, 0.1, 0.2, 0, 0, 0, 1]),
            np.array([0.7, -0.1, 0.2, 0, 0, 0, 1]),
        ]
        
        action, is_int = self.device._compute_absolute_action(poses)
        
        self.assertEqual(len(action), 16)
        np.testing.assert_array_almost_equal(action[:7], poses[0], decimal=5)
        
    def test_quaternion_multiplication(self):
        """Test quaternion multiplication."""
        q1 = np.array([0, 0, 0, 1])  # Identity
        q2 = np.array([0.5, 0.5, 0.5, 0.5])  # Normalized
        
        result = self.device._quat_multiply(q2, q1)
        
        np.testing.assert_array_almost_equal(result, q2, decimal=5)


class TestDeviceRegistry(unittest.TestCase):
    """Tests for device registry."""
    
    def test_vivetracker_registered(self):
        """Test ViveTracker is registered."""
        self.assertIn("vivetracker", INPUT_DEVICE_REGISTRY)
        
        device = get_input_device("vivetracker")
        self.assertIsInstance(device, ViveTrackerInputDevice)
        
    def test_dual_vivetracker_registered(self):
        """Test DualViveTracker is registered."""
        self.assertIn("dual_vivetracker", INPUT_DEVICE_REGISTRY)
        
        device = get_input_device("dual_vivetracker")
        self.assertIsInstance(device, DualViveTrackerInputDevice)


class TestWithMockHTCViveTracker(unittest.TestCase):
    """Tests with mocked HTCViveTracker (simulating hardware)."""
    
    def setUp(self):
        """Set up mock HTCViveTracker."""
        self.mock_tracker = MagicMock()
        self.mock_tracker.get_aligned_robot_pose.return_value = [
            INIT_ROBOT_POSE_LEFT.copy(),
            INIT_ROBOT_POSE_RIGHT.copy(),
        ]
        
        self.mock_module = MagicMock()
        self.mock_module.HTCViveTracker.return_value = self.mock_tracker
        
    def test_start_with_mock_tracker(self):
        """Test start with mocked HTCViveTracker (as in teleop_demo)."""
        with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
            config = ViveTrackerConfig(
                serials=TEST_SERIALS,
                init_robot_poses=[INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT],
                interactive_setup=False,
            )
            
            device = ViveTrackerInputDevice(tracker_config=config)
            device.start()
            
            self.mock_module.HTCViveTracker.assert_called_once_with(
                serials=TEST_SERIALS,
                auto_calib_tracker=True,
            )
            
    def test_calibrate_from_file(self):
        """Test calibration from file (like teleop_demo set_calibration_from_file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_path = os.path.join(tmpdir, "calib.json")
            
            # Create calibration data
            calib_data = {
                "0": {
                    "calib_robot_pose": INIT_ROBOT_POSE_LEFT.tolist(),
                    "calib_tracker_pose": [0.7, 0.1, 0, 0, 0, 1, 0],
                },
                "1": {
                    "calib_robot_pose": INIT_ROBOT_POSE_RIGHT.tolist(),
                    "calib_tracker_pose": [0.7, -0.1, 0, 0, 0, 1, 0],
                },
            }
            with open(calib_path, 'w') as f:
                json.dump(calib_data, f)
            
            with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
                config = ViveTrackerConfig(
                    serials=TEST_SERIALS,
                    init_robot_poses=[INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT],
                    calib_path=calib_path,
                    interactive_setup=False,
                )
                
                device = ViveTrackerInputDevice(tracker_config=config)
                device._tracker = self.mock_tracker
                
                # Manually call _handle_calibration which loads from file
                device._handle_calibration()
                
                self.mock_tracker.set_calibration_from_file.assert_called_once()
                
    def test_get_action_with_mock_tracker(self):
        """Test get_action returns tracker data (simulating teleop loop)."""
        with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
            config = ViveTrackerConfig(
                serials=TEST_SERIALS,
                use_delta_actions=True,
                interactive_setup=False,
            )
            
            device = DualViveTrackerInputDevice(tracker_config=config)
            device._tracker = self.mock_tracker
            device._is_calibrated = True
            device._is_running = True
            
            # First call establishes baseline
            action1, _ = device.get_action()
            
            # Update mock to simulate tracker motion
            self.mock_tracker.get_aligned_robot_pose.return_value = [
                np.array([0.71, 0.11, 0.01, 0.0, 0.0, 0.0, 1.0]),
                np.array([0.71, -0.09, 0.01, 0.0, 0.0, 0.0, 1.0]),
            ]
            
            # Second call should detect motion
            action2, is_int2 = device.get_action()
            
            self.assertEqual(len(action2), 14)
            self.assertTrue(is_int2)
            
    def test_pause_start_tracker(self):
        """Test pause/start on tracker (as in teleop_demo main loop)."""
        with patch.dict('sys.modules', {'vive_tracker': self.mock_module}):
            config = ViveTrackerConfig(
                serials=TEST_SERIALS,
                interactive_setup=False,
            )
            
            device = DualViveTrackerInputDevice(tracker_config=config)
            device._tracker = self.mock_tracker
            device._is_running = True
            
            # Pause tracking
            device.pause()
            self.mock_tracker.pause.assert_called_once()
            
            # Resume tracking
            device.resume()
            self.mock_tracker.start.assert_called_once()


# Pytest-style tests for additional coverage
class TestPytestStyle:
    """Pytest-style tests."""
    
    def test_config_with_all_parameters(self):
        """Test config with all parameters set."""
        config = ViveTrackerConfig(
            serials=["LHR-A", "LHR-B"],
            init_robot_poses=[INIT_ROBOT_POSE_LEFT, INIT_ROBOT_POSE_RIGHT],
            calib_path="custom_calib.json",
            use_delta_actions=False,
            translation_scale=2.0,
            rotation_scale=1.5,
            deadzone=0.01,
            max_velocity=0.5,
            max_angular_velocity=1.0,
            control_freq=50.0,
            interactive_setup=False,
        )
        
        assert config.serials == ["LHR-A", "LHR-B"]
        assert config.translation_scale == 2.0
        assert config.rotation_scale == 1.5
        assert config.deadzone == 0.01
        assert config.control_freq == 50.0
        assert not config.use_delta_actions
        
    def test_device_get_action_not_running(self):
        """Test get_action when device not running returns zeros."""
        device = ViveTrackerInputDevice(
            tracker_config=ViveTrackerConfig(interactive_setup=False)
        )
        
        action, is_int = device.get_action()
        
        assert len(action) == device.config.action_dim
        assert not is_int
        np.testing.assert_array_equal(action, np.zeros(device.config.action_dim))
        
    @pytest.mark.parametrize("action_dim", [7, 14, 21])
    def test_custom_action_dim(self, action_dim):
        """Test device with custom action dimensions."""
        device = ViveTrackerInputDevice(
            config=InputDeviceConfig(action_dim=action_dim),
            tracker_config=ViveTrackerConfig(interactive_setup=False),
        )
        device.start()
        
        action, _ = device.get_action()
        
        assert len(action) == action_dim
        
        device.stop()


# Real-time tracking test (requires hardware, skip in CI)
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Requires ViveTracker hardware"
)
class TestRealtimeTracking:
    """Real-time tracking tests (require hardware)."""
    
    def test_realtime_tracking(self, duration: float = 5.0):
        """Test real-time tracking with actual hardware."""
        try:
            device = DualViveTrackerInputDevice.from_config_file(
                config_path="vive_robot_config.yaml",
                create_if_missing=True,
                interactive=False,
            )
            device.start()
            
            if device._tracker is None:
                pytest.skip("No tracker connected")
                
            print(f"\nTracking for {duration} seconds...")
            
            start_time = time.time()
            sample_count = 0
            while time.time() - start_time < duration:
                action, is_int = device.get_action()
                if is_int:
                    sample_count += 1
                time.sleep(0.05)
                
            device.stop()
            
            assert sample_count > 0, "No motion detected during test"
            
        except Exception as e:
            pytest.skip(f"Hardware test failed: {e}")


def main():
    """Run tests with optional command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ViveTracker Integration Tests")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no real hardware)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["config", "device", "dual", "file", "action", "registry", "mock_tracker", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  ViveTracker Integration Tests")
    print("=" * 60)
    print(f"  Mode: {'Mock' if args.mock else 'Real Hardware'}")
    print(f"  Test: {args.test}")
    print("=" * 60 + "\n")
    
    # Build test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_mapping = {
        "config": TestViveTrackerConfigCreation,
        "device": TestViveTrackerDeviceLifecycle,
        "dual": TestDualViveTrackerDevice,
        "file": TestFromConfigFile,
        "action": TestActionComputation,
        "registry": TestDeviceRegistry,
        "mock_tracker": TestWithMockHTCViveTracker,
    }
    
    if args.test == "all":
        for test_class in test_mapping.values():
            suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        suite.addTests(loader.loadTestsFromTestCase(test_mapping[args.test]))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ✓ All tests passed!\n")
        return 0
    else:
        print(f"\n  ✗ {failures} failure(s), {errors} error(s)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
