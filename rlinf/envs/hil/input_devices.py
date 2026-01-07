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

"""Input device interfaces for Human-in-the-Loop control.

This module provides a base interface for human input devices (spacemouse, 
vivetracker, keyboard, etc.) and implementations for common devices.

To add a new input device (e.g., ViveTracker):
    1. Create a new class inheriting from BaseInputDevice
    2. Implement get_action() to return (action, is_intervening)
    3. Register it in the INPUT_DEVICE_REGISTRY
"""

import abc
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import numpy as np


# Registry for input devices
INPUT_DEVICE_REGISTRY: Dict[str, Type["BaseInputDevice"]] = {}


def register_input_device(name: str):
    """Decorator to register an input device class."""
    def decorator(cls):
        INPUT_DEVICE_REGISTRY[name] = cls
        return cls
    return decorator


def get_input_device(name: str, **kwargs) -> "BaseInputDevice":
    """Get an input device by name."""
    if name not in INPUT_DEVICE_REGISTRY:
        raise ValueError(
            f"Unknown input device: {name}. "
            f"Available devices: {list(INPUT_DEVICE_REGISTRY.keys())}"
        )
    return INPUT_DEVICE_REGISTRY[name](**kwargs)


@dataclass
class InputDeviceConfig:
    """Configuration for input devices."""
    action_dim: int = 7  # xyz + rpy + gripper
    action_scale: np.ndarray = None  # Scale factors for each action dimension
    deadzone: float = 0.01  # Minimum action magnitude to trigger intervention
    
    def __post_init__(self):
        if self.action_scale is None:
            self.action_scale = np.ones(self.action_dim)


class BaseInputDevice(abc.ABC):
    """Base class for human input devices.
    
    All input devices must implement get_action() which returns:
        - action: numpy array of shape (action_dim,)
        - is_intervening: bool indicating if human is actively controlling
        
    The intervention detection can be based on:
        - Button press (spacemouse, gamepad)
        - Pedal press (sigma7, g29)
        - Motion threshold (vivetracker)
        - Any custom logic
    """
    
    def __init__(self, config: Optional[InputDeviceConfig] = None):
        self.config = config or InputDeviceConfig()
        self._is_running = False
        
    @abc.abstractmethod
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Get the current action from the input device.
        
        Returns:
            Tuple of (action, is_intervening):
                - action: numpy array of shape (action_dim,)
                - is_intervening: True if human is actively controlling
        """
        pass
    
    def start(self):
        """Start the input device (if needed for async reading)."""
        self._is_running = True
        
    def stop(self):
        """Stop the input device."""
        self._is_running = False
        
    def reset(self):
        """Reset the input device state."""
        pass
    
    @property
    def is_running(self) -> bool:
        return self._is_running


@register_input_device("mock")
class MockInputDevice(BaseInputDevice):
    """Mock input device for testing without real hardware.
    
    Simulates human interventions based on:
        - Random interventions with configurable probability
        - Keyboard input (if enabled)
        - Scripted intervention patterns
    """
    
    def __init__(
        self,
        config: Optional[InputDeviceConfig] = None,
        intervention_prob: float = 0.0,
        random_action_scale: float = 0.5,
    ):
        super().__init__(config)
        self.intervention_prob = intervention_prob
        self.random_action_scale = random_action_scale
        self._force_intervene = False
        self._intervention_action = None
        
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Get mock action - randomly intervenes with configurable probability."""
        # Check if forced intervention is active
        if self._force_intervene and self._intervention_action is not None:
            return self._intervention_action, True
            
        # Random intervention
        if np.random.random() < self.intervention_prob:
            action = np.random.uniform(
                -self.random_action_scale, 
                self.random_action_scale, 
                size=(self.config.action_dim,)
            )
            return action, True
            
        # No intervention
        return np.zeros(self.config.action_dim), False
    
    def set_intervention(self, action: np.ndarray):
        """Force an intervention with a specific action."""
        self._force_intervene = True
        self._intervention_action = action
        
    def clear_intervention(self):
        """Clear forced intervention."""
        self._force_intervene = False
        self._intervention_action = None


@register_input_device("keyboard")
class KeyboardInputDevice(BaseInputDevice):
    """Keyboard-based input device for testing.
    
    Uses pynput to capture keyboard input for robot control.
    Useful for quick testing without specialized hardware.
    
    Key mappings:
        - W/S: Forward/Backward (X)
        - A/D: Left/Right (Y)  
        - Q/E: Up/Down (Z)
        - I/K: Pitch
        - J/L: Yaw
        - U/O: Roll
        - Space: Toggle gripper
        - CAPS LOCK: Toggle intervention mode ON/OFF
    """
    
    def __init__(
        self,
        config: Optional[InputDeviceConfig] = None,
        translation_scale: float = 0.02,
        rotation_scale: float = 0.05,
    ):
        super().__init__(config)
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        
        # Key states
        self._key_states = {
            'w': False, 's': False,  # X axis
            'a': False, 'd': False,  # Y axis
            'q': False, 'e': False,  # Z axis
            'i': False, 'k': False,  # Pitch
            'j': False, 'l': False,  # Yaw
            'u': False, 'o': False,  # Roll
            'space': False,          # Gripper toggle
        }
        self._intervention_enabled = False  # CAPS LOCK toggle state
        self._gripper_state = 1.0  # Open
        
        self._listener = None
        self._lock = threading.Lock()
        
    def start(self):
        """Start keyboard listener."""
        super().start()
        try:
            from pynput import keyboard
            
            def on_press(key):
                with self._lock:
                    try:
                        k = key.char.lower() if hasattr(key, 'char') and key.char else None
                        if k in self._key_states:
                            self._key_states[k] = True
                    except AttributeError:
                        pass
                    
                    # Handle special keys
                    if key == keyboard.Key.caps_lock:
                        # Toggle intervention mode
                        self._intervention_enabled = not self._intervention_enabled
                        status = "ON" if self._intervention_enabled else "OFF"
                        print(f"[Keyboard] Intervention {status}")
                    elif key == keyboard.Key.space:
                        self._key_states['space'] = True
                        self._gripper_state *= -1  # Toggle
                            
            def on_release(key):
                with self._lock:
                    try:
                        k = key.char.lower() if hasattr(key, 'char') and key.char else None
                        if k in self._key_states:
                            self._key_states[k] = False
                    except AttributeError:
                        pass
                    
                    if key == keyboard.Key.space:
                        self._key_states['space'] = False
            
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
            print("[KeyboardInputDevice] Started.")
            print("[KeyboardInputDevice] Press CAPS LOCK to toggle intervention ON/OFF")
            print("[KeyboardInputDevice] Use W/A/S/D/Q/E for movement when intervention is ON")
            
        except ImportError:
            print("[KeyboardInputDevice] pynput not installed. Keyboard control disabled.")
            
    def stop(self):
        """Stop keyboard listener."""
        super().stop()
        if self._listener:
            self._listener.stop()
            self._listener = None
            
    def reset(self):
        """Reset intervention state."""
        with self._lock:
            self._intervention_enabled = False
            for k in self._key_states:
                self._key_states[k] = False
            
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Get action from keyboard state."""
        with self._lock:
            is_intervening = self._intervention_enabled
            
            if not is_intervening:
                return np.zeros(self.config.action_dim), False
                
            # Build action from key states
            action = np.zeros(self.config.action_dim)
            
            # Translation (XYZ)
            action[0] = (self._key_states['w'] - self._key_states['s']) * self.translation_scale
            action[1] = (self._key_states['a'] - self._key_states['d']) * self.translation_scale
            action[2] = (self._key_states['q'] - self._key_states['e']) * self.translation_scale
            
            # Rotation (RPY)
            action[3] = (self._key_states['u'] - self._key_states['o']) * self.rotation_scale  # Roll
            action[4] = (self._key_states['i'] - self._key_states['k']) * self.rotation_scale  # Pitch
            action[5] = (self._key_states['j'] - self._key_states['l']) * self.rotation_scale  # Yaw
            
            # Gripper
            if self.config.action_dim > 6:
                action[6] = self._gripper_state
                
            return action, True
    
    @property
    def intervention_enabled(self) -> bool:
        """Check if intervention is currently enabled."""
        with self._lock:
            return self._intervention_enabled


@dataclass
class ViveTrackerConfig:
    """Configuration for ViveTracker input device.
    
    Attributes:
        serials: List of tracker serial numbers (e.g., ["LHR-37267EB0", "LHR-ED74041F"])
        init_robot_poses: Initial robot TCP poses for each tracker [x, y, z, qx, qy, qz, qw]
        calib_path: Path to calibration file
        robot_config_path: Path to robot configuration file (optional)
        auto_calib_tracker: Whether to auto-calibrate tracker on startup
        use_delta_actions: If True, return delta actions; if False, return absolute poses
        translation_scale: Scale factor for translation actions
        rotation_scale: Scale factor for rotation actions
        deadzone: Minimum motion threshold to consider as intervention
        max_velocity: Maximum velocity limit (m/s)
        max_angular_velocity: Maximum angular velocity limit (rad/s)
        intervention_button: Button index for explicit intervention toggle (-1 for motion-based)
        control_freq: Control frequency in Hz
        interactive_setup: If True, prompt user for missing config; if False, use defaults
    """
    serials: list = field(default_factory=list)
    init_robot_poses: list = field(default_factory=list)
    calib_path: str = "vive_calib.json"
    robot_config_path: str = "vive_robot_config.yaml"
    auto_calib_tracker: bool = True
    use_delta_actions: bool = True
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    deadzone: float = 0.005  # 5mm motion threshold
    max_velocity: float = 1.0
    max_angular_velocity: float = 2.0
    intervention_button: int = -1  # -1 means motion-based intervention
    control_freq: float = 20.0
    interactive_setup: bool = True  # Prompt user for missing config
    
    # Default robot poses for dual Flexiv arms
    DEFAULT_INIT_POSE_LEFT = [0.7, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0]
    DEFAULT_INIT_POSE_RIGHT = [0.7, -0.1, -0.02, 0.0, 0.0, 1.0, 0.0]
    DEFAULT_SERIAL_LEFT = "LHR-XXXXXXXX"
    DEFAULT_SERIAL_RIGHT = "LHR-YYYYYYYY"
    
    @classmethod
    def get_default_config_template(cls) -> dict:
        """Get default configuration template as a dictionary."""
        return {
            "vivetracker": {
                "serials": {
                    "left": cls.DEFAULT_SERIAL_LEFT,
                    "right": cls.DEFAULT_SERIAL_RIGHT,
                },
                "calibration_path": "vive_calib.json",
                "auto_calib_tracker": True,
                "use_delta_actions": True,
                "translation_scale": 1.0,
                "rotation_scale": 1.0,
                "deadzone": 0.005,
                "max_velocity": 1.0,
                "max_angular_velocity": 2.0,
                "control_freq": 20.0,
            },
            "robot": {
                "init_poses": {
                    "left": cls.DEFAULT_INIT_POSE_LEFT,
                    "right": cls.DEFAULT_INIT_POSE_RIGHT,
                },
                "motion_limits": {
                    "max_vel": 1.0,
                    "max_acc": 3.0,
                    "max_angular_vel": 2.0,
                    "max_angular_acc": 8.0,
                },
            },
        }
    
    @classmethod
    def create_default_config_file(cls, filepath: str) -> str:
        """Create a default configuration file.
        
        Args:
            filepath: Path to save the config file (.yaml or .json)
            
        Returns:
            Path to the created file
        """
        import os
        config = cls.get_default_config_template()
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback to JSON if yaml not available
                filepath = filepath.replace('.yaml', '.json').replace('.yml', '.json')
                import json
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
        else:
            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
                
        return filepath
    
    @classmethod
    def from_file(cls, filepath: str) -> "ViveTrackerConfig":
        """Load configuration from a YAML or JSON file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            ViveTrackerConfig instance
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
            
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required to load .yaml files. Install with: pip install pyyaml")
        else:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                
        vt_cfg = data.get("vivetracker", {})
        robot_cfg = data.get("robot", {})
        
        # Extract serials
        serials_data = vt_cfg.get("serials", {})
        if isinstance(serials_data, dict):
            serials = [serials_data.get("left", ""), serials_data.get("right", "")]
            serials = [s for s in serials if s]  # Remove empty
        elif isinstance(serials_data, list):
            serials = serials_data
        else:
            serials = []
            
        # Extract init poses
        poses_data = robot_cfg.get("init_poses", {})
        if isinstance(poses_data, dict):
            init_poses = []
            if "left" in poses_data:
                init_poses.append(np.array(poses_data["left"]))
            if "right" in poses_data:
                init_poses.append(np.array(poses_data["right"]))
        elif isinstance(poses_data, list):
            init_poses = [np.array(p) for p in poses_data]
        else:
            init_poses = []
            
        return cls(
            serials=serials,
            init_robot_poses=init_poses,
            calib_path=vt_cfg.get("calibration_path", "vive_calib.json"),
            robot_config_path=filepath,
            auto_calib_tracker=vt_cfg.get("auto_calib_tracker", True),
            use_delta_actions=vt_cfg.get("use_delta_actions", True),
            translation_scale=vt_cfg.get("translation_scale", 1.0),
            rotation_scale=vt_cfg.get("rotation_scale", 1.0),
            deadzone=vt_cfg.get("deadzone", 0.005),
            max_velocity=vt_cfg.get("max_velocity", 1.0),
            max_angular_velocity=vt_cfg.get("max_angular_velocity", 2.0),
            control_freq=vt_cfg.get("control_freq", 20.0),
        )
    
    def to_file(self, filepath: str):
        """Save configuration to a file.
        
        Args:
            filepath: Path to save config file (.yaml or .json)
        """
        import os
        
        config = {
            "vivetracker": {
                "serials": {
                    "left": self.serials[0] if len(self.serials) > 0 else "",
                    "right": self.serials[1] if len(self.serials) > 1 else "",
                },
                "calibration_path": self.calib_path,
                "auto_calib_tracker": self.auto_calib_tracker,
                "use_delta_actions": self.use_delta_actions,
                "translation_scale": self.translation_scale,
                "rotation_scale": self.rotation_scale,
                "deadzone": self.deadzone,
                "max_velocity": self.max_velocity,
                "max_angular_velocity": self.max_angular_velocity,
                "control_freq": self.control_freq,
            },
            "robot": {
                "init_poses": {
                    "left": self.init_robot_poses[0].tolist() if len(self.init_robot_poses) > 0 else self.DEFAULT_INIT_POSE_LEFT,
                    "right": self.init_robot_poses[1].tolist() if len(self.init_robot_poses) > 1 else self.DEFAULT_INIT_POSE_RIGHT,
                },
            },
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                filepath = filepath.replace('.yaml', '.json').replace('.yml', '.json')
                import json
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
        else:
            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)


@register_input_device("vivetracker")
class ViveTrackerInputDevice(BaseInputDevice):
    """ViveTracker input device for teleoperation.
    
    Supports single or dual tracker setups for robot teleoperation.
    Uses the vive_tracker module (HTCViveTracker) for pose tracking.
    
    Features:
        - 6DoF pose tracking from HTC Vive Tracker
        - Automatic coordinate calibration between tracker and robot frames
        - Delta action or absolute pose output modes
        - Motion-based or button-based intervention detection
        - Pause/resume functionality
        - Automatic config file creation with defaults if missing
        - Interactive setup prompts for missing configuration
        
    Example usage for single arm:
        config = ViveTrackerConfig(
            serials=["LHR-37267EB0"],
            init_robot_poses=[np.array([0.7, 0.1, 0, 0, 0, 1, 0])],
            calib_path="calib.json",
        )
        device = ViveTrackerInputDevice(config=input_config, tracker_config=config)
        device.start()
        device.calibrate()  # First time setup
        
        action, is_intervening = device.get_action()
        
    Example with config file:
        # Creates default config if not present
        device = ViveTrackerInputDevice.from_config_file("vive_config.yaml")
        device.start()
    """
    
    def __init__(
        self,
        config: Optional[InputDeviceConfig] = None,
        tracker_config: Optional[ViveTrackerConfig] = None,
        **kwargs,
    ):
        super().__init__(config)
        self.tracker_config = tracker_config or ViveTrackerConfig()
        
        # Tracker instance
        self._tracker = None
        self._lock = threading.Lock()
        
        # State tracking
        self._is_paused = False
        self._is_calibrated = False
        self._last_poses = None
        self._intervention_enabled = False
        
        # For delta action computation
        self._prev_tracker_poses = None
        
        # Gripper state (can be toggled via button)
        self._gripper_states = [1.0] * max(1, len(self.tracker_config.serials))  # Open
        
    @classmethod
    def from_config_file(
        cls,
        config_path: str = "vive_robot_config.yaml",
        input_config: Optional[InputDeviceConfig] = None,
        create_if_missing: bool = True,
        interactive: bool = True,
    ) -> "ViveTrackerInputDevice":
        """Create ViveTrackerInputDevice from a config file.
        
        If the config file doesn't exist and create_if_missing=True,
        creates a default config file and optionally prompts user to edit it.
        
        Args:
            config_path: Path to config file (.yaml or .json)
            input_config: Optional InputDeviceConfig
            create_if_missing: Create default config if file not found
            interactive: Prompt user for input when config is missing
            
        Returns:
            ViveTrackerInputDevice instance
        """
        import os
        
        if not os.path.exists(config_path):
            if create_if_missing:
                print(f"[ViveTrackerInputDevice] Config file not found: {config_path}")
                print("[ViveTrackerInputDevice] Creating default configuration file...")
                
                created_path = ViveTrackerConfig.create_default_config_file(config_path)
                print(f"[ViveTrackerInputDevice] Created: {created_path}")
                print()
                print("=" * 60)
                print("DEFAULT CONFIGURATION FILE CREATED")
                print("=" * 60)
                print(f"Please edit {created_path} to configure:")
                print("  1. Tracker serial numbers (find in SteamVR)")
                print("  2. Robot initial poses")
                print("  3. Motion parameters")
                print("=" * 60)
                
                if interactive:
                    response = input("\nWould you like to edit the config now? [y/N]: ").strip().lower()
                    if response == 'y':
                        cls._interactive_config_setup(created_path)
                    else:
                        print(f"\nUsing defaults. Edit {created_path} manually and restart.")
                        
                config_path = created_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        tracker_config = ViveTrackerConfig.from_file(config_path)
        tracker_config.interactive_setup = interactive
        
        return cls(config=input_config, tracker_config=tracker_config)
    
    @staticmethod
    def _interactive_config_setup(config_path: str):
        """Interactive configuration setup via terminal prompts."""
        print("\n" + "=" * 60)
        print("INTERACTIVE CONFIGURATION SETUP")
        print("=" * 60)
        
        # Load current config
        try:
            tracker_config = ViveTrackerConfig.from_file(config_path)
        except Exception:
            tracker_config = ViveTrackerConfig()
            
        # Prompt for tracker serials
        print("\n[Step 1/3] Tracker Serial Numbers")
        print("Find these in SteamVR settings. Format: LHR-XXXXXXXX")
        
        left_serial = input(f"  Left tracker serial [{tracker_config.DEFAULT_SERIAL_LEFT}]: ").strip()
        if not left_serial:
            left_serial = tracker_config.DEFAULT_SERIAL_LEFT
            
        right_serial = input(f"  Right tracker serial [{tracker_config.DEFAULT_SERIAL_RIGHT}]: ").strip()
        if not right_serial:
            right_serial = tracker_config.DEFAULT_SERIAL_RIGHT
            
        tracker_config.serials = [left_serial, right_serial]
        
        # Prompt for robot poses
        print("\n[Step 2/3] Robot Initial Poses")
        print("Format: x, y, z, qx, qy, qz, qw (7 values)")
        
        def parse_pose(prompt: str, default: list) -> np.ndarray:
            default_str = ", ".join(f"{v:.2f}" for v in default)
            user_input = input(f"  {prompt} [{default_str}]: ").strip()
            if not user_input:
                return np.array(default)
            try:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) == 7:
                    return np.array(values)
                else:
                    print(f"    Expected 7 values, got {len(values)}. Using default.")
                    return np.array(default)
            except ValueError:
                print("    Invalid input. Using default.")
                return np.array(default)
                
        pose_left = parse_pose("Left arm initial pose", tracker_config.DEFAULT_INIT_POSE_LEFT)
        pose_right = parse_pose("Right arm initial pose", tracker_config.DEFAULT_INIT_POSE_RIGHT)
        tracker_config.init_robot_poses = [pose_left, pose_right]
        
        # Motion parameters
        print("\n[Step 3/3] Motion Parameters (press Enter for defaults)")
        
        def parse_float(prompt: str, default: float) -> float:
            user_input = input(f"  {prompt} [{default}]: ").strip()
            if not user_input:
                return default
            try:
                return float(user_input)
            except ValueError:
                print("    Invalid input. Using default.")
                return default
                
        tracker_config.max_velocity = parse_float("Max velocity (m/s)", tracker_config.max_velocity)
        tracker_config.max_angular_velocity = parse_float("Max angular velocity (rad/s)", tracker_config.max_angular_velocity)
        tracker_config.control_freq = parse_float("Control frequency (Hz)", tracker_config.control_freq)
        
        # Save config
        tracker_config.to_file(config_path)
        print(f"\nConfiguration saved to {config_path}")
        print("=" * 60)
        
    def start(self):
        """Start ViveTracker connection and initialize tracking.
        
        Handles missing calibration and config files with prompts or defaults.
        """
        super().start()
        
        import os
        
        # Check and handle missing robot config
        if not self.tracker_config.serials or all(s.startswith("LHR-X") or s.startswith("LHR-Y") for s in self.tracker_config.serials):
            print("[ViveTrackerInputDevice] Warning: Tracker serials not configured or using defaults.")
            if self.tracker_config.interactive_setup:
                response = input("Would you like to configure tracker serials now? [y/N]: ").strip().lower()
                if response == 'y':
                    self._interactive_config_setup(self.tracker_config.robot_config_path)
                    # Reload config
                    try:
                        self.tracker_config = ViveTrackerConfig.from_file(self.tracker_config.robot_config_path)
                    except Exception:
                        pass
            else:
                print("[ViveTrackerInputDevice] Using mock mode (no real trackers).")
                self._tracker = None
                return
                
        try:
            from vive_tracker import HTCViveTracker
            
            serials = self.tracker_config.serials
            if not serials:
                print("[ViveTrackerInputDevice] No tracker serials provided. Using mock mode.")
                self._tracker = None
                return
                
            print(f"[ViveTrackerInputDevice] Initializing trackers: {serials}")
            self._tracker = HTCViveTracker(
                serials=serials,
                auto_calib_tracker=self.tracker_config.auto_calib_tracker,
            )
            
            # Handle calibration
            self._handle_calibration()
                    
            print("[ViveTrackerInputDevice] Started successfully.")
            
        except ImportError as e:
            print(f"[ViveTrackerInputDevice] Failed to import vive_tracker: {e}")
            print("[ViveTrackerInputDevice] Please ensure vive_tracker is installed.")
            self._tracker = None
            
    def _handle_calibration(self):
        """Handle calibration file - load existing or prompt for new."""
        import os
        
        calib_path = self.tracker_config.calib_path
        init_poses = self.tracker_config.init_robot_poses
        
        if not init_poses:
            # Use defaults
            init_poses = [
                np.array(ViveTrackerConfig.DEFAULT_INIT_POSE_LEFT),
                np.array(ViveTrackerConfig.DEFAULT_INIT_POSE_RIGHT),
            ]
            self.tracker_config.init_robot_poses = init_poses
            
        if os.path.exists(calib_path):
            # Load existing calibration
            try:
                self._tracker.set_calibration_from_file(init_poses, calib_path)
                self._is_calibrated = True
                print(f"[ViveTrackerInputDevice] Loaded calibration from {calib_path}")
            except Exception as e:
                print(f"[ViveTrackerInputDevice] Failed to load calibration: {e}")
                self._prompt_calibration()
        else:
            print(f"[ViveTrackerInputDevice] Calibration file not found: {calib_path}")
            self._prompt_calibration()
            
    def _prompt_calibration(self):
        """Prompt user to perform calibration."""
        if not self.tracker_config.interactive_setup:
            print("[ViveTrackerInputDevice] Calibration required. Call calibrate() manually.")
            return
            
        print()
        print("=" * 60)
        print("CALIBRATION REQUIRED")
        print("=" * 60)
        print("To calibrate:")
        print("  1. Position robot arms at their initial poses")
        print("  2. Attach trackers firmly to end-effectors")
        print("  3. Ensure SteamVR is running and trackers are tracked")
        print("=" * 60)
        
        response = input("\nPerform calibration now? [Y/n]: ").strip().lower()
        if response != 'n':
            success = self.calibrate(save=True)
            if not success:
                print("[ViveTrackerInputDevice] Calibration failed. Manual calibration required.")
        else:
            print("[ViveTrackerInputDevice] Calibration skipped. Call calibrate() when ready.")
            
    def stop(self):
        """Stop ViveTracker connection."""
        super().stop()
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception as e:
                print(f"[ViveTrackerInputDevice] Error stopping tracker: {e}")
            self._tracker = None
        print("[ViveTrackerInputDevice] Stopped.")
        
    def calibrate(self, save: bool = True):
        """Perform calibration between tracker and robot coordinate frames.
        
        This should be called once when first setting up the system.
        The robot should be at the initial poses specified in tracker_config.
        
        Args:
            save: Whether to save calibration to file
        """
        if self._tracker is None:
            print("[ViveTrackerInputDevice] Cannot calibrate - tracker not initialized.")
            return False
            
        if not self.tracker_config.init_robot_poses:
            print("[ViveTrackerInputDevice] Cannot calibrate - no init_robot_poses provided.")
            return False
            
        try:
            print("[ViveTrackerInputDevice] Starting calibration...")
            print("[ViveTrackerInputDevice] Ensure robot is at initial pose and tracker is attached.")
            
            self._tracker.calibrate_robot_tracker_transform(
                self.tracker_config.init_robot_poses,
                self.tracker_config.calib_path if save else None,
            )
            
            self._is_calibrated = True
            print(f"[ViveTrackerInputDevice] Calibration complete. Saved to {self.tracker_config.calib_path}")
            return True
            
        except Exception as e:
            print(f"[ViveTrackerInputDevice] Calibration failed: {e}")
            return False
            
    def pause(self):
        """Pause tracking (intervention disabled)."""
        with self._lock:
            self._is_paused = True
            if self._tracker is not None:
                try:
                    self._tracker.pause()
                except AttributeError:
                    pass
                    
    def resume(self):
        """Resume tracking (intervention enabled)."""
        with self._lock:
            self._is_paused = False
            self._prev_tracker_poses = None  # Reset delta tracking
            if self._tracker is not None:
                try:
                    self._tracker.start()
                except AttributeError:
                    pass
                    
    def reset(self):
        """Reset intervention state."""
        with self._lock:
            self._intervention_enabled = False
            self._prev_tracker_poses = None
            self._gripper_states = [1.0] * max(1, len(self.tracker_config.serials))
            
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Get action from ViveTracker.
        
        Returns:
            Tuple of (action, is_intervening):
                - action: numpy array of shape (action_dim,)
                  For single tracker: [dx, dy, dz, drx, dry, drz, gripper]
                  For dual trackers: [dx1, dy1, dz1, drx1, dry1, drz1, g1, dx2, dy2, dz2, drx2, dry2, drz2, g2]
                - is_intervening: True if human is actively controlling
        """
        with self._lock:
            # Check if paused
            if self._is_paused:
                return np.zeros(self.config.action_dim), False
                
            # Check if tracker available
            if self._tracker is None or not self._is_calibrated:
                return np.zeros(self.config.action_dim), False
                
            try:
                # Get aligned robot poses from tracker
                aligned_poses = self._tracker.get_aligned_robot_pose()
                
                # Handle single vs multiple trackers
                if not isinstance(aligned_poses, (list, tuple)):
                    aligned_poses = [aligned_poses]
                    
                # Compute actions
                if self.tracker_config.use_delta_actions:
                    action, is_intervening = self._compute_delta_action(aligned_poses)
                else:
                    action, is_intervening = self._compute_absolute_action(aligned_poses)
                    
                return action, is_intervening
                
            except Exception as e:
                print(f"[ViveTrackerInputDevice] Error getting action: {e}")
                return np.zeros(self.config.action_dim), False
                
    def _compute_delta_action(
        self, 
        current_poses: list,
    ) -> Tuple[np.ndarray, bool]:
        """Compute delta action from pose change.
        
        Args:
            current_poses: List of current aligned robot poses [x, y, z, qx, qy, qz, qw]
            
        Returns:
            Tuple of (action, is_intervening)
        """
        num_trackers = len(current_poses)
        action_per_tracker = 7  # dx, dy, dz, drx, dry, drz, gripper
        
        action = np.zeros(num_trackers * action_per_tracker)
        is_intervening = False
        
        if self._prev_tracker_poses is None:
            self._prev_tracker_poses = [pose.copy() for pose in current_poses]
            return action, False
            
        for i, (current, prev) in enumerate(zip(current_poses, self._prev_tracker_poses)):
            # Compute position delta
            delta_pos = (current[:3] - prev[:3]) * self.tracker_config.translation_scale
            
            # Compute rotation delta (simplified - using quaternion difference)
            delta_rot = self._quat_diff_to_axis_angle(prev[3:7], current[3:7])
            delta_rot *= self.tracker_config.rotation_scale
            
            # Apply velocity limits
            dt = 1.0 / self.tracker_config.control_freq
            max_delta_pos = self.tracker_config.max_velocity * dt
            max_delta_rot = self.tracker_config.max_angular_velocity * dt
            
            delta_pos = np.clip(delta_pos, -max_delta_pos, max_delta_pos)
            delta_rot = np.clip(delta_rot, -max_delta_rot, max_delta_rot)
            
            # Check if motion exceeds deadzone
            motion_magnitude = np.linalg.norm(delta_pos) + np.linalg.norm(delta_rot) * 0.1
            if motion_magnitude > self.tracker_config.deadzone:
                is_intervening = True
                
            # Build action
            offset = i * action_per_tracker
            action[offset:offset+3] = delta_pos
            action[offset+3:offset+6] = delta_rot
            action[offset+6] = self._gripper_states[i]
            
        # Update previous poses
        self._prev_tracker_poses = [pose.copy() for pose in current_poses]
        
        return action, is_intervening
        
    def _compute_absolute_action(
        self, 
        current_poses: list,
    ) -> Tuple[np.ndarray, bool]:
        """Return absolute poses as actions.
        
        Args:
            current_poses: List of current aligned robot poses [x, y, z, qx, qy, qz, qw]
            
        Returns:
            Tuple of (action, is_intervening)
        """
        num_trackers = len(current_poses)
        action_per_tracker = 8  # x, y, z, qx, qy, qz, qw, gripper
        
        action = np.zeros(num_trackers * action_per_tracker)
        
        for i, pose in enumerate(current_poses):
            offset = i * action_per_tracker
            action[offset:offset+7] = pose
            action[offset+7] = self._gripper_states[i]
            
        # For absolute mode, always consider as intervention when tracker is active
        is_intervening = not self._is_paused
        
        return action, is_intervening
        
    def _quat_diff_to_axis_angle(
        self, 
        q1: np.ndarray, 
        q2: np.ndarray,
    ) -> np.ndarray:
        """Compute axis-angle rotation from q1 to q2.
        
        Args:
            q1: Source quaternion [qx, qy, qz, qw]
            q2: Target quaternion [qx, qy, qz, qw]
            
        Returns:
            Axis-angle rotation [rx, ry, rz]
        """
        # Compute relative quaternion: q_rel = q2 * q1_inv
        q1_inv = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
        q_rel = self._quat_multiply(q2, q1_inv)
        
        # Convert to axis-angle
        angle = 2.0 * np.arccos(np.clip(q_rel[3], -1.0, 1.0))
        
        if np.abs(angle) < 1e-6:
            return np.zeros(3)
            
        sin_half = np.sin(angle / 2.0)
        if np.abs(sin_half) < 1e-6:
            return np.zeros(3)
            
        axis = q_rel[:3] / sin_half
        return axis * angle
        
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.
        
        Args:
            q1, q2: Quaternions [qx, qy, qz, qw]
            
        Returns:
            Product quaternion [qx, qy, qz, qw]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])
        
    def set_gripper(self, gripper_idx: int, state: float):
        """Set gripper state for a specific arm.
        
        Args:
            gripper_idx: Index of the gripper (0 for left/single, 1 for right)
            state: Gripper state (-1 to 1, where 1 is open)
        """
        with self._lock:
            if 0 <= gripper_idx < len(self._gripper_states):
                self._gripper_states[gripper_idx] = state
                
    def toggle_gripper(self, gripper_idx: int):
        """Toggle gripper state for a specific arm.
        
        Args:
            gripper_idx: Index of the gripper (0 for left/single, 1 for right)
        """
        with self._lock:
            if 0 <= gripper_idx < len(self._gripper_states):
                self._gripper_states[gripper_idx] *= -1
                
    @property
    def is_calibrated(self) -> bool:
        """Check if tracker is calibrated."""
        return self._is_calibrated
        
    @property
    def is_paused(self) -> bool:
        """Check if tracking is paused."""
        with self._lock:
            return self._is_paused


@register_input_device("dual_vivetracker")
class DualViveTrackerInputDevice(ViveTrackerInputDevice):
    """Specialized ViveTracker input device for dual-arm robot teleoperation.
    
    This is a convenience wrapper that ensures proper configuration for
    dual-arm setups like dual Flexiv arms.
    
    Example usage with config file:
        device = DualViveTrackerInputDevice.from_config_file("vive_config.yaml")
        device.start()
        
    Example usage programmatic:
        config = ViveTrackerConfig(
            serials=["LHR-37267EB0", "LHR-ED74041F"],  # Left, Right
            init_robot_poses=[
                np.array([0.7, 0.1, 0, 0, 0, 1, 0]),   # Left arm
                np.array([0.7, -0.1, -0.02, 0, 0, 1, 0]),  # Right arm
            ],
            calib_path="dual_calib.json",
        )
        device = DualViveTrackerInputDevice(tracker_config=config)
    """
    
    def __init__(
        self,
        config: Optional[InputDeviceConfig] = None,
        tracker_config: Optional[ViveTrackerConfig] = None,
        **kwargs,
    ):
        # Ensure action_dim is set for dual arm (14 = 7 per arm)
        if config is None:
            config = InputDeviceConfig(action_dim=14)
        elif config.action_dim < 14:
            config.action_dim = 14
            
        super().__init__(config, tracker_config, **kwargs)
        
        if tracker_config and len(tracker_config.serials) != 2:
            print(f"[DualViveTrackerInputDevice] Warning: Expected 2 tracker serials, "
                  f"got {len(tracker_config.serials)}")
                  
    @classmethod
    def from_config_file(
        cls,
        config_path: str = "vive_robot_config.yaml",
        input_config: Optional[InputDeviceConfig] = None,
        create_if_missing: bool = True,
        interactive: bool = True,
    ) -> "DualViveTrackerInputDevice":
        """Create DualViveTrackerInputDevice from a config file.
        
        If the config file doesn't exist and create_if_missing=True,
        creates a default config file and optionally prompts user to edit it.
        
        Args:
            config_path: Path to config file (.yaml or .json)
            input_config: Optional InputDeviceConfig (defaults to 14 action dims)
            create_if_missing: Create default config if file not found
            interactive: Prompt user for input when config is missing
            
        Returns:
            DualViveTrackerInputDevice instance
        """
        import os
        
        # Ensure dual-arm action dimensions
        if input_config is None:
            input_config = InputDeviceConfig(action_dim=14)
        elif input_config.action_dim < 14:
            input_config.action_dim = 14
        
        if not os.path.exists(config_path):
            if create_if_missing:
                print(f"[DualViveTrackerInputDevice] Config file not found: {config_path}")
                print("[DualViveTrackerInputDevice] Creating default configuration file...")
                
                created_path = ViveTrackerConfig.create_default_config_file(config_path)
                print(f"[DualViveTrackerInputDevice] Created: {created_path}")
                print()
                print("=" * 60)
                print("DEFAULT CONFIGURATION FILE CREATED")
                print("=" * 60)
                print(f"Please edit {created_path} to configure:")
                print("  1. Tracker serial numbers (find in SteamVR)")
                print("  2. Robot initial poses")
                print("  3. Motion parameters")
                print("=" * 60)
                
                if interactive:
                    response = input("\nWould you like to edit the config now? [y/N]: ").strip().lower()
                    if response == 'y':
                        ViveTrackerInputDevice._interactive_config_setup(created_path)
                    else:
                        print(f"\nUsing defaults. Edit {created_path} manually and restart.")
                        
                config_path = created_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        tracker_config = ViveTrackerConfig.from_file(config_path)
        tracker_config.interactive_setup = interactive
        
        return cls(config=input_config, tracker_config=tracker_config)
                  
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Get dual-arm action from ViveTrackers.
        
        Returns:
            Tuple of (action, is_intervening):
                - action: numpy array of shape (14,)
                  [dx_L, dy_L, dz_L, drx_L, dry_L, drz_L, grip_L,
                   dx_R, dy_R, dz_R, drx_R, dry_R, drz_R, grip_R]
                - is_intervening: True if either arm is being controlled
        """
        return super().get_action()
        
    def get_left_action(self) -> Tuple[np.ndarray, bool]:
        """Get action for left arm only."""
        action, is_intervening = self.get_action()
        return action[:7], is_intervening
        
    def get_right_action(self) -> Tuple[np.ndarray, bool]:
        """Get action for right arm only."""
        action, is_intervening = self.get_action()
        return action[7:14] if len(action) >= 14 else np.zeros(7), is_intervening

