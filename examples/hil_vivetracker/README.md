# ViveTracker HIL for Dual Flexiv Arms

This example demonstrates how to use HTC Vive Trackers for Human-in-the-Loop (HIL) teleoperation of dual Flexiv robot arms during RL training.

## Quick Start

### First Run (Interactive Setup)

Simply run the script - it will guide you through configuration:

```bash
python simple_vivetracker_teleop.py
```

On first run, the script will:
1. Create a default config file (`vive_robot_config.yaml`)
2. Prompt you to enter tracker serial numbers
3. Prompt for robot initial poses
4. Perform calibration

### Subsequent Runs

After initial setup, just run:

```bash
python simple_vivetracker_teleop.py
```

## Configuration Files

### Automatic Config Creation

If `vive_robot_config.yaml` doesn't exist, the wrapper automatically creates one:

```yaml
vivetracker:
  serials:
    left: LHR-XXXXXXXX   # Replace with your tracker serial
    right: LHR-YYYYYYYY  # Replace with your tracker serial
  calibration_path: vive_calib.json
  use_delta_actions: true
  translation_scale: 1.0
  rotation_scale: 1.0
  max_velocity: 1.0
  max_angular_velocity: 2.0
  control_freq: 20.0

robot:
  init_poses:
    left: [0.7, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0]
    right: [0.7, -0.1, -0.02, 0.0, 0.0, 1.0, 0.0]
```

### Calibration File

The calibration file (`vive_calib.json`) stores the transformation between tracker and robot coordinate frames. If missing:

- **Interactive mode**: Prompts you to perform calibration
- **Non-interactive mode**: Uses identity transform (may cause coordinate misalignment)

## Usage Examples

```bash
# Interactive setup (first run)
python simple_vivetracker_teleop.py

# Custom config path
python simple_vivetracker_teleop.py --config my_config.yaml

# Force recalibration
python simple_vivetracker_teleop.py --calibrate

# Non-interactive mode (use defaults)
python simple_vivetracker_teleop.py --no-interactive

# Test with mock robots (no hardware)
python simple_vivetracker_teleop.py --mock

# Create config file only
python simple_vivetracker_teleop.py --create-config
```

## Controls

| Key | Action |
|-----|--------|
| `P` | Pause/Resume tracking |
| `G` | Toggle left gripper |
| `H` | Toggle right gripper |
| `ESC` | Stop |

## Integration with RLinf

The `ViveTrackerInputDevice` is a proper wrapper under `rlinf/envs/hil/input_devices.py`:

### Using Config File (Recommended)

```python
from rlinf.envs.hil.input_devices import DualViveTrackerInputDevice

# Automatically handles missing config and calibration
tracker = DualViveTrackerInputDevice.from_config_file(
    config_path="vive_robot_config.yaml",
    create_if_missing=True,  # Create default config if not found
    interactive=True,        # Prompt user for missing values
)
tracker.start()

# Control loop
while running:
    action, is_intervening = tracker.get_action()
    if is_intervening:
        env.step(action)
```

### Programmatic Configuration

```python
from rlinf.envs.hil.input_devices import (
    DualViveTrackerInputDevice,
    ViveTrackerConfig,
    InputDeviceConfig,
)
import numpy as np

# Create config programmatically
tracker_config = ViveTrackerConfig(
    serials=["LHR-37267EB0", "LHR-ED74041F"],
    init_robot_poses=[
        np.array([0.7, 0.1, 0, 0, 0, 1, 0]),
        np.array([0.7, -0.1, -0.02, 0, 0, 1, 0]),
    ],
    calib_path="calib.json",
    interactive_setup=True,  # Prompt if calibration missing
)

tracker = DualViveTrackerInputDevice(tracker_config=tracker_config)
tracker.start()
```

### Config File Utilities

```python
from rlinf.envs.hil.input_devices import ViveTrackerConfig

# Create default config file
ViveTrackerConfig.create_default_config_file("my_config.yaml")

# Load config from file
config = ViveTrackerConfig.from_file("my_config.yaml")

# Save config to file
config.to_file("my_config_backup.yaml")
```

## HIL Training Integration

For HIL-augmented RL training, the wrapper integrates with `EmbodiedHILRunner`:

```python
from rlinf.runners.embodied_hil_runner import EmbodiedHILRunner, EmbodiedHILConfig

config = EmbodiedHILConfig(
    enable_hil=True,
    input_device="dual_vivetracker",  # Uses registered wrapper
    bc_weight=0.1,
)

runner = EmbodiedHILRunner(env, policy, value_fn, config)
runner.train(max_steps=100000)
```

## File Structure

```
examples/hil_vivetracker/
├── README.md                           # This file
├── simple_vivetracker_teleop.py        # Simple teleoperation script
├── train_dual_flexiv_vivetracker.py    # Full HIL training script
└── config/
    └── dual_flexiv_hil.yaml            # Example training config
```

## Troubleshooting

### "Config file not found"
The wrapper will automatically create a default config. Follow the prompts.

### "Calibration file not found"
Run calibration when prompted, or use `--calibrate` flag.

### "vive_tracker module not found"
Install the vive_tracker module from your internal repo.

### "pynput not installed"
```bash
pip install pynput
```

### Trackers not detected
- Ensure SteamVR is running
- Check trackers are powered on and paired
- Verify serial numbers in config match SteamVR

### Coordinate mismatch
- Delete calibration file and recalibrate
- Ensure robot is at exact initial pose during calibration
