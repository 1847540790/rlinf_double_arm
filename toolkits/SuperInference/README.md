# SuperInference - Multi-Device Data Management System

> **Note**: This README is AI-generated.

## Installation

### PyTorch Installation

PyTorch and related dependencies (`torch`, `torchvision`, `timm`) are **not** included in `requirements.txt`. Please install them manually based on your CUDA version and hardware.
For more options, visit the official PyTorch installation guide: https://pytorch.org/get-started/locally/

### Other Dependencies

After installing PyTorch, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Cursor Rules Usage
If you write code with Cursor, please refer to this [document](https://docs.cursor.com/en/context/rules) to add existing cursor rules of this project under `.cursor/rules`. These rules are very important for AI to improve code quality and performance for complex tasks.

## Overview

SuperInference is a robust and flexible system for managing and processing data from various devices, including physical sensors, cameras, and simulated robots. The system provides centralized device management, configurable data processing, and real-time visualization capabilities.

## System Architecture

The system consists of several key components working together to provide a unified data management solution:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Devices       │    │   Manager       │    │   Consumers     │
│                 │    │                 │    │                 │
│ • BaseDevice    │───▶│ • BaseManager   │───▶│ • Latency       │
│ • Camera        │    │ • SyncManager   │    │ • DataSaver     │
│ • Robot         │    │                 │    │ • Visualizers   │
│ • Teleoperator  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Shared        │    │   Summary       │    │   Local         │
│   Memory        │    │   SHM           │    │   Storage       │
│   (SHM)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Devices (`devices/`)

Devices are data sources that generate and write data to shared memory buffers.

#### BaseDevice
- **Purpose**: Generic device for testing and debugging
- **Features**: Random data generation, configurable shape and data type
- **Use Case**: System testing, data format validation

#### OpenCVCameraDevice
- **Purpose**: Real-time camera data capture and streaming
- **Features**: OpenCV integration, configurable resolution and FPS
- **Use Case**: Video streaming, computer vision applications

#### SimRobot
- **Purpose**: PyBullet-based simulated robot control
- **Features**: 7-DOF robotic arm simulation, joint/end-effector control
- **Use Case**: Robotics research, control algorithm testing

#### JointSlider
- **Purpose**: GUI-based teleoperator for robot control
- **Features**: Tkinter sliders, real-time joint position control
- **Use Case**: Manual robot control, teaching mode

### 2. Device Manager (`manager/`)

The Device Manager aggregates data from multiple devices into a unified summary shared memory.

#### BaseDeviceManager
- **Purpose**: Basic device data aggregation
- **Features**: Connects to device SHMs, reads latest frames
- **Use Case**: Simple multi-device data collection

#### SynchronizedDeviceManager
- **Purpose**: Time-synchronized data aggregation
- **Features**: Master device synchronization, hardware latency compensation
- **Use Case**: Multi-sensor fusion, time-critical applications

### 3. Consumers (`consumer/`)

Consumers process and utilize the aggregated data from the Device Manager.

#### BaseConsumer
- **Purpose**: Base class for all consumers
- **Features**: SHM connection, device data parsing
- **Use Case**: Foundation for custom consumers

#### LatencyConsumer
- **Purpose**: Device latency monitoring and visualization
- **Features**: Real-time latency calculation, trend analysis
- **Use Case**: System performance monitoring

#### DataSaverConsumer
- **Purpose**: Data storage to local files
- **Features**: Multi-format saving (images, NPZ), timestamp-based organization
- **Use Case**: Data logging, offline analysis

### 4. Visualizers (`visualizers/`)

Visualizers provide real-time data visualization and monitoring.

#### BaseVisualizer
- **Purpose**: Generic device status monitoring
- **Features**: FPS tracking, latency monitoring, status display
- **Use Case**: Device health monitoring

#### CameraVisualizer
- **Purpose**: Real-time camera stream display
- **Features**: Live video feed, status overlay
- **Use Case**: Camera monitoring, video analysis

### 5. Utilities (`utils/`)

Core utilities for shared memory management and configuration.

#### SHM Utils (`shm_utils.py`)
- **Purpose**: Unified shared memory format and utilities
- **Features**: Header packing/unpacking, buffer management
- **Use Case**: Standardized SHM operations across all components

#### Config Parser (`config_parser.py`)
- **Purpose**: YAML configuration parsing
- **Features**: Device configuration loading, validation
- **Use Case**: System configuration management

## Configuration

The system is configured through `config.yaml`, which defines:

```yaml
devices:
  - class: "BaseDevice"
    device_id: 0
    data_shape: [7]
    fps: 1000.0
    data_dtype: "float64"
    buffer_size: 1000
    hardware_latency_ms: 1.0

device_manager:
  type: "synchronized"
  master_device_id: 0
```

## Key Features

### 1. Shared Memory Architecture
- **Unified Format**: Consistent SHM layout across all components
- **Efficient Communication**: Zero-copy data transfer between processes
- **Buffer Management**: Circular buffer support for historical data

### 2. Time Synchronization
- **Master Device**: One device serves as timing reference
- **Latency Compensation**: Hardware latency differences are accounted for
- **Binary Search**: Efficient timestamp-based data retrieval

### 3. Modular Design
- **Plugin Architecture**: Easy addition of new device types
- **Configuration-Driven**: System behavior controlled via YAML
- **Process Isolation**: Each component runs in separate process

### 4. Real-Time Capabilities
- **Low Latency**: Optimized for real-time applications
- **Visualization**: Live data monitoring and display
- **Performance Monitoring**: Built-in latency and FPS tracking

## Usage Examples

### Starting the System
```bash
# Start all devices and manager
python device_starter.py

# Start specific components
python -m devices.camera
python -m manager.synchronized_device_manager
```

### Running Consumers
```bash
# Monitor system latency
python -m consumer.latency_visualizer

# Save data to files
python -m consumer.data_saver

# Visualize camera feed
python -m visualizers.camera_visualizer
```

### Robot Control
```bash
# Start joint slider teleoperator
python -m devices.teleoperator

# Start simulated robot
python -m devices.robot
```

## File Structure

```
SuperInference/
├── config.yaml                 # Main configuration file
├── device_starter.py           # System startup script
├── devices/                    # Device implementations
│   ├── base.py                # Base device class
│   ├── camera.py              # Camera device
│   ├── robot.py               # Robot implementations
│   ├── teleoperator.py        # GUI teleoperator
│   └── __init__.py
├── manager/                    # Device managers
│   ├── base_device_manager.py
│   ├── synchronized_device_manager.py
│   └── __init__.py
├── consumer/                   # Data consumers
│   ├── base.py
│   ├── latency_visualizer.py
│   ├── data_saver.py
│   └── __init__.py
├── visualizers/               # Data visualizers
│   ├── base_visualizer.py
│   ├── camera_visualizer.py
│   └── __init__.py
├── utils/                     # Utility functions
│   ├── shm_utils.py
│   ├── config_parser.py
│   └── __init__.py
└── scripts/                   # Utility scripts
    └── scan_opencv_cameras.py
```