#!/usr/bin/env python3
"""
Device Starter - Starts device processes and the Device Manager.

This module provides centralized management for running multiple devices simultaneously
using separate processes for isolation and reliability, plus the new Device Manager
for data aggregation.

Author: Jun Lv
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from typing import Dict, List, Any, Optional
import numpy as np

# Add devices to path
sys.path.append(os.path.dirname(__file__))
from devices import DEVICE_CLASSES
from manager import MANAGER_CLASSES
from visualizers import VISUALIZER_CLASSES
from utils.shm_utils import get_dtype
from utils.config_parser import ensure_config_dict, parse_device_configs_with_fields, load_config
from utils.logger_config import logger
import matplotlib

class DeviceStarter:
    """
    Device starter that handles multiple device processes and the Device Manager.
    
    Features:
    - Load configuration from YAML file
    - Start/stop multiple devices in separate processes
    - Start/stop the Device Manager for data aggregation
    - Monitor process health and restart on failure
    - Graceful shutdown handling
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        enable_visualizers: bool = False,
        enable_device_manager: bool = True,
    ) -> None:
        """
        Initialize the device starter.
        
        Args:
            config_path: Path to configuration YAML file
            enable_visualizers: Whether to start visualizers for each device
            enable_device_manager: Whether to start the Device Manager for data aggregation
        """
        if config is None:
            resolved_path = config_path or "config.yaml"
            logger.info(f"Loading configuration from {resolved_path}")
            self.config_path = resolved_path
            self.config = load_config(resolved_path)
        else:
            # When an explicit config dictionary is provided, prefer using it directly
            # and avoid falling back to a default file path to prevent accidental overrides.
            self.config_path = config_path
            self.config = ensure_config_dict(config)
        self.devices: List[Dict[str, Any]] = []
        self.running = False
        self.enable_visualizers = enable_visualizers
        self.enable_device_manager = enable_device_manager
        self.device_manager_process = None
        self.policy_runner_process = None
        
        # Load configuration
        logger.info("DeviceStarter initialized")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    
    def _parse_device_configs(self) -> List[Dict[str, Any]]:
        """Parse device configurations and create device dictionaries."""
        # Use generic parser with additional fields for device starter
        additional_fields = {
            'process': None,
            'visualizer_process': None,
            'restart_count': 0,
            'last_restart': None
        }
        return parse_device_configs_with_fields(self.config, additional_fields)
    

    
    def _start_generic_device_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        """Start a device process using dynamic import based on device class."""
        try:
            # Get device class from mapping
            device_class = DEVICE_CLASSES.get(device_info['device_class'])
            if not device_class:
                logger.error(f"Unknown device class: {device_info['device_class']}")
                return None
            
            def run_device() -> None:
                try:
                    # Get device class from devices module
                    from devices import DEVICE_CLASSES
                    device_class = DEVICE_CLASSES.get(device_info['device_class'])
                    if not device_class:
                        raise ValueError(f"Unknown device class: {device_info['device_class']}")
                    
                    # Create device with all config parameters
                    device_config = device_info['config'].copy()
                    
                    # Remove fields that are not device constructor parameters
                    device_config.pop('class', None)
                    
                    # Get device class constructor parameters
                    import inspect
                    sig = inspect.signature(device_class.__init__)
                    valid_params = list(sig.parameters.keys())
                    
                    # Filter config to only include valid constructor parameters
                    filtered_config = {k: v for k, v in device_config.items() if k in valid_params}
                    
                    # Create device instance with filtered config parameters
                    device = device_class(**filtered_config)
                    
                    logger.info(
                        f"Starting {device_info['device_class']} {device_info['config']['device_id']}"
                    )
                    device.start_server()
                    
                except Exception as e:
                    logger.error(f"{device_info['device_class']} {device_info['config']['device_id']} error: {e}")
                    raise
            
            process = mp.Process(target=run_device, name=f"{device_info['device_class']}-{device_info['config']['device_id']}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to create device process for {device_info['device_class']}: {e}")
            return None
    
    def _start_generic_visualizer_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        """Start a visualizer process using dynamic import based on device class."""
        try:
            def run_visualizer() -> None:
                try:
                    # Set matplotlib backend to avoid GUI conflicts in multiprocessing
                    matplotlib.use('TkAgg')
                    
                    # Get visualizer class from mapping
                    visualizer_class = VISUALIZER_CLASSES.get(device_info['device_class'])
                    if not visualizer_class:
                        logger.error(f"Unknown visualizer type: {device_info['device_class']}")
                        return
                    
                    # Get device name directly from device class
                    device_name = device_info['device_class']
                    
                    shared_memory_name = f"{device_name}_{device_info['device_id']}_data"
                    # Calculate update interval based on fps (convert to milliseconds)
                    update_interval = int(1000 / device_info['config']['fps'])
                    
                    # Get data type from config
                    data_dtype = get_dtype(device_info['config'].get('data_dtype', 'float32'))
                    
                    # Create visualizer
                    visualizer = visualizer_class(
                        shared_memory_name=shared_memory_name,
                        data_dtype=data_dtype,
                        smoothing_window=5
                    )
                    
                    # Try to connect with retries
                    max_retries = 10
                    retry_delay = 0.5
                    for attempt in range(max_retries):
                        if visualizer.connect():
                            break
                        logger.info(f"{device_info['device_class'].capitalize()} visualizer {device_info['device_id']}: Waiting for shared memory (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{device_info['device_class'].capitalize()} visualizer {device_info['device_id']}: Failed to connect after {max_retries} attempts")
                        return
                    
                    # Start visualization
                    visualizer.start_visualization(update_interval=update_interval)
                    
                except Exception as e:
                    logger.error(f"{device_info['device_class'].capitalize()} visualizer error: {e}")
            
            process = mp.Process(target=run_visualizer, name=f"{device_info['device_class']}_visualizer_{device_info['device_id']}")
            process.start()
            return process
            
        except Exception as e:
            logger.error(f"Failed to create visualizer process for {device_info['device_class']}: {e}")
            return None
    
    def _start_device_manager_process(self) -> Optional[mp.Process]:
        """Start the Device Manager process for data aggregation."""
        try:
            def run_device_manager() -> None:
                try:                    
                    # Get device manager configuration
                    device_manager_config = self.config.get('device_manager', {})
                    manager_type = device_manager_config.get('type', 'base')
                    
                    # Get manager class from mapping
                    manager_class = MANAGER_CLASSES.get(manager_type)
                    if not manager_class:
                        raise ValueError(f"Unknown device manager type: {manager_type}")

                    manager_params = device_manager_config.get('params', {}) or {}
                    manager_params = ensure_config_dict(manager_params)
                    
                    # For Hydra configs, pass the config object; for traditional configs, pass config_path
                    if hasattr(self, 'config_path') and self.config_path and not manager_params.get('config_path'):
                        # Traditional config mode - pass config_path
                        manager_params.setdefault('config_path', self.config_path)
                    else:
                        # Hydra config mode - pass config object
                        manager_params.setdefault('config', self.config)
                    
                    manager = manager_class(**manager_params)
                    logger.info(f"Device Manager ({manager_type}) started for data aggregation")
                    manager.start()
                    
                except Exception as e:
                    logger.error(f"Device Manager error: {e}")
                    raise
            
            process = mp.Process(target=run_device_manager, name="DeviceManager")
            return process
            
        except Exception as e:
            logger.error(f"Failed to create Device Manager process: {e}")
            return None
    
    def _start_policy_runner_process(self) -> Optional[mp.Process]:
        """Start the Policy Runner process for action dispatching."""
        try:
            def run_policy_runner() -> None:
                try:                    
                    # Get policy runner configuration
                    policy_runner_config = self.config.get('action_executor', {})
                    if not policy_runner_config:
                        logger.warning("No action_executor configuration found, skipping PolicyRunner")
                        return
                    
                    runner_type = policy_runner_config.get('type', 'base_policy_runner')
                    
                    # Get manager class from mapping
                    manager_class = MANAGER_CLASSES.get(runner_type)
                    if not manager_class:
                        raise ValueError(f"Unknown policy runner type: {runner_type}")                    
                    
                    # Create policy runner instance - extract only known parameters
                    valid_params = {
                        'config_path', 'policy_shm_name', 'robot_control_shms',
                        'execution_fps', 'chunk_length', 'chunk_manager', 'config'
                        'enable_action_visualizer', 'action_visualizer_config', 'device_delays',
                        'device_dimension_delays'
                    }
                    
                    runner_config = {}
                    for key, value in policy_runner_config.items():
                        if key in valid_params:
                            runner_config[key] = value
                    
                    # For Hydra configs, pass the config object; for traditional configs, pass config_path
                    if hasattr(self, 'config_path') and self.config_path and not runner_config.get('config_path'):
                        # Traditional config mode - pass config_path
                        runner_config.setdefault('config_path', self.config_path)
                    else:
                        # Hydra config mode - pass config object
                        runner_config.setdefault('config', self.config)
                    
                    logger.debug(f"Policy runner config keys: {list(runner_config.keys())}")
                    runner = manager_class(**runner_config)
                    logger.info(f"Policy Runner ({runner_type}) started for action dispatching")
                    runner.start()
                    
                except Exception as e:
                    logger.error(f"Policy Runner error: {e}")
                    raise
            
            process = mp.Process(target=run_policy_runner, name="PolicyRunner")
            return process
            
        except Exception as e:
            logger.error(f"Failed to create Policy Runner process: {e}")
            return None
    
    def _start_device_process(self, device_info: Dict[str, Any]) -> bool:
        """Start a single device process and optionally its visualizer."""
        try:
            # Start device process using dynamic import
            process = self._start_generic_device_process(device_info)
            if not process:
                return False
            
            process.start()
            device_info['process'] = process
            device_info['last_restart'] = time.time()
            
            logger.info(f"Started {device_info['device_class']} device {device_info['device_id']} (PID: {process.pid})")
            
            # Start visualizer if enabled
            if self.enable_visualizers:
                try:
                    # Wait a bit for device to initialize
                    time.sleep(1.0)
                    
                    visualizer_process = self._start_generic_visualizer_process(device_info)
                    if visualizer_process:
                        device_info['visualizer_process'] = visualizer_process
                        logger.info(f"Started {device_info['device_class']} visualizer for device {device_info['device_id']} (PID: {visualizer_process.pid})")
                    
                except Exception as e:
                    logger.warning(f"Failed to start visualizer for {device_info['device_class']} device {device_info['device_id']}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {device_info['device_class']} device {device_info['device_id']}: {e}")
            return False
    
    def _monitor_devices(self) -> None:
        """Monitor device processes and restart failed ones if configured."""
        global_settings = self.config.get('global_settings', {})
        restart_on_failure = global_settings.get('restart_on_failure', True)
        max_restart_attempts = global_settings.get('max_restart_attempts', 3)
        restart_delay = global_settings.get('restart_delay', 2.0)
        
        logger.info("Starting device monitoring...")
        
        try:
            while self.running:
                for device_info in self.devices:
                    if device_info['process'] and not device_info['process'].is_alive():
                        logger.warning(f"Device {device_info['device_class']}:{device_info['device_id']} process died")
                        
                        if restart_on_failure and device_info['restart_count'] < max_restart_attempts:
                            logger.info(f"Restarting device {device_info['device_class']}:{device_info['device_id']} "
                                       f"(attempt {device_info['restart_count'] + 1}/{max_restart_attempts})")
                            
                            time.sleep(restart_delay)
                            device_info['restart_count'] += 1
                            self._start_device_process(device_info)
                        else:
                            logger.error(f"Device {device_info['device_class']}:{device_info['device_id']} "
                                        f"exceeded max restart attempts or restart disabled")
                
                time.sleep(1.0)  # Check every second
        except KeyboardInterrupt:
            logger.info("Device monitoring interrupted")
            self.running = False
    
    def start_all(self) -> None:
        """Start all enabled devices and the Device Manager."""
        if self.running:
            logger.warning("Device starter is already running")
            return
        
        logger.info("Starting Device Starter")
        self.devices = self._parse_device_configs()
        
        if not self.devices:
            logger.warning("No enabled devices found in configuration")
            return
        
        # Start all devices first
        started_count = 0
        for device_info in self.devices:
            if self._start_device_process(device_info):
                started_count += 1
        
        if started_count == 0:
            logger.error("Failed to start any devices")
            return False
        
        # Wait for devices to initialize
        logger.info("Waiting for devices to initialize...")
        device_init_delay = self.config.get('global_settings', {}).get('device_init_delay', 5.0)
        logger.info(f"Waiting {device_init_delay} seconds for devices to initialize...")
        time.sleep(device_init_delay)
        
        # Start Device Manager after devices are ready
        if self.enable_device_manager:
            logger.info("Starting Device Manager for data aggregation...")
            self.device_manager_process = self._start_device_manager_process()
            if self.device_manager_process:
                self.device_manager_process.start()
                logger.info(f"Started Device Manager (PID: {self.device_manager_process.pid})")
            else:
                logger.error("Failed to start Device Manager")
        
        # Start policy runner if configured and not disabled
        action_executor_config = self.config.get('action_executor', {})
        if action_executor_config and action_executor_config.get('type') not in [None, 'null', 'disabled']:
            logger.info("Starting Policy Runner for action dispatching...")
            self.policy_runner_process = self._start_policy_runner_process()
            if self.policy_runner_process:
                self.policy_runner_process.start()
                logger.info(f"Started Policy Runner (PID: {self.policy_runner_process.pid})")
            else:
                logger.error("Failed to start Policy Runner")
        
        self.running = True
        logger.info(f"Successfully started {started_count}/{len(self.devices)} devices")
        return True
    
    def stop_all(self) -> None:
        """Stop all running devices, visualizers, and the Device Manager."""
        if not self.running:
            return
        
        logger.info("Stopping all devices, visualizers, and Device Manager...")
        self.running = False
        
        # Terminate all device and visualizer processes
        for device_info in self.devices:
            # Stop device process
            if device_info['process']:
                try:
                    if device_info['process'].is_alive():
                        logger.info(f"Stopping {device_info['device_class']} device {device_info['device_id']}")
                        device_info['process'].terminate()
                except (AssertionError, ValueError):
                    # Process may have already terminated
                    pass
            
            # Stop visualizer process
            if device_info['visualizer_process']:
                try:
                    if device_info['visualizer_process'].is_alive():
                        logger.info(f"Stopping {device_info['device_class']} visualizer for device {device_info['device_id']}")
                        device_info['visualizer_process'].terminate()
                except (AssertionError, ValueError):
                    # Process may have already terminated
                    pass
        
        # Wait for processes to terminate
        for device_info in self.devices:
            # Wait for device process
            if device_info['process']:
                try:
                    device_info['process'].join(timeout=5.0)
                    if device_info['process'].is_alive():
                        logger.warning(f"Force killing {device_info['device_class']} device {device_info['device_id']}")
                        device_info['process'].kill()
                except (AssertionError, ValueError):
                    # Process may have already terminated
                    pass
            
            # Wait for visualizer process
            if device_info['visualizer_process']:
                try:
                    device_info['visualizer_process'].join(timeout=5.0)
                    if device_info['visualizer_process'].is_alive():
                        logger.warning(f"Force killing {device_info['device_class']} visualizer for device {device_info['device_id']}")
                        device_info['visualizer_process'].kill()
                except (AssertionError, ValueError):
                    # Process may have already terminated
                    pass
        
        # Stop Device Manager
        if self.device_manager_process:
            try:
                if self.device_manager_process.is_alive():
                    logger.info("Stopping Device Manager")
                    self.device_manager_process.terminate()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass
            
            try:
                self.device_manager_process.join(timeout=5.0)
                if self.device_manager_process.is_alive():
                    logger.warning("Force killing Device Manager")
                    self.device_manager_process.kill()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass
        
        # Stop policy runner
        if self.policy_runner_process:
            try:
                if self.policy_runner_process.is_alive():
                    logger.info("Stopping Policy Runner")
                    self.policy_runner_process.terminate()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass
            
            try:
                self.policy_runner_process.join(timeout=5.0)
                if self.policy_runner_process.is_alive():
                    logger.warning("Force killing Policy Runner")
                    self.policy_runner_process.kill()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass
        
        logger.info("All devices, visualizers, and Device Manager stopped")
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed devices, visualizers, and Device Manager."""
        # Parse device configs to get total count
        all_devices = self._parse_device_configs()
        
        status = {
            'running': self.running,
            'total_devices': len(all_devices),
            'enable_visualizers': self.enable_visualizers,
            'enable_device_manager': self.enable_device_manager,
            'device_manager_running': False,
            'device_manager_pid': None,
            'devices': []
        }
        
        # Check Device Manager status
        if self.device_manager_process:
            try:
                status['device_manager_running'] = self.device_manager_process.is_alive()
                status['device_manager_pid'] = self.device_manager_process.pid
            except (AssertionError, ValueError):
                status['device_manager_running'] = False
                status['device_manager_pid'] = None
        
        # Show configured devices
        for device_info in all_devices:
            device_status = {
                'type': device_info['device_class'],
                'id': device_info['device_id'],
                'configured': True,
                'running': False,
                'pid': None,
                'visualizer_running': False,
                'visualizer_pid': None,
                'restart_count': 0
            }
            
            # If starter is running, check actual status
            if self.running and self.devices:
                # Find matching device in running devices
                for running_device in self.devices:
                    if (running_device['device_class'] == device_info['device_class'] and 
                        running_device['device_id'] == device_info['device_id']):
                        device_status.update({
                            'running': running_device['process'].is_alive() if running_device['process'] else False,
                            'pid': running_device['process'].pid if running_device['process'] else None,
                            'visualizer_running': running_device['visualizer_process'].is_alive() if running_device['visualizer_process'] else None,
                            'visualizer_pid': running_device['visualizer_process'].pid if running_device['visualizer_process'] else None,
                            'restart_count': running_device['restart_count']
                        })
                        break
            
            status['devices'].append(device_status)
        
        return status


def _flatten_config(cfg: "DictConfig") -> Dict[str, Any]:
    """Convert DictConfig to plain dict for DeviceStarter."""
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def main() -> None:
    """Main function to run the device starter using Hydra."""
    import hydra
    from omegaconf import DictConfig
    import signal
    import sys
    
    # Pre-process custom flags before Hydra parses arguments
    # Support -v / --visualize flag for enabling visualizers
    if '-v' in sys.argv or '--visualize' in sys.argv:
        # Remove the flag from argv
        if '-v' in sys.argv:
            sys.argv.remove('-v')
        if '--visualize' in sys.argv:
            sys.argv.remove('--visualize')
        # Add Hydra override
        sys.argv.append('runtime.enable_visualizers=True')
    
    @hydra.main(config_path="configs", config_name="config", version_base="1.3")
    def hydra_main(cfg: DictConfig) -> None:
        merged_cfg = _flatten_config(cfg)
        runtime_cfg = merged_cfg.get("runtime", {})

        # Hydra resolves defaults into nested dictionaries, so normalize the 'devices' section
        # to be a list directly, as expected by parse_device_configs.
        if "devices" in merged_cfg and isinstance(merged_cfg["devices"], dict):
            merged_cfg["devices"] = merged_cfg["devices"].get("devices", [])

        # Normalize optional nested sections produced by config groups
        # Some group files may nest under their own key, e.g., {'action_executor': {...}}
        if "device_manager" in merged_cfg and isinstance(merged_cfg["device_manager"], dict):
            if "type" not in merged_cfg["device_manager"] and "device_manager" in merged_cfg["device_manager"]:
                merged_cfg["device_manager"] = merged_cfg["device_manager"]["device_manager"]

        if "action_executor" in merged_cfg and isinstance(merged_cfg["action_executor"], dict):
            if "type" not in merged_cfg["action_executor"] and "action_executor" in merged_cfg["action_executor"]:
                merged_cfg["action_executor"] = merged_cfg["action_executor"]["action_executor"]

        starter = DeviceStarter(
            config=merged_cfg,
            config_path=None,
            enable_visualizers=runtime_cfg.get("enable_visualizers", False),
            enable_device_manager=runtime_cfg.get("enable_device_manager", True),
        )

        def _shutdown(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, stopping DeviceStarter")
            starter.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            starter.start_all()
            starter._monitor_devices()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            starter.stop_all()
    
    hydra_main()


if __name__ == "__main__":
    main() 