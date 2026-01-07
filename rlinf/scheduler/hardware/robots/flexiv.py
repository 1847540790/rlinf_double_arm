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

import ipaddress
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class FlexivHWInfo(HardwareInfo):
    """Hardware information for a Flexiv mock robot."""

    config: "FlexivConfig"


@Hardware.register()
class MockFlexivRobot(Hardware):
    """Hardware policy for Flexiv mock robotic systems.
    
    This is a mock implementation that simulates Flexiv robot hardware
    without requiring actual hardware connections. It can be used for
    testing and development purposes.
    """

    HW_TYPE = "flexiv"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["FlexivConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the mock robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, (
            "Flexiv mock robot hardware requires explicit configurations for robot IP and camera serials for its controller nodes."
        )
        robot_configs: list["FlexivConfig"] = []
        for config in configs:
            if isinstance(config, FlexivConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            flexiv_infos = []

            for config in robot_configs:
                flexiv_infos.append(
                    FlexivHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

                # For mock robot, we skip hardware validation
                # Just log that we're using mock hardware
                print(
                    f"[MockFlexivRobot] Initializing mock Flexiv robot on node rank {node_rank} "
                    f"with IP {config.robot_ip} (mock mode - no actual hardware connection)"
                )
                if config.camera_serials:
                    print(
                        f"[MockFlexivRobot] Mock cameras configured: {config.camera_serials}"
                    )

            return HardwareResource(type=cls.HW_TYPE, infos=flexiv_infos)
        return None


@NodeHardwareConfig.register_hardware_config(MockFlexivRobot.HW_TYPE)
@dataclass
class FlexivConfig(HardwareConfig):
    """Configuration for a Flexiv mock robotic system.
    
    This configuration is used for mock Flexiv robots that simulate
    hardware behavior without requiring actual hardware connections.
    """

    robot_ip: str
    """IP address of the robotic system (used for configuration, not validated in mock mode)."""

    camera_serials: Optional[list[str]] = None
    """List of camera serial numbers associated with the robot (used for configuration, not validated in mock mode)."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in flexiv config must be an integer. But got {type(self.node_rank)}."
        )

        # Validate IP format (but don't check connectivity in mock mode)
        try:
            ipaddress.ip_address(self.robot_ip)
        except ValueError:
            raise ValueError(
                f"'robot_ip' in flexiv config must be a valid IP address format. But got {self.robot_ip}."
            )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)

