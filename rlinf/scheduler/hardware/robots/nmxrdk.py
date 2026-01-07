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
class NmxRdkHWInfo(HardwareInfo):
    """Hardware information for a NmxRdk robot."""

    config: "NmxRdkConfig"


@Hardware.register()
class NmxRdkRobot(Hardware):
    """Hardware policy for NmxRdk robotic systems.

    This implementation handles NmxRdk robot hardware configuration
    for use in the RLinf training framework.
    """

    HW_TYPE = "NMXRDK"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["NmxRdkConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, (
            "NmxRdk robot hardware requires explicit configurations for robot IP and camera serials for its controller nodes."
        )
        robot_configs: list["NmxRdkConfig"] = []
        for config in configs:
            if isinstance(config, NmxRdkConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            nmxrdk_infos = []

            for config in robot_configs:
                nmxrdk_infos.append(
                    NmxRdkHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

                print(
                    f"[NmxRdkRobot] Initializing NmxRdk robot on node rank {node_rank} "
                    f"with IP {config.robot_ip}"
                )
                if config.camera_serials:
                    print(f"[NmxRdkRobot] Cameras configured: {config.camera_serials}")

            return HardwareResource(type=cls.HW_TYPE, infos=nmxrdk_infos)
        return None


@NodeHardwareConfig.register_hardware_config(NmxRdkRobot.HW_TYPE)
@dataclass
class NmxRdkConfig(HardwareConfig):
    """Configuration for a NmxRdk robotic system."""

    robot_ip: str
    """IP address of the robotic system."""

    camera_serials: Optional[list[str]] = None
    """List of camera serial numbers associated with the robot."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in nmxrdk config must be an integer. But got {type(self.node_rank)}."
        )

        # Validate IP format
        try:
            ipaddress.ip_address(self.robot_ip)
        except ValueError:
            raise ValueError(
                f"'robot_ip' in nmxrdk config must be a valid IP address format. But got {self.robot_ip}."
            )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
