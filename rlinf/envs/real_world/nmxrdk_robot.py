"""NmxRdk Robot SDK.

Author: Weimin Zhou
"""

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from lib_py.base import Logging
from lib_py.base.custom_types import ErrorCode, Types, Optional
from lib_py.robot_sdk.upper_body.robot_base import RobotBase
from lib_py.utils.vision import pose_to_4x4, pose_to_7D

logger = Logging.init_lib_logger("NmxRdkRobot", "info")


class NmxRdkRobot(RobotBase):
    def __init__(self, config, controller):
        super().__init__(config, controller)
        self.controller = controller
        self.config = config

    def get_tcp_force_value(self):
        """Reads the current force-torque values at the robot's TCP.

        Measurements are typically in the tool coordinate frame.

        Returns:
            nd.array: [Fx, Fy, Fz,Mx, My, Mz]
        """
        msg = self.controller[0]["arm"].get_end_force()
        return np.array(msg.end_force, dtype=np.float32)

    def move_arm_joint_online(
        self,
        target_jnt_pos: Types.Array = None,
        target_jnt_vel: Types.Array = None,
        target_jnt_acc: Types.Array = None,
        freq: int = None,
        arm_id: int = None,
        follow: Optional[bool] = True,
    ) -> ErrorCode:
        """Move the arm to the target joint position with online control.

        Args:
            target_jnt_pos (Types.Array): array of target joint position
            target_jnt_vel (Types.Array): array of target joint velocity
            target_jnt_acc (Types.Array): array of target joint acceleration
            freq (int): frequency of the control loop
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.
        """
        if target_jnt_vel is None:
            target_jnt_vel = [90.0, 90.0, 112.5, 112.5, 112.5, 112.5]
        if target_jnt_acc is None:
            target_jnt_acc = [600.0, 600.0, 600.0, 600.0, 600.0, 600.0]

        self.controller[0]["arm"].move_joint(
            pos=target_jnt_pos,
            vel=target_jnt_vel,
            acc=target_jnt_acc,
            blocking=False,
            follow=follow,
            connect=False,
            high_follow=True,
            trajectory_mode=1,
            radio=100,
        )
        return ErrorCode("success!", 0)

    def move_arm_joint_offline(
        self,
        target_jnt_pos: Types.Array = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        arm_id: int = 0,
        target_jnt_vel=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        target_jnt_acc=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    ) -> ErrorCode:
        """Move the arm to the target joint position with offline control.

        Args:
            target_jnt_pos (Types.Array): array of target joint position
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.

        Returns:
            ErrorCode: error code
        """
        self.controller[0]["arm"].move_joint(
            pos=target_jnt_pos,
            vel=target_jnt_vel,
            acc=target_jnt_acc,
            blocking=True,
            follow=False,
            connect=False,
            high_follow=False,
            trajectory_mode=0,
            radio=10,
        )
        return ErrorCode("success!", 0)

    def move_heads(
        self,
        target_jnt_pos: Types.Array,
    ) -> ErrorCode:
        """Move head joints to the target joint position offline.

        Args:
            target_jnt_pos (Types.Array): array of target joint position

        Return:
            ErrorCode: error code
        """
        logger.warn("move_heads is not achieved in NmxRdkRobot Robot! ")
        return ErrorCode("failed!", -1)

    def move_waist(
        self,
        target_jnt_pos: Types.Array,
    ) -> ErrorCode:
        """Move waist joints to the target joint position offline.

        Args:
            target_jnt_pos (Types.Array): array of target joint position

        Return:
            ErrorCode: error code
        """
        logger.warn("move_waist is not achieved in NmxRdkRobot Robot! ")
        return ErrorCode("failed!", -1)

    def move_joint_online(
        self,
        target_jnt_pos: Types.Array,
        target_jnt_vel: Types.Array,
        target_jnt_acc: Types.Array,
        freq: int = None,
        arm_id: int = None,
    ) -> ErrorCode:
        """Move whole robot joint to the target joint position with online control.

        Args:
            target_jnt_pos (Types.Array): array of target joint position
            target_jnt_vel (Types.Array): array of target joint velocity
            target_jnt_acc (Types.Array): array of target joint acceleration
            freq (int): frequency of the control loop
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.

        Returns:
            ErrorCode: error code
        """
        logger.warn("move_joint_online is not achieved in NmxRdkRobot Robot! ")
        return ErrorCode("failed!", -1)

    def move_joint_offline(
        self,
        target_jnt_pos: Types.Array,
        arm_id: int = None,
    ) -> ErrorCode:
        """Move whole robot joint to the target joint position with offline control.

        Args:
            target_jnt_pos (Types.Array): array of target joint position
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.

        Returns:
            ErrorCode: error code
        """
        logger.warn("move_joint_offline is not achieved in NmxRdkRobot Robot! ")

    def move_line_online(
        self,
        target: Types.Array,
        target_vel: float = 0.05,
        target_acc: float = 0.5,
        prefer_jnt_pos: Types.Array = None,
        freq: int = None,
        arm_id: int = None,
        follow: bool = True,
    ) -> ErrorCode:
        """Move tcp to the target position with online control in line Cartesian space.

        Args:
            target (Types.Array): array of target tcp position [x, y, z, qw, qx, qy, qz] or [x,y,z,rx,ry,rz]
            target_v (float): float of target tcp velocity
            target_a (float): float of target tcp acceleration
            target_w (float): float of target tcp angular velocity
            target_dw (float): float of target tcp angular acceleration
            prefer_jnt_pos (Types.Array): array of prefer joint position in tcp moving
            freq (int): frequency of the control loop
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.

        Returns:
            ErrorCode: error code
        """
        if len(target) == 7:
            xyz = target[:3]
            quat_wxyz = target[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            r = R.from_quat(quat_xyzw).as_euler("xyz")
            target_tcp_pose = np.concatenate([xyz, r])
        else:
            target_tcp_pose = target

        self.controller[0]["arm"].move_line(
            tcp=target_tcp_pose,
            vel=target_vel,
            acc=target_acc,
            blocking=False,
            follow=follow,
        )
        return ErrorCode("success!", 0)

    def move_line_offline(
        self,
        target: Types.Array,
        max_v: float = 0.1,
        max_a: float = 1.6,
        max_w: float = None,
        max_dw: float = None,
        prefer_jnt_pos: Types.Array = None,
        arm_id: int = None,
        block: bool = True,
    ) -> ErrorCode:
        """Move tcp to the target position with offline control in line Cartesian space.

        Args:
            target (Types.Array): array of target tcp position
            max_v (float): float of max tcp velocity
            max_a (float): float of max tcp acceleration
            max_w (float): float of max tcp angular velocity
            max_dw (float): float of max tcp angular acceleration
            prefer_jnt_pos (Types.Array): array of prefer joint position in tcp moving
            arm_id (int): arm id, if dual arm, 0 for left, 1 for right, None for all. If single arm, default 0.
            block (bool): bool of whether to use block mode.

        Returns:
            ErrorCode: error code
        """
        if len(target) == 7:
            xyz = target[:3]
            quat_wxyz = target[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            r = R.from_quat(quat_xyzw).as_euler("xyz")
            target_tcp_pose = np.concatenate([xyz, r])
        else:
            target_tcp_pose = target

        self.controller[0]["arm"].move_line(
            tcp=target_tcp_pose, vel=max_v, acc=max_a, blocking=block, follow=False
        )

    def move_ptp(
        self,
        target: Types.Array,
        waypoints: list,
        prefer_jnt_pos: Types.Array,
        max_jnt_vel: Types.Array,
        max_jnt_acc: Types.Array,
        arm_id: int = None,
    ) -> ErrorCode:
        """Move tcp to the target position with online control in joint space.

        Args:
            target (Types.Array): array of target tcp position
            waypoints (list): list of waypoints for the tcp moving
            prefer_jnt_pos (Types.Array): array of prefer joint position in tcp moving
            max_jnt_vel (Types.Array): array of max joint velocity
            max_jnt_acc (Types.Array): array of max joint acceleration

        Returns:
            ErrorCode: error code
        """
        logger.warn("move_ptp is not achieved in NmxRdkRobot Robot! ")
        return ErrorCode("failed!", -1)

    def get_flange_pose(self, arm_id=None) -> Types.Array:
        """Get the current flange pose of the robot.

        Returns:
            flange_pose: array of flange pose
        """
        # TODO (chenxi&danqing): add flange pose to sdk, current calib is get_tcp_pose
        tcp_pose = self.get_tcp_pose()
        flange_pose = pose_to_7D(
            pose_to_4x4(tcp_pose).dot(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.16], [0, 0, 0, 1]])
            )
        )  # TODO(wemin.zhou): make magic number -0.16 can be get automatically
        return flange_pose

    def get_tcp_pose(self, arm_id=None) -> Types.Array:
        """Get the current tcp pose of the robot.

        Returns:
            tcp_pose: array of tcp pose
        """
        msg = self.controller[0]["arm"].get_tcp_pose()
        return np.array(msg.tcp_pose, dtype=np.float32)

    def get_joint_pos(self, arm_id=None) -> Types.Array:
        """Get the current joint position of the robot.

        Returns:
            joint_pos: array of joint position
        """
        msg = self.controller[0]["arm"].get_joint_pose()
        joint = np.array(msg.joint_pose, dtype=np.float32)
        return joint

    def get_camera_pose(self, camera_id: int) -> Types.Array:
        """Get the current pose of the camera.

        Args:
            camera_id (int): camera id

        Returns:
            camera_pose: array of camera pose
        """
        logger.warn("get_camera_pose is not achieved in NmxRdkRobot Robot! ")
        return np.zeros(7, dtype=np.float32)

    def enable(self, arm_id: int = None) -> ErrorCode:
        """Enable robot.

        Args:
            arm_id (int): arm id

        Returns:
            ErrorCode: error code
        """
        self.controller[0]["arm"].enable()
        return ErrorCode("success!", 0)

    def stop(self, arm_id: int = None) -> ErrorCode:
        """Enable robot.

        Args:
            arm_id (int): arm id

        Returns:
            ErrorCode: error code
        """
        self.controller[0]["arm"].stop()
        return ErrorCode("success!", 0)

    def is_operational(self, arm_id: int = None) -> bool:
        """Check if arm is operational.

        Returns:
            status: True/False
        """
        logger.warn("is_operational is not achieved in NmxRdkRobot Robot! ")
        return True

    """ Force control, only enable for supported robots"""

    def cali_force_sensor(
        self, data_collection_time: float = None, arm_id=None
    ) -> ErrorCode:
        """Calibrate the force sensor.

        Args:
            data_collection_time (float): time for data collection

        Returns:
            ErrorCode: error code
        """
        self.controller[0]["arm"].cali_force_sensor()

    def relative_contact(
        self,
        force: int,
        contact_dir: int,
        max_v: Optional[float] = 0.05,
        max_time: Optional[float] = 10,
    ):
        """Relative contact control. Force and position cannot share the same dimension.

        Args:
            force: int, directional force along the target dimension
            contact_dir: int, the axis to use force control,
                         0-x, 1-y, 2-z, 3-rx, 4-ry, 5-rz
            vel_ratio: int, velocity ratio, in [1, 100].
            max_time: float, maximum execution time.
        """
        start_forces_norm = np.linalg.norm(self.get_tcp_force_value()[:3])
        current_pose = np.array(self.get_tcp_pose())
        max_tcp_vel = self.get_max_tcp_vel()  # TODO add get_max_tcp_vel function

        # move robot in tool frame
        trans_matrix = np.identity(4)
        trans = np.zeros(3)
        trans[contact_dir] = max_tcp_vel * max_time
        trans_matrix[:3, 3] = trans
        target_7d = pose_to_7D(pose_to_4x4(current_pose) @ trans_matrix)

        start_time = time.time()
        self.move_line_offline(
            target=target_7d,
            max_v=max_v,
            blocking=False,
        )

        result = False
        while time.time() - start_time < max_time:
            forces = self.get_tcp_force_value()[:3]
            if abs(np.linalg.norm(forces) - start_forces_norm) > force:
                result = True
                break
            time.sleep(0.001)

        self.stop(arm_id=0)
        return result

    def world_contact(
        self,
        force: int,
        contact_dir: int,
        max_v: Optional[float] = 0.05,
        max_time: Optional[float] = 10,
    ):
        """Relative contact control. Force and position cannot share the same dimension.

        Args:
            force: int, directional force along the target dimension
            contact_dir: int, the axis to use force control,
                         0-x, 1-y, 2-z, 3-rx, 4-ry, 5-rz
            max_v: float, velocity in m/s.
            max_time: float, maximum execution time.
        """
        start_forces_norm = np.linalg.norm(self.get_tcp_force_value()[:3])
        current_pose = np.array(self.get_tcp_pose())
        max_tcp_vel = self.get_max_tcp_vel()  # TODO add get_max_tcp_vel function

        # move robot in world frame
        trans = np.zeros(3)
        trans[contact_dir] = max_tcp_vel * max_time
        target_7d = current_pose.copy()
        target_7d[2] -= max_tcp_vel * max_time

        start_time = time.time()
        self.move_line_offline(
            target=target_7d,
            max_v=max_v,
            blocking=False,
        )

        result = False
        while (time.time() - start_time) < max_time:
            forces = self.get_tcp_force_value()[:3]
            if abs(np.linalg.norm(forces) - start_forces_norm) > force:
                result = True
                break
            time.sleep(0.001)

        self.stop(arm_id=0)
        return result

    def move_compliant(
        self,
        waypoints=None,
        max_vel=0.05,
        max_acc=0.1,
        max_wrench=[20, 20, 20, 6, 6, 6],
        stiff_scale=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        time_out=10,
        arm_id: int = None,
        tcp=False,
        max_force=8,
        target_joint_vel=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        target_joint_acc=[30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    ) -> ErrorCode:
        """compliant move

        Args:
            waypoints (_type_): move waypoints
            target_pose_left (Types.Array, (x,y,z, w,x,y,z)): _description_. Defaults to None.
            target_pose_right (Types.Array,  (x,y,z, w,x,y,z)): _description_. Defaults to None.
            max_vel (float, optional): _description_. Defaults to 0.1.
            max_acc (float, optional): _description_. Defaults to 0.1.
            max_wrench (list, optional): _description_. Defaults to [20, 20, 20, 6, 6, 6].
            stiff_scale (list, optional): _description_. Defaults to [0.5, 0.5, 0.5, 0.5, 0.5, 0.5].
            arm_id (int, optional): _description_. Defaults to None.

        Returns:
            ErrorCode: _description_
        """
        joint_delta = 0.01
        close_gripper_buffer = 0.01
        liftup_vel = 0.02
        raw_jnt_vel = [180.0, 180.0, 225, 225, 225, 225]
        raw_jnt_acc = [600.0, 600.0, 600.0, 600.0, 600.0, 600.0]

        if waypoints is None:
            if tcp is False:
                exec_result = self.world_contact(
                    force=max_force,
                    contact_dir=2,
                    max_v=max_vel,
                    max_time=time_out,
                )
            else:
                exec_result = self.relative_contact(
                    force=max_force,
                    contact_dir=2,
                    max_v=max_vel,
                    max_time=time_out,
                )

            if exec_result is False:
                logger.error("Move compliant failed, cannot execute contact control.")
                return False

            current_tcp_pose = self.get_tcp_pose(arm_id=0)
            target_tcp_pose = pose_to_7D(
                pose_to_4x4(current_tcp_pose)
                @ np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -close_gripper_buffer],
                        [0, 0, 0, 1],
                    ]
                )
            )
            self.move_line_offline(
                target=target_tcp_pose,
                max_v=liftup_vel,
            )
            return exec_result
        else:
            start_forces_norm = np.linalg.norm(self.get_tcp_force_value()[:3])
            start_time = time.time()
            target_joint_pos = np.array(waypoints[0])
            self.move_arm_joint_online(
                target_jnt_pos=target_joint_pos,
                target_jnt_vel=target_joint_vel,
                target_jnt_acc=target_joint_acc,
                follow=False,
            )

            result = False
            while (time.time() - start_time) < time_out:
                forces = self.get_tcp_force_value()[:3]
                current_joints = self.get_joint_pos(arm_id=0)
                if (
                    abs(np.linalg.norm(forces) - start_forces_norm) > max_force
                    or np.linalg.norm(current_joints - target_joint_pos) < joint_delta
                ):
                    logger.info(
                        f"Move compliant ends, force: {np.linalg.norm(forces) > max_force}, joint: {np.linalg.norm(current_joints - target_joint_pos) < joint_delta}"
                    )
                    result = True
                    break
                time.sleep(0.001)

            self.stop(arm_id=0)

            if close_gripper_buffer != 0:
                current_tcp_pose = self.get_tcp_pose(arm_id=0)
                target_tcp_pose = pose_to_7D(
                    pose_to_4x4(current_tcp_pose)
                    @ np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, -close_gripper_buffer],
                            [0, 0, 0, 1],
                        ]
                    )
                )
                self.move_line_offline(
                    target=target_tcp_pose,
                    max_v=liftup_vel,
                )

            if result is False:
                return ErrorCode("failed!", -1)
            else:
                return ErrorCode("success!", -1)

    def contact(
        self,
        target_pose: np.ndarray,
        force: int,
        contact_dir: int,
        vel_ratio: Optional[int] = 0.02,
        max_time: Optional[float] = 2,
    ) -> ErrorCode:
        self.cali_force_sensor()  # Calibration of force sensors
        start_time = time.time()
        self.move_line_online(
            target=target_pose,
            target_vel=vel_ratio,
            follow=False,
        )
        reach_max_force = False
        while time.time() - start_time < max_time:
            forces = np.array(self.get_tcp_force_value())[:3]
            if np.linalg.norm(forces) > force:
                reach_max_force = True
                logger.info(f"Reach max force: {forces}")
                break
            time.sleep(0.0003)
        if not reach_max_force:
            logger.info(
                f"Contact end without reach max force, time: {time.time() - start_time}/{max_time}"
            )
        self.stop()

    def compliant_grasp(
        self,
        gripSpeed: float,
        gripWidth: float,
        maxGripForce: float,
        compliantTCPAxis: str,
        time_out=1.0,
        arm_id: int = None,
    ) -> ErrorCode:
        logger.warn("compliant_grasp is not really achieved in NmxRdkRobot Robot! ")
        return ErrorCode("failed!", -1)
