""" HTC Vive tracker IO and teleop API.
Built upon libsurvive (https://github.com/collabora/libsurvive).

Author: Chenxi Wang
"""

import sys
import json
import time
import pysurvive
import threading
import numpy as np

from transforms3d.quaternions import mat2quat, quat2mat

def color_string(s, color="g"):
    color_map = {"r": 31, "g": 32, "b": 34}
    assert color in color_map
    s = f"\033[{color_map[color]}m{s}\033[0m"
    return s

def diff_pose(pose1, pose2):
    trans_diff = np.linalg.norm(pose1[..., :3, 3] - pose2[..., :3, 3], axis=-1)
    rot_diff = np.matmul(pose1[..., :3, :3], np.swapaxes(pose2, -2, -1)[..., :3, :3])
    rot_diff = np.diagonal(rot_diff, axis1=-2, axis2=-1).sum(axis=-1)
    rot_diff = np.minimum(np.maximum((rot_diff - 1) / 2.0, -1), 1)
    rot_diff = np.arccos(rot_diff)
    return trans_diff, rot_diff

def pose_to_4x4(pose_7d):
    pose_4x4 = np.eye(4)
    pose_4x4[:3, 3] = pose_7d[0:3]
    pose_4x4[:3, :3] = quat2mat(pose_7d[3:7])
    return pose_4x4

def pose_to_7d(pose_4x4):
    trans = pose_4x4[:3, 3]
    rot = mat2quat(pose_4x4[:3, :3])
    pose_7d = np.concatenate([trans, rot], axis=0)
    return pose_7d

def quat_slerp(q1: np.ndarray, q2: np.ndarray, alpha) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions (supports batch operations).
    
    Args:
        q1: First quaternion(s) [..., 4] where last dim is [w, x, y, z]
        q2: Second quaternion(s) [..., 4] where last dim is [w, x, y, z] 
        alpha: Interpolation parameter(s) [0, 1] - scalar or array broadcastable with q1/q2
    
    Returns:
        Interpolated quaternion(s) [..., 4]
        
    Examples:
        # Single interpolation: q1=[4], q2=[4], alpha=scalar -> result=[4]
        # Batch alpha: q1=[4], q2=[4], alpha=[N] -> result=[N, 4] 
        # Full batch: q1=[N,4], q2=[N,4], alpha=[N] -> result=[N, 4]
    """
    q1 = q1 / np.linalg.norm(q1, axis = -1, keepdims = True)
    q2 = q2 / np.linalg.norm(q2, axis = -1, keepdims = True)
    
    dot_product = np.sum(q1 * q2, axis = -1, keepdims = True)
    
    q2 = np.where(dot_product < 0.0, -q2, q2)
    dot_product = np.abs(dot_product)
    
    omega = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    alpha = np.asarray(alpha)
    alpha_expanded = alpha[..., np.newaxis] if alpha.ndim > 0 else alpha
    
    sin_omega = np.sin(omega)
    if np.abs(sin_omega) < 1e-6:
        sin_omega = 1e-6
    linear_result = (1 - alpha_expanded) * q1 + alpha_expanded * q2
    slerp_result = (np.sin((1 - alpha_expanded) * omega) * q1 + 
                   np.sin(alpha_expanded * omega) * q2) / sin_omega
    
    return np.where(omega < 1e-6, linear_result, slerp_result)

def adaptive_interpolate_trajectories(tgt_pose, ref_pose, trans_trunc=0.005, trans_scale=0.01, rot_trunc=np.pi/200, rot_scale=np.pi/50, temprature=5):
    trans_diff, rot_diff = diff_pose(tgt_pose, ref_pose)
    trans_weight = 1 / (1 + max(0, (trans_diff - trans_trunc) / trans_scale) ** temprature)
    rot_weight = 1 / (1 + max(0, (rot_diff - rot_trunc) / rot_scale) ** temprature)
    rot_weight = min(trans_weight, rot_weight)
    interpolated_pose = pose_interpolation(tgt_pose, ref_pose, trans_weight, rot_weight)
    
    return interpolated_pose

def pose_interpolation(tgt_pose, ref_pose, trans_alpha, rot_alpha):
    tgt_pose = pose_to_7d(tgt_pose)
    ref_pose = pose_to_7d(ref_pose)
    interpolated_trans = trans_alpha * tgt_pose[:3] + (1 - trans_alpha) * ref_pose[:3]
    interpolated_rot = quat_slerp(ref_pose[3:7], tgt_pose[3:7], rot_alpha)
    interpolated_pose = pose_to_4x4(np.concatenate([interpolated_trans, interpolated_rot]))
    return interpolated_pose


class TrackerConverter:
    def __init__(self):
        self._init_robot_pose = None
        self._init_tracker_pose = None
        self._calib_robot_pose = None
        self._calib_tracker_pose = None
        self._delta_tracker_pose = np.eye(4)
        self._curr_tracker_pose = None
        self._cached_tracker_pose = None
        self._last_robot_pose = None
        self._last_tracker_pose = None
        self._trans_thresh = 0.002
        self._rot_thresh = 0.02
        self._is_running = False

    def reset(self, init_robot_pose=None, calib_robot_pose=None, calib_tracker_pose=None):
        self._is_running = False
        self._init_robot_pose = pose_to_4x4(np.asarray(init_robot_pose))
        self._init_tracker_pose = pose_to_4x4(self.tracker_pose)
        self._calib_robot_pose = pose_to_4x4(np.asarray(calib_robot_pose))
        self._calib_tracker_pose = pose_to_4x4(np.asarray(calib_tracker_pose))

        curr_tracker_pose = self._curr_tracker_pose.copy()
        self._cached_tracker_pose = curr_tracker_pose.copy()
        self._last_tracker_pose = curr_tracker_pose.copy()
        self._last_robot_pose = np.asarray(init_robot_pose).copy()

        # align coordinate system
        self._tracker_rot_transform = np.linalg.inv(self._init_tracker_pose[:3, :3]) @ self._calib_tracker_pose[:3, :3]
        self._robot_rot_transform = np.linalg.inv(self._init_robot_pose[:3, :3]) @ self._calib_robot_pose[:3, :3]
        self._init_tracker_pose[:3, :3] = self._calib_tracker_pose[:3, :3]
        self._init_robot_pose[:3, :3] = self._calib_robot_pose[:3, :3]

    def update_tracker_pose(self, curr_tracker_pose, adaptive_interpolation=True):
        if self._last_tracker_pose is not None and adaptive_interpolation:
            curr_tracker_pose = adaptive_interpolate_trajectories(curr_tracker_pose, self._last_tracker_pose)
        self._curr_tracker_pose = curr_tracker_pose.copy()
        self._last_tracker_pose = curr_tracker_pose.copy()

    def pause(self):
        if not self._is_running:
            return
        self._cached_tracker_pose = self._curr_tracker_pose.copy()
        self._is_running = False

    def start(self):
        if self._is_running:
            return
        assert self._cached_tracker_pose is not None
        curr_robot_pose = self.aligned_robot_pose.copy()
        calib_tracker_pose = pose_to_7d(self._calib_tracker_pose @ (np.linalg.inv(self._calib_robot_pose) @ pose_to_4x4(curr_robot_pose)))
        self.reset(curr_robot_pose, curr_robot_pose, calib_tracker_pose)
        self._cached_tracker_pose = None
        self._is_running = True

    @property
    def tracker_pose(self):
        curr_tracker_pose = self._curr_tracker_pose
        if curr_tracker_pose is not None:
            curr_tracker_pose = pose_to_7d(self._curr_tracker_pose)
        return curr_tracker_pose

    @property
    def aligned_robot_pose(self):
        if not self._is_running:
            return self._last_robot_pose
        if self._curr_tracker_pose is None:
            return None
        assert self._tracker_rot_transform is not None
        assert self._init_robot_pose is not None
        assert self._init_tracker_pose is not None
        curr_tracker_pose = self._delta_tracker_pose @ self._curr_tracker_pose
        curr_tracker_pose[:3, :3] = curr_tracker_pose[:3, :3] @ self._tracker_rot_transform
        next_tcp_pose = self._init_robot_pose @ (np.linalg.inv(self._init_tracker_pose) @ curr_tracker_pose)
        next_tcp_pose = pose_interpolation(next_tcp_pose, self._init_robot_pose, trans_alpha=1.5, rot_alpha=1) # TODO: make alpha configurable
        next_tcp_pose[:3, :3] = next_tcp_pose[:3, :3] @ np.linalg.inv(self._robot_rot_transform)
        next_tcp_pose = pose_to_7d(next_tcp_pose)

        if self._last_robot_pose is not None:
            trans_diff, rot_diff = diff_pose(pose_to_4x4(self._last_robot_pose), pose_to_4x4(next_tcp_pose))
            if trans_diff < self._trans_thresh and rot_diff < self._rot_thresh:
                return self._last_robot_pose
        self._last_robot_pose = next_tcp_pose.copy()

        return next_tcp_pose


class HTCViveTracker:
    def __init__(self, serials=None, num_tracker=1, auto_calib_tracker=False, adaptive_interpolation=True):
        self.num_tracker = num_tracker
        self.adaptive_interpolation = adaptive_interpolation
        self.ctx = pysurvive.SimpleContext(sys.argv)

        if isinstance(serials, list) and len(serials) > 0:
            # self.num_tracker will be reset according to tracker serial list
            self.num_tracker = len(serials)
            self.tracker_ids = {s: i for (i, s) in enumerate(serials)}
        else:
            self.tracker_ids = {}
        self.tracker_serials = {}

        self.converters = {}
        for tid in range(self.num_tracker):
            converter = TrackerConverter()
            self.converters[tid] = converter

        self._start_pub()

        success = self._wait_for_signal(timeout=30)
        if not success:
            self.stop()
            raise RuntimeError("Tracker initialization failed")

        if auto_calib_tracker:
            success = self.calibrate_tracker(timeout=30)
            if not success:
                self.stop()
                raise RuntimeError("Tracker initialization failed")

        print(color_string("Vive Tracker is ready to use."))

    def _wait_for_signal(self, step_time=0.01, timeout=30):
        tic = time.time()
        last_status_time = 0
        print(color_string(f"Waiting for {self.num_tracker} tracker(s) to connect (timeout: {timeout}s)...", "b"))
        while True:
            tracker_poses = self.get_tracker_pose()
            init_finished = True
            connected_count = 0
            for tracker_pose in tracker_poses:
                if tracker_pose is None:
                    init_finished = False
                else:
                    connected_count += 1
            if init_finished:
                break
            
            elapsed = time.time() - tic
            # Print status every 2 seconds
            if elapsed - last_status_time >= 2.0:
                connected_serials = [self.tracker_serials.get(i, "?") for i in range(self.num_tracker) if tracker_poses[i] is not None]
                print(f"  [{elapsed:.1f}s] Connected: {connected_count}/{self.num_tracker} {connected_serials}")
                last_status_time = elapsed
            
            if elapsed > timeout:
                connected_tracker_serials = [self.tracker_serials[i] for i in range(self.num_tracker) if tracker_poses[i] is not None]
                print(color_string(f"Failed to get all tracker signals in {timeout}s, {connected_tracker_serials} are connected now", "r"))
                return False

            time.sleep(step_time)

        print(color_string(f"All {self.num_tracker} tracker(s) connected successfully!", "g"))
        return True

    def set_calibration(self, tracker_id, init_robot_pose, calib_robot_pose, calib_tracker_pose):
        assert tracker_id in list(range(self.num_tracker))
        self.converters[tracker_id].reset(init_robot_pose, calib_robot_pose, calib_tracker_pose)

    def set_calibration_from_file(self, init_robot_pose_list, calib_file):
        assert len(init_robot_pose_list) == self.num_tracker
        with open(calib_file, "r") as f:
            calib_dict = json.load(f)

        for i in range(self.num_tracker):
            self.set_calibration(
                i,
                init_robot_pose_list[i],
                calib_dict[str(i)]["calib_robot_pose"],
                calib_dict[str(i)]["calib_tracker_pose"]
            )

    def calibrate_tracker(self, tracker_id=None, check_time=5, step_time=0.1, trans_thresh=0.002, rot_thresh=0.02, timeout=30):
        tracker_poses_window = []
        check_length = int(check_time / step_time)
        input(color_string("Please keep trackers still for calibration. Press 'Enter' when finished."))

        tic = time.time()
        while True:
            if len(tracker_poses_window) >= check_length:
                print(color_string("\nCalbration finished."))
                return True
            if time.time() - tic > timeout:
                print(color_string("\nCalibration unfinished.", "r"))
                return False

            curr_tracker_poses = self.get_tracker_pose(tracker_id)
            if tracker_id is not None:
                curr_tracker_poses = [curr_tracker_poses]
            curr_tracker_poses = [pose_to_4x4(p) for p in curr_tracker_poses]
            curr_tracker_poses = np.stack(curr_tracker_poses, axis=0)
            tracker_poses_window.append(curr_tracker_poses)

            tracker_poses_window_array = np.stack(tracker_poses_window, axis=0)
            trans_delta, rot_delta = diff_pose(tracker_poses_window_array, tracker_poses_window_array[0:1])
            trans_outlier_mask = trans_delta > trans_thresh
            rot_outlier_mask = rot_delta > rot_thresh
            outlier_mask = trans_outlier_mask | rot_outlier_mask
            print(f"Outlier rate: {outlier_mask.mean():10.6f}, maintaining time: {len(tracker_poses_window)*step_time:6.2f}s", end="\r")

            if np.any(outlier_mask):
                tracker_poses_window = []

            time.sleep(step_time)

    def calibrate_robot_tracker_transform(self, calib_robot_pose_list, save_path=None, dist_thresh=0.01, cos_thresh=0.25):
        assert len(calib_robot_pose_list) == self.num_tracker
        def _get_direction_vector(p1, p2, dist_thresh):
            dist = np.linalg.norm(p2 - p1)
            assert dist > dist_thresh, "distance of two points should be larger than 1cm!"
            vec = (p2 - p1) / dist
            return vec

        def _get_coordinate_frame_from_axes(axis_x, axis_y, cos_thresh):
            dot = axis_x @ axis_y
            assert np.abs(dot) < cos_thresh, "axis_x is not vertical enough to axis_y!"
            axis_y -= axis_x * (axis_x @ axis_y)
            axis_y /= np.linalg.norm(axis_y)
            axis_z = np.cross(axis_x, axis_y)
            rotation_matrix = np.c_[axis_x, axis_y, axis_z]
            return rotation_matrix
        
        calib_dict = dict()
        for i in range(self.num_tracker):
            calib_dict[i] = dict()
            calib_dict[i]["calib_robot_pose"] = calib_robot_pose_list[i].tolist()
            input(color_string(f"Please keep tracker {i} (0-indexed) still. Press 'Enter' when finished."))

            tracker_pose_0 = self.get_tracker_pose(i)
            input(color_string(f"Please move tracker {i} (0-indexed) along positive X direction of the aligned robot tcp frame. Press 'Enter' when finished."))
            tracker_pose_1 = self.get_tracker_pose(i)
            axis_x = _get_direction_vector(tracker_pose_0[:3], tracker_pose_1[:3], dist_thresh)

            input(color_string(f"Please move tracker {i} (0-indexed) along positive Y direction of the aligned robot tcp frame. Press 'Enter' when finished."))
            tracker_pose_2 = self.get_tracker_pose(i)
            axis_y = _get_direction_vector(tracker_pose_1[:3], tracker_pose_2[:3], dist_thresh)

            rotation_matrix = _get_coordinate_frame_from_axes(axis_x, axis_y, cos_thresh)
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = tracker_pose_0[:3]
            transform = pose_to_7d(transform)
            calib_dict[i]["calib_tracker_pose"] = transform.tolist()

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(calib_dict, f, indent=4)

        return calib_dict

    def update_data(self, wait_time=0.001):
        first_pose_received = set()
        while self.ctx.Running():
            if self.stopped:
                return
            time.sleep(wait_time)
            updated = self.ctx.NextUpdated()
            if updated:
                pose_obj = updated.Pose()
                tracker_key = str(updated.Name(), 'utf-8')
                if "WM" not in tracker_key:
                    continue
                tracker_serial = str(updated.Serial(), 'utf-8')

                # add tracker serial info if not specified during initialization
                if len(self.tracker_ids) < self.num_tracker:
                    tracker_id = int(tracker_key[2:])
                    if tracker_id >= self.num_tracker:
                        continue
                    if tracker_serial not in self.tracker_ids:
                        self.tracker_ids[tracker_serial] = tracker_id
                        self.tracker_serials[tracker_id] = tracker_serial
                        print(color_string(f"  Discovered tracker {tracker_id}: {tracker_serial} ({tracker_key})", "g"))
                else:
                    if tracker_serial not in self.tracker_ids.keys():
                        continue
                    tracker_id = self.tracker_ids[tracker_serial]

                pose_7d = np.concatenate([pose_obj[0].Pos, pose_obj[0].Rot])
                # Check for valid pose (not all zeros or NaN)
                if np.any(np.isnan(pose_7d)) or np.allclose(pose_7d[:3], 0):
                    continue
                pose_4x4 = pose_to_4x4(pose_7d)
                self.converters[tracker_id].update_tracker_pose(pose_4x4, self.adaptive_interpolation)
                
                if tracker_id not in first_pose_received:
                    first_pose_received.add(tracker_id)
                    print(color_string(f"  First valid pose from tracker {tracker_id} ({tracker_serial}): pos={pose_7d[:3]}", "g"))

    def get_tracker_serial(self, tracker_id=None):
        if tracker_id is not None:
            assert tracker_id in list(range(self.num_tracker))
            tracker_serial = self.tracker_serials[tracker_id]
        else:
            tracker_serial = [self.tracker_serials[i] for i in range(self.num_tracker)]
        return tracker_serial

    def get_tracker_pose(self, tracker_id=None):
        if tracker_id is not None:
            assert tracker_id in list(range(self.num_tracker))
            tracker_pose = self.converters[tracker_id].tracker_pose
        else:
            tracker_pose = [self.converters[tid].tracker_pose for tid in range(self.num_tracker)]
        return tracker_pose
    
    def get_aligned_robot_pose(self, tracker_id=None):
        if tracker_id is not None:
            assert tracker_id in list(range(self.num_tracker))
            robot_pose = self.converters[tracker_id].aligned_robot_pose
        else:
            robot_pose = [self.converters[tid].aligned_robot_pose for tid in range(self.num_tracker)]
        return robot_pose

    def _start_pub(self):
        self.stopped = False
        thread = threading.Thread(target=self.update_data, name="DataUpdater")
        thread.start()

    def pause(self):
        for tid in range(self.num_tracker):
            self.converters[tid].pause()

    def start(self):
        for tid in range(self.num_tracker):
            self.converters[tid].start()

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    tracker = HTCViveTracker(num_tracker=2, auto_calib_tracker=True)
    # tracker = HTCViveTracker(serials=["LH R-EFDCEC3F", "LHR-146E8F33"], auto_calib_tracker=True)
    # tracker = HTCViveTracker(auto_calib_tracker=True)
    print(tracker.tracker_serials)
    input("Press Enter to continue")
    calib_robot_pose = np.array([0.5, 0, 0.3, 0, 0, 1, 0])
    # # uncomment the following line for the first time
    tracker.calibrate_robot_tracker_transform([calib_robot_pose, calib_robot_pose], save_path='calib.json')
    tracker.set_calibration_from_file([calib_robot_pose, calib_robot_pose], 'calib.json')
    tracker.start()

    for i in range(100):
        print(tracker.get_tracker_pose()[0][:3], tracker.get_aligned_robot_pose()[0][:3])
        time.sleep(0.5)

    tracker.stop()