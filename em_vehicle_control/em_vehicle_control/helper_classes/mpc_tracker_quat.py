import numpy as np
import cvxpy as cp
from shapely import Point, LineString
from typing import Tuple, List, Union
from copy import deepcopy

from .path_planner import PathPointDatum
from .segment import * 

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values


class MPCTracker:
    """
    Path tracker using MPC
    """

    def __init__(self) -> None:
        # time parameters:
        self.N = 5
        self.dt = 0.15  # follows tracker node rate

        # state limits
        self.v_max = 0.2  # m/s
        self.omega_max = 1  # rad/s
        self.delta_v_max = 0.1  # rate of change limit for linear speed
        self.delta_omega_max = 0.3  # rate of change limit for angular speed

        # cost function weights
        self.q_p = 5.0  # position error weight
        self.d_p = 1.0  # discount factor for position errors
        self.q_theta = 1.0  # orientation error weight
        self.d_theta = 1.0  # discount factor for orientation errors
        self.r_v = 2.0  # linear speed control effort
        self.r_omega = 2.0  # angular speed control effort
        self.r_v_smooth = 0.1  # smoothing linear velocity command
        self.r_omega_smooth = 0.1  # smoothing angular velocity command
        self.q_p_terminal = 10.0  # terminal error cost
        self.q_theta_terminal = 20.0  # terminal error cost

        # other parameters
        self.nominal_speed = self.v_max

        # Progress variable
        self.s = None  # percentage progress along the path
        self.path_length = None  # total length of the path

    def compute_path_length(self, path: List[Segment]) -> None:
        """
        Computes cumulative distances along the path and stores the total path length.

        Args:
            path(List[Segment]): List of desired positions, with direction of motion
        """
        self.path_length = 0
        for seg in path:
            self.path_length += seg.length

    def find_nearest_point_on_path(
        self, current_pose: PosePt2D, path: List[Segment], s: float
    ) -> Tuple[int, float, Tuple[float, float]]:
        """
        Calculates the closest point on the path segment from s% to 100% of the path.

        Args:
            current_pose(PosePt2D): Current position of the robot
            path(List[Segment]): List of desired positions, with direction of motion
            s(float) : progress variable on the path
        Returns:
            (Tuple[int, float, Tuple[float, float]]): Index of segment, percentage length of segment,
            (x,y) position of nearest point
        """
        path_travelled = self.path_length * s
        path_travelled_tmp = 0
        # first get path travelled from s%
        for full_path_idx, seg in enumerate(path):
            path_travelled_tmp += seg.length
            if seg.length == 0:
                # prevents division by 0
                percent_of_segment_travelled = 1
                break
            if path_travelled_tmp >= path_travelled:
                percent_of_segment_travelled = 1 - (
                    path_travelled_tmp - path_travelled
                ) / seg.length
                break
        path_seg_tmp = deepcopy(path[full_path_idx])
        path_seg_tmp.start.x = (
            path[full_path_idx].end.x - path[full_path_idx].start.x
        ) * percent_of_segment_travelled + path[full_path_idx].start.x
        path_seg_tmp.start.y = (
            path[full_path_idx].end.y - path[full_path_idx].start.y
        ) * percent_of_segment_travelled + path[full_path_idx].start.y
        path_seg_tmp.re_init()
        if full_path_idx + 1 < len(path):
            path_remaining = [path_seg_tmp] + path[full_path_idx + 1 :]
        else:
            path_remaining = [path_seg_tmp]
        min_distance = float("inf")
        robot_position = Point(current_pose[0], current_pose[1])
        # in the remaining path, find closest point
        for rem_seg_idx, seg in enumerate(path_remaining):
            nearest_pt_tmp = seg.line.interpolate(seg.line.project(robot_position))
            distance = robot_position.distance(nearest_pt_tmp)
            if distance < min_distance:
                min_distance = distance
                nearest_pt = nearest_pt_tmp
                nearest_seg_idx = rem_seg_idx + full_path_idx
        dx_tmp = path[nearest_seg_idx].end.x - path[nearest_seg_idx].start.x
        dy_tmp = path[nearest_seg_idx].end.y - path[nearest_seg_idx].start.y
        px_tmp = nearest_pt.x - path[nearest_seg_idx].start.x
        py_tmp = nearest_pt.y - path[nearest_seg_idx].start.y
        if (dx_tmp**2 + dy_tmp**2) == 0:
            # prevents division by 0
            percent_of_nearest_segment = 1
        else:
            percent_of_nearest_segment = np.sqrt(
                (px_tmp**2 + py_tmp**2) / (dx_tmp**2 + dy_tmp**2)
            )
        return nearest_seg_idx, percent_of_nearest_segment, (nearest_pt.x, nearest_pt.y)

    def update_progress_variable(
        self,
        path: List[Segment],
        nearest_seg_idx: int,
        percent_of_nearest_segment: float,
    ) -> float:
        """
        Updates or computes progress variable s, which contains the percentage the robot is along the path

        Args:
            path(List[Segment]): List of desired positions, with direction of motion
            nearest_seg_idx(int): segment index closest to the current position
            percent_of_nearest_segment(float): percentage distance along the segment to the nearest point to the robot

        Returns:
            (float): updated path progress variable
        """
        length_travelled = 0
        for seg in path[0:nearest_seg_idx]:
            length_travelled += seg.length
        length_travelled += path[nearest_seg_idx].length * percent_of_nearest_segment

        new_s = length_travelled / self.path_length
        if self.s is not None and new_s < self.s:
            print("WARNING: s decreasing", flush=True)
        return new_s

    def get_remaining_path(
        self, current_pose: PosePt2D, path: List[Segment]
    ) -> List[Segment]:
        """
        Updates or computes progress variable s, which contains the percentage the robot is along the path

        Args:
            current_pose(PosePt2D): Current position of the robot
            path(List[Segment]): List of desired positions, with direction of motion
        Returns:
            List[Segment]: Path remaining after the current pose, and not before path progress
        """
        if self.s is None:
            # get nearest position to the path, and compute the value s
            nearest_seg_idx, percent_of_nearest_segment, nearest_point = (
                self.find_nearest_point_on_path(current_pose, path, 0.0)
            )
        else:
            # restrict path to s onward, and then compute nearest position to path
            nearest_seg_idx, percent_of_nearest_segment, nearest_point = (
                self.find_nearest_point_on_path(current_pose, path, self.s)
            )
        self.s = self.update_progress_variable(
            path, nearest_seg_idx, percent_of_nearest_segment
        )
        path_seg_tmp = deepcopy(path[nearest_seg_idx])
        path_seg_tmp.start.x = nearest_point[0]
        path_seg_tmp.start.y = nearest_point[1]
        path_seg_tmp.re_init()
        if nearest_seg_idx + 1 < len(path):
            remaining_path = [path_seg_tmp] + path[nearest_seg_idx + 1 :]
        else:
            remaining_path = [path_seg_tmp]
        return remaining_path

    def calculate_theta(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        current_yaw: float,
        direction: List[int],
    ) -> np.ndarray[float]:
        """
        Computes the angle between the current position and the next positon.
        If direction is 0 (stop waypoint), either follow the previous or following theta.

        Args:
            x(np.ndarray[float]): x coordinates
            y(np.ndarray[float]): y coordinates
            current_yaw(float): Current yaw of the robot
            direction(List[int]): direction, either 1 (forward), -1 (backwards) or 0 (stop)
        Returns:
            (np.ndarray[float]): Array of thetas
        """
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)
        theta = np.zeros(len(x))
        theta[0] = current_yaw
        for i in range(1, len(theta)):
            if direction[i - 1] == 1:
                desired_theta = angles[i - 1]
            elif direction[i - 1] == -1:
                desired_theta = angles[i - 1] + np.pi
            else:
                desired_theta = theta[i - 1]
            theta[i] = (desired_theta + np.pi) % (2 * np.pi) - np.pi
        return theta

    def get_reference_path(
        self, current_pose: PosePt2D, path: List[Segment]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Finds where the robot is closest to the path, interpolates the next N poses.

        Args:
            current_pose (PosePt2D): Current position of the robot
            path (List[Segment]): List of desired positions, with direction of motion
        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]): Desired x, desired y,
            desired yaw and desired direction of the robot of the next N timesteps.
        """
        if self.path_length is None:
            self.compute_path_length(path)
        remaining_spatial_path = self.get_remaining_path(current_pose, path)
        nominal_dlength = self.nominal_speed * self.dt

        x_ref = np.zeros(self.N + 1)
        y_ref = np.zeros(self.N + 1)
        theta_ref = np.zeros(self.N + 1)
        direction_ref = [0] * (self.N + 1)

        x_ref[0], y_ref[0] = current_pose[0], current_pose[1]
        for k in range(self.N):
            dlenghth_remaining = nominal_dlength
            direction_change_flag = False
            for seg_idx, seg in enumerate(remaining_spatial_path):
                if direction_change_flag:
                    direction_change_flag = False
                    next_x = seg.start.x
                    next_y = seg.start.y
                    next_dir = 0  # stop waypoint
                    break
                if seg.length >= dlenghth_remaining:
                    # still on the same path segment
                    next_x = seg.start.x + dlenghth_remaining / seg.length * (
                        seg.end.x - seg.start.x
                    )
                    next_y = seg.start.y + dlenghth_remaining / seg.length * (
                        seg.end.y - seg.start.y
                    )
                    next_dir = seg.direction
                    break
                else:
                    # first check if direction changes
                    if seg_idx + 1 >= len(remaining_spatial_path):
                        next_x = seg.end.x
                        next_y = seg.end.y
                        next_dir = seg.direction
                    elif seg.direction != remaining_spatial_path[seg_idx + 1].direction:
                        next_x = seg.end.x
                        next_y = seg.end.y
                        next_dir = seg.direction
                        direction_change_flag = True
                        seg_idx += 1  # start on the next segment
                        break
                    dlenghth_remaining -= seg.length
            tmp_seg = deepcopy(seg)
            tmp_seg.start.x = (
                next_x  # if direction change, current segment must remains the same
            )
            tmp_seg.start.y = next_y
            tmp_seg.re_init()
            if seg_idx + 1 < len(remaining_spatial_path):
                remaining_spatial_path = [tmp_seg] + remaining_spatial_path[
                    seg_idx + 1 :
                ]
            else:
                remaining_spatial_path = [tmp_seg]
            x_ref[k + 1] = next_x
            y_ref[k + 1] = next_y
            direction_ref[k + 1] = next_dir
            if k == 0:
                direction_ref[0] = next_dir

        theta_ref = self.calculate_theta(x_ref, y_ref, current_pose[2], direction_ref)

        return x_ref, y_ref, theta_ref, direction_ref
    
    def yaw_to_quat(self, yaw: Union[float, np.ndarray]) -> Tuple[float, float]:
        """
        Converts yaw to quaternions in a 2D plane

        Args:
            yaw Union[float, np.ndarray]: yaw angle in radians, either as scalar or vector
        Returns:
            Tuple[float, float]: q_w and q_z respectively
        """
        q_w = np.cos(yaw / 2)
        q_z = np.sin(yaw / 2)
        return q_w, q_z
    

    def track(self, current_pose: PosePt2D, path: List[Segment]) -> Tuple[float, float]:
        """
        Tracks the path of the robot using an MPC solver

        Args:
            current_pose (PosePt2D): Current position of the robot
            path (List[Segment]): List of desired positions, with direction of motion
        Returns:
            Tuple[float, float]: command linear and angular velocity
        """
        # TODO: Reset if get new path

        x_ref, y_ref, theta_ref, direction_ref = self.get_reference_path(
            current_pose, path
        )
        q_w_ref, q_z_ref = self.yaw_to_quat(theta_ref)

        print("x_ref ", x_ref, flush=True) 
        print("y_ref ", y_ref, flush=True)
        print("theta ", theta_ref)

        # state var
        x = cp.Variable(self.N + 1)
        y = cp.Variable(self.N + 1)
        q_w = cp.Variable(self.N + 1) # scalar part of quaternion
        q_z = cp.Variable(self.N + 1) # z-component of quaternion is only needed for 2d 

        # control var
        v = cp.Variable(self.N)
        omega = cp.Variable(self.N)

        # define objective function
        position_error = 0
        orientation_error = 0
        control_effort = 0
        control_smoothness = 0
        
        objective = 0
        constraints = []

        for k in range(self.N + 1):
            position_error += (
                self.d_p**k * self.q_p * (x[k] - x_ref[k]) ** 2
                + self.d_p**k * self.q_p * (y[k] - y_ref[k]) ** 2
            )

            orientation_error += (
                self.d_theta**k * self.q_theta * cp.sum_squares(q_w[k] - q_w_ref[k]) + cp.sum_squares(q_z[k] - q_z_ref[k])
            )

            if k == self.N:
                position_error += self.q_p_terminal * (
                    cp.square(x[k] - x_ref[k]) + cp.square(y[k] - y_ref[k])
                )
                orientation_error += self.q_theta_terminal * cp.sum_squares(q_w[k] - q_w_ref[k]) + cp.sum_squares(q_z[k] - q_z_ref[k])

        control_effort += self.r_v * cp.sum_squares(v) + self.r_omega * cp.sum_squares(
            omega
        )

        for k in range(self.N - 1):
            control_smoothness += self.r_v_smooth * cp.square(v[k + 1] - v[k])
            control_smoothness += self.r_omega_smooth * cp.square(
                omega[k + 1] - omega[k]
            )

        objective = (
            position_error + orientation_error + control_effort + control_smoothness
        )

        # initial conditions
        q_w0, q_z0 = self.yaw_to_quat(current_pose[2])
        initial_conditions = (
            [x[0] == current_pose[0]]
            + [y[0] == current_pose[1]]
            + [q_w[0] == q_w0]
            + [q_z[0] == q_z0]
        )

        # system dynamics
        system_dynamics = []
        for k in range(self.N):
            cos_theta_ref = np.cos(theta_ref[k])
            sin_theta_ref = np.sin(theta_ref[k])
            sin_half_w_dt = omega[k] * self.dt / 2
            system_dynamics += [x[k + 1] == x[k] + v[k] * cos_theta_ref * self.dt]
            system_dynamics += [y[k + 1] == y[k] + v[k] * sin_theta_ref * self.dt]
            system_dynamics += [q_w[k+1] == q_w[k] - sin_half_w_dt * q_z[k]] # WARNING NOT CONVEX
            system_dynamics += [q_z[k+1] == q_z[k] + sin_half_w_dt * q_w[k]]

        # velocity constraints
        velocity_limits = []
        velocity_limits += [cp.abs(v) <= self.v_max]
        velocity_limits += [cp.abs(omega) <= self.omega_max]

        # direction constraints
        direction_constraints = []
        for k in range(self.N):
            if direction_ref[k] == 1:  # forward
                direction_constraints += [v[k] >= 0]
            elif direction_ref[k] == -1:  # backward
                direction_constraints += [v[k] <= 0]
            elif direction_ref[k] == 0:  # stop
                direction_constraints += [cp.abs(v[k]) <= self.v_max * 0.1]

        # rate of change constraints
        roc_constraints = []
        roc_constraints += [
            cp.abs(v[k + 1] - v[k]) <= self.delta_v_max for k in range(self.N - 1)
        ]
        roc_constraints += [
            cp.abs(omega[k + 1] - omega[k]) <= self.delta_omega_max
            for k in range(self.N - 1)
        ]

        # quaternion costraint
        quat_constraint = [q_w[k+1]**2 + q_z[k+1]**2 == 1 for k in range(self.N)]

        constraints += (
            initial_conditions
            + system_dynamics
            + velocity_limits
            + direction_constraints
            + roc_constraints
            + quat_constraint
        )

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)

        if prob.status != cp.OPTIMAL:
            # TODO: We still need it to return something
            raise ValueError("Solver did not find an optimal solution.")

        v_command = v.value[0]
        omega_command = omega.value[0]
        # print(v_command, omega_command)

        return v_command, omega_command
    