import numpy as np
from shapely import Point, LineString
from typing import Tuple, List, Union
from copy import deepcopy

from .segment import *

PosePt2D = Tuple[float, float, float]  # (x,y,yaw) values


class PIDTracker:
    """
    Path tracker using PID (actually just P)
    """

    def __init__(self) -> None:
        ##############
        # Parameters #
        ##############
        self.look_ahead_distance = 0.1  # metres
        self.cross_track_error_bounds = 0.01  # metres
        self.orientation_error_bounds = 10 / 180 * np.pi  # radians
        self.max_cte = 0.3  # metres, as given by
        self.cte_to_oe_ratio = 0.5  # 1: ignores oe, 0, ignores cte
        self.K_omega = 2.0  # propotional gain for turning speed
        self.v_max = 0.1  # m/s, maximum allowable forward speed
        self.omega_max = 0.2  # rad/s
        self.k_v = 0.5  # gain for reducing linear speed when far from track
        self.k_w = 5.0  # steepness of the blending transition
        self.d_blend = 0.5  # blending threshold for cross-track error

        ##########################
        # Path progress variable #
        ##########################
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
        for full_path_idx, seg in enumerate(path):
            path_travelled_tmp += seg.length
            if path_travelled_tmp >= path_travelled:
                percent_of_segment_travelled = (
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
        if nearest_seg_idx + 1 < len(path):
            remaining_path = [path_seg_tmp] + path[nearest_seg_idx + 1 :]
        else:
            remaining_path = [path_seg_tmp]
        return remaining_path
    
    def is_valid_point(
        self, pt: Point, segment: Segment, robot_position: Point, robot_yaw: float
    ) -> bool:
        """
        Determine if an intersection point is valid based on the segment's direction and dot product.

        Args:
            pt (Point): The intersection point.
            segment (Segment): The current path segment.
            robot_position (Point): The robot's current position.
            robot_yaw (float): The robot's current yaw

        Returns:
            bool: True if the point is valid, False otherwise.
        """
        path_vector = np.array(
            [
                segment.end.x - segment.start.x,
                segment.end.y - segment.start.y,
            ]
        )
        robot_heading = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        robot_to_pt = np.array([pt.x - robot_position.x, pt.y - robot_position.y])

        heading_path_dot = np.dot(robot_heading, path_vector)
        heading_pt_dot = np.dot(robot_heading, robot_to_pt)

        if segment.direction == 1: # forward
            if heading_path_dot > 0 and heading_pt_dot > 0:
                return True
        elif segment.direction == -1: # backward
            if heading_path_dot < 0 and heading_pt_dot < 0:
                return True
        return False

    def calculate_look_ahead_point_on_path(
        self,
        current_pose: PosePt2D,
        path_remaining: List[Segment],
        look_ahead_distance: float,
    ) -> Union[Tuple[Tuple[float, float], float, float, int], Tuple[None, None, None, None]]:
        """
        Calculate the look-ahead point on the path.

        Args:
            current_pose (PosePt2D): The robot's current pose.
            segments (List[Segment]): List of path segments.
            look_ahead_distance (float): The look-ahead distance.

        Returns:
            Tuple[Tuple[float, float], float, float, Direction]:
                - Look-ahead point coordinates (x, y)
                - Cross-track error (cte)
                - Orientation error (oe)
                - Direction of the segment (1 for forward or -1 for backward)
        """
        robot_position = Point(current_pose[0], current_pose[1])
        look_ahead_circle = robot_position.buffer(look_ahead_distance)
        dead_zone = robot_position.buffer(look_ahead_circle * 0.1) # TODO: when brain power, think if it's necessary


    def calculate_velocities(
        self, cte: float, oe: float, direction: int
    ) -> Tuple[float, float]:
        """
        Calculates velocities from errors using pre-defined gains

        Args:
            cte (float): Cross track error
            oe (float): Orientation error
            direction (int): 1 for forward gear, -1 for reverse gear
        Returns:
            Tuple[float, float]: command linear and angular velocity
        """
        if direction == 1:
            desired_velocity = self.v_max
            velocity_sign = 1
        else:
            desired_velocity = -self.v_max
            velocity_sign = -1

        # Orientation adjustment
        if not -self.orientation_error_bounds < oe < self.orientation_error_bounds:
            omega = self.K_omega * oe
            omega = np.clip(omega, -self.omega_max, self.omega_max)
            v = 0.0
        else:
            w = 1 / (1 / np.exp(-self.k_w * cte - self.d_blend))
            v_move = self.v_max * np.exp(-self.k_v * cte)
            v_move *= velocity_sign  # Adjust based on direction
            omega = self.K_omega * oe
            omega = np.clip(omega, -self.omega_max, self.omega_max)
            v = w * v_move + (1 - w) * desired_velocity

        return v, omega

    def track(self, current_pose: PosePt2D, path: List[Segment]) -> Tuple[float, float]:
        """
        Tracks the path of the robot using an PID estimator

        Args:
            current_pose (PosePt2D): Current position of the robot
            path (List[Segment]): List of desired positions, with direction of motion
        Returns:
            Tuple[float, float]: command linear and angular velocity
        """

        path_remaining = self.get_remaining_path(current_pose, path)
        look_ahead_point, cte, oe, direction = self.calculate_look_ahead_point_on_path(
            current_pose, path_remaining, self.look_ahead_distance
        )
        # TODO: add failure case

        command_v, command_omega = self.calculate_velocities(cte, oe, direction)

        return command_v, command_omega
