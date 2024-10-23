from scipy.spatial.transform import Rotation as R
import networkx as nx
from shapely import LineString, Point, MultiLineString, MultiPoint
import threading
import numpy as np
from typing import Tuple, Union, List
from enum import Enum

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Transform, Twist
from em_vehicle_control_msgs.msg import Pose2D, Path2D
import tf2_ros
from rclpy.duration import Duration


class Direction(Enum):
    FORWARD = 1
    REVERSE = -1


class Segment:
    def __init__(self, start: Pose2D, end: Pose2D):
        self.start = start
        self.end = end
        self.line = LineString([(start.x, start.y), (end.x, end.y)])
        if start.direction_flag == Pose2D.BACKWARD:
            self.direction = Direction.REVERSE
        else:
            self.direction = Direction.FORWARD

    def __repr__(self) -> str:
        return f"{self.line}, {self.direction}"


class Tracker(Node):
    """
    This node will manage a single robot.
    It will receive the pose of its own robot as quickly as it can,
    and set command vel appropriately based on the path it has stored.
    The path is updated whenever there is a new path message
    """

    def __init__(self):
        super().__init__("tracker")
        self.declare_parameter("robot_name", "unnamed_robot")
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_cmd_vel = self.create_publisher(Twist, f"cmd_vel", 10)

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

        self.path = None
        self.path_msg_lock = threading.Lock()

        self.path_subscription = self.create_subscription(
            Path2D, f"path", self.path_subscription, 1
        )
        self.path_subscription
        self.timer = self.create_timer(0.02, self.control_loop)  # 50Hz

        #########
        # WRITE #
        #########
        self.write = True
        if self.write:
            with open("src/data.txt", "w") as file:
                file.write(
                    f"xR, yR, yaw_R, xA, yA, xB, yB, curr_cte, curr_oe, v, omega \n"
                )

    def path_subscription(self, msg):
        with self.path_msg_lock:
            self.path = msg.poses

    def calculate_cross_track_error(
        self,
        path: LineString,
        robot_position: Point,
    ) -> float:
        """
        path: of the robot
        robot_pose: pose of robot from tf tree
        return: shortest distance between robot and line segment
        """
        projected_distance = path.project(robot_position)
        closest_point_on_path = path.interpolate(projected_distance)
        cross_track_error = robot_position.distance(closest_point_on_path)
        return cross_track_error

    def calculate_look_ahead_based_orientation_error(
        self,
        look_ahead_point: Tuple[float, float],
        robot_position: Point,
        robot_yaw: float,
        direction: Direction,
    ) -> float:
        """
        look_ahead_point: point the robot should move towards
        robot_pose: pose of robot from tf tree
        return: angle between the robot and the look ahead point
        """
        xP, yP = look_ahead_point
        xR, yR = robot_position.coords[0]
        theta_lookahead = np.arctan2(yP - yR, xP - xR)
        orientation_error = theta_lookahead - robot_yaw
        if direction == Direction.REVERSE:
            orientation_error += np.pi
        return (orientation_error + np.pi) % (2 * np.pi) - np.pi

    def create_segments(self, path: List[Pose2D]) -> List[Segment]:
        segments = []
        for i in range(len(path) - 1):
            segments.append(Segment(path[i], path[i + 1]))
        return segments

    def find_nearest_point_on_path(
        self, segments: List[Segment], robot_position: Point
    ) -> Tuple[Point, LineString, Direction]:
        min_distance = float("inf")
        nearest_pt = None
        nearest_direction = Direction.FORWARD  # Default value

        for segment in segments:
            # Compute the nearest point on the segment to the robot
            nearest_point = segment.line.interpolate(
                segment.line.project(robot_position)
            )
            distance = robot_position.distance(nearest_point)

            if distance < min_distance:
                min_distance = distance
                nearest_pt = nearest_point
                segment_containing_nearest_point = segment.line
                nearest_direction = segment.direction

        if nearest_pt is not None:
            print(
                f"Nearest point found at ({nearest_pt.x}, {nearest_pt.y}) on segment with direction {nearest_direction}"
            )

        return nearest_point, segment_containing_nearest_point, nearest_direction

    def get_intersection_points(
        self, segment: Segment, look_ahead_circle: Point
    ) -> List[Point]:
        """
        Retrieve all intersection points between a path segment and the look-ahead circle.

        Args:
            segment (Segment): The current path segment.
            look_ahead_circle (Point): The circular area around the robot's position.

        Returns:
            List[Point]: A list of intersection points.
        """
        intersection = segment.line.intersection(look_ahead_circle)
        intersection_points = []

        if intersection.is_empty:
            return intersection_points

        if isinstance(intersection, Point):
            intersection_points.append(intersection)
        elif isinstance(intersection, MultiPoint):
            intersection_points.extend(list(intersection))
        elif isinstance(intersection, LineString):
            # Intersection is a line segment; take start and end points
            intersection_points.append(Point(intersection.coords[0]))
            intersection_points.append(Point(intersection.coords[-1]))
        elif isinstance(intersection, MultiLineString):
            # Multiple line segments; take start and end points of each
            for linestring in intersection:
                intersection_points.append(Point(linestring.coords[0]))
                intersection_points.append(Point(linestring.coords[-1]))
        else:
            print("ERROR: Unknown geometry when calculating look-ahead points.")

        return intersection_points

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

        if segment.direction == Direction.FORWARD:
            if heading_path_dot > 0 and heading_pt_dot > 0:
                return True
        elif segment.direction == Direction.REVERSE:
            if heading_path_dot < 0 and heading_pt_dot < 0:
                return True
        return False

    def calculate_look_ahead_point_on_path(
        self,
        segments: List[Segment],
        robot_pose: Transform,
        look_ahead_distance: float,
        search_window_ratio: float = 0.25,  # Represents the next 25% of segments
    ) -> Tuple[Tuple[float, float], float, float, Direction]:
        """
        Calculate the look-ahead point on the path considering a search window to handle direction changes.

        Args:
            segments (List[Segment]): List of path segments.
            robot_pose (Transform): The robot's current pose.
            look_ahead_distance (float): The look-ahead distance.
            search_window_ratio (float): The fraction of total segments to include in the search window.

        Returns:
            Tuple[Tuple[float, float], float, float, Direction]:
                - Look-ahead point coordinates (x, y)
                - Cross-track error (cte)
                - Orientation error (oe)
                - Direction of the segment (FORWARD or REVERSE)
        """
        robot_position = Point(robot_pose.translation.x, robot_pose.translation.y)
        look_ahead_circle = robot_position.buffer(look_ahead_distance)
        dead_zone = robot_position.buffer(look_ahead_distance * 0.1)
        total_segments = len(segments)
        search_window_size = max(1, int(total_segments * search_window_ratio))
        look_ahead_point = None
        selected_segment = None
        found_first_valid_point = False
        max_search_idx = -1

        xR = robot_pose.translation.x
        yR = robot_pose.translation.y
        robot_position = Point(xR, yR)
        qx = robot_pose.rotation.x
        qy = robot_pose.rotation.y
        qz = robot_pose.rotation.z
        qw = robot_pose.rotation.w
        r = R.from_quat([qx, qy, qz, qw])
        robot_yaw = r.as_euler("zyx", degrees=False)[0]

        for idx, segment in enumerate(segments):
            intersection_points = self.get_intersection_points(
                segment, look_ahead_circle
            )
            valid_points = []
            segment_contains_valid_points = False
            for pt in intersection_points:
                if self.is_valid_point(
                    pt, segment, robot_position, robot_yaw
                ) and not dead_zone.contains(pt):
                    # if not dead_zone.contains(pt):
                    found_first_valid_point = True
                    segment_contains_valid_points = True
                    valid_points.append(pt)
            if segment_contains_valid_points:
                selected_segment = segment
                look_ahead_point = max(
                    valid_points, key=lambda pt: segment.line.project(pt)
                )
            if found_first_valid_point and max_search_idx == -1:
                max_search_idx = min(len(segments), idx + search_window_size)
            if found_first_valid_point and idx >= max_search_idx:
                break

        if look_ahead_point is None:
            # No valid points found within the look-ahead distance
            print(
                "No valid look-ahead points found within the initial look-ahead distance."
            )
            nearest_point, nearest_segment, nearest_segment_direction = (
                self.find_nearest_point_on_path(segments, robot_position)
            )
            if nearest_point is None:
                print("ERROR: Unable to find the nearest point on the path.")
                return None, None, None, None
            cte = self.calculate_cross_track_error(nearest_segment, robot_position)
            oe = self.calculate_look_ahead_based_orientation_error(
                (nearest_point.x, nearest_point.y),
                robot_position,
                robot_yaw,
                nearest_segment_direction,
            )
            return (
                (nearest_point.x, nearest_point.y),
                cte,
                oe,
                nearest_segment_direction,
            )

        # Calculate Cross-Track Error (cte)
        cte = self.calculate_cross_track_error(selected_segment.line, robot_position)

        # Calculate Orientation Error (oe)
        oe = self.calculate_look_ahead_based_orientation_error(
            (look_ahead_point.x, look_ahead_point.y), robot_position, robot_yaw, selected_segment.direction
        )
        # print(
        #         f"({xR:.3f}, {yR:.3f}, {robot_yaw:.2f})",
        #         look_ahead_point,
        #         selected_segment,
        #         f"{oe:.2f}",
        #     )

        return (look_ahead_point.x, look_ahead_point.y), cte, oe, selected_segment.direction

    def pub_twist(self, v: float, omega: float) -> None:
        """
        v: linear velocity
        omega: angular velocity
        """
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        self.pub_cmd_vel.publish(twist_msg)

    def control_loop(self):
        with self.path_msg_lock:
            # check path segment by path segment if the robot is closest to it
            # if the robot is closer to the next path segment, move on
            # If the robot is too far the track, reduce the error
            # If the robot is within acceptable limits, do path tracking mode
            if self.path is None:
                return
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    "world",  # Target frame
                    f"{self.robot_name}/base_link",  # Source frame
                    rclpy.time.Time(),
                    Duration(seconds=1.0),
                )
            except:
                return
            if transform is None or transform.transform is None:
                return
            robot_pose = transform.transform

            end_position = Point(self.path[-1].x, self.path[-1].y)
            robot_position = Point(robot_pose.translation.x, robot_pose.translation.y)
            if robot_position.distance(end_position) < self.look_ahead_distance * 0.1:
                omega = 0.0
                v = 0.0
                self.pub_twist(v, omega)
                return

            segments = self.create_segments(self.path)
            look_ahead_point, cte, oe, direction = (
                self.calculate_look_ahead_point_on_path(
                    segments, robot_pose, self.look_ahead_distance
                )
            )
            if (
                look_ahead_point is None
                or cte is None
                or oe is None
                or direction is None
            ):
                return

            if direction == Direction.FORWARD:
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

            # if (
            #     direction == Direction.REVERSE
            # ):  # TODO: confirm that omega must be reversed.
            #     omega = -omega

            if omega is None or v is None:
                return

            qx = robot_pose.rotation.x
            qy = robot_pose.rotation.y
            qz = robot_pose.rotation.z
            qw = robot_pose.rotation.w
            r = R.from_quat([qx, qy, qz, qw])
            yaw_r = r.as_euler("zyx", degrees=False)[0]
            print(
                f"({robot_pose.translation.x:.3f}, {robot_pose.translation.y:.3f}, {yaw_r:.2f})",
                look_ahead_point,
                direction,
                f"{cte:.2f}",
                f"{oe:.2f}",
                f"{v:.2f}",
                f"{omega:.2f}",
            )

            self.pub_twist(v, omega)
            #########
            # WRITE #
            #########
            # xR = robot_pose.translation.x
            # yR = robot_pose.translation.y
            # qx = robot_pose.rotation.x
            # qy = robot_pose.rotation.y
            # qz = robot_pose.rotation.z
            # qw = robot_pose.rotation.w
            # r = R.from_quat([qx, qy, qz, qw])
            # yaw_r = r.as_euler("zyx", degrees=False)[0]
            # if self.write:
            #     if current_path_seg is not None:
            #         xA = current_path_seg[0].pose.position.x
            #         yA = current_path_seg[0].pose.position.y
            #         xB = current_path_seg[1].pose.position.x
            #         yB = current_path_seg[1].pose.position.y
            #         # print(f"Robot at {xR, yR}, aiming for ")
            #         with open("src/data.txt", "a") as file:
            #             file.write(
            #                 f"{xR:.2f}, {yR:.2f}, {yaw_r:.2f}, {xA:.2f}, {yA:.2f}, {xB:.2f}, {yB:.2f}, {cte:.2f}, {oe:.2f}, {v:.2f}, {omega:.2f} \n"
            #             )


def main(args=None):
    rclpy.init(args=args)
    node = Tracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
