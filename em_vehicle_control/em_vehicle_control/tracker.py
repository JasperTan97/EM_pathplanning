from scipy.spatial.transform import Rotation as R
import networkx as nx
import threading
import numpy as np
from typing import Tuple, Union, List

from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadGraph
from em_vehicle_control.helper_classes.global_plannar import GlobalPlannar
from em_vehicle_control.helper_classes.vehicles import EdyMobile

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Transform, PoseStamped, Pose, Twist
from nav_msgs.msg import Path
import tf2_ros
from rclpy.duration import Duration


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
        self.orientation_error_bounds = 5 / 180 * np.pi  # radians
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
            Path, f"path", self.path_subscription, 1
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
        path_seg: Tuple[
            Union[Pose, PoseStamped, float], Union[Pose, PoseStamped, float]
        ],
        robot_pose: Transform,
    ) -> float:
        """
        path_seg: 2 points that form a line segment
        robot_pose: pose of robot from tf tree
        return: shortest distance between robot and line segment
        """
        if type(path_seg[0]) == PoseStamped:
            xA = path_seg[0].pose.position.x
            yA = path_seg[0].pose.position.y
        elif type(path_seg[0]) == Pose:
            xA = path_seg[0].position.x
            yA = path_seg[0].position.y
        else:
            xA = path_seg[0][0]
            yA = path_seg[0][1]
        if type(path_seg[1]) == PoseStamped:
            xB = path_seg[1].pose.position.x
            yB = path_seg[1].pose.position.y
        elif type(path_seg[1]) == Pose:
            xB = path_seg[1].position.x
            yB = path_seg[1].position.y
        else:
            xB = path_seg[1][0]
            yB = path_seg[1][1]
        xR = robot_pose.translation.x
        yR = robot_pose.translation.y
        AB = np.array([xB - xA, yB - yA])
        AR = np.array([xR - xA, yR - yA])
        return np.abs(np.cross(AB, AR)) / np.linalg.norm(AB)

    def calculate_segment_based_orientation_error(
        self,
        path_head: Union[Tuple[Pose, Pose], Tuple[PoseStamped, PoseStamped]],
        robot_pose: Transform,
    ) -> float:
        """
        path_head: 1st point of the path segment, should provide the orientation of the path segment
        robot_pose: pose of robot from tf tree
        return: angle between the line segment and robot_pose
        """
        if type(path_head) == PoseStamped:
            pt_A = path_head.pose
        else:
            pt_A = path_head
        qx = pt_A.orientation.x
        qy = pt_A.orientation.y
        qz = pt_A.orientation.z
        qw = pt_A.orientation.w
        r = R.from_quat([qx, qy, qz, qw])
        yaw_des = r.as_euler("zyx", degrees=False)[0]
        qx = robot_pose.rotation.x
        qy = robot_pose.rotation.y
        qz = robot_pose.rotation.z
        qw = robot_pose.rotation.w
        r = R.from_quat([qx, qy, qz, qw])
        yaw_r = r.as_euler("zyx", degrees=False)[0]
        return np.mod(yaw_des - yaw_r + np.pi, 2 * np.pi) - np.pi

    def calculate_look_ahead_based_orientation_error(
        self,
        look_ahead_point: Tuple[float, float],
        robot_pose: Transform,
    ) -> float:
        """
        look_ahead_point: point the robot should move towards
        robot_pose: pose of robot from tf tree
        return: angle between the robot and the look ahead point
        """
        xP, yP = look_ahead_point
        xR = robot_pose.translation.x
        yR = robot_pose.translation.y
        qx = robot_pose.rotation.x
        qy = robot_pose.rotation.y
        qz = robot_pose.rotation.z
        qw = robot_pose.rotation.w
        r = R.from_quat([qx, qy, qz, qw])
        yaw_r = r.as_euler("zyx", degrees=False)[0]
        theta_lookahead = np.arctan2(yP - yR, xP - xR)
        orientation_error = theta_lookahead - yaw_r
        return (orientation_error + np.pi) % (2 * np.pi) - np.pi

    def calculate_closest_point(
        self,
        path_seg: Tuple[Union[Pose, PoseStamped], Union[Pose, PoseStamped]],
        robot_pose: Transform,
    ) -> Tuple[float, float]:
        """
        path_seg: 2 points that form a line segment
        robot_pose: pose of robot from tf tree
        return: point giving the shortest distance between robot and line segment
        """
        if type(path_seg[0]) == PoseStamped:
            pt_A = path_seg[0].pose
            pt_B = path_seg[1].pose
        else:
            pt_A = path_seg[0]
            pt_B = path_seg[1]
        xA = pt_A.position.x
        yA = pt_A.position.y
        xB = pt_B.position.x
        yB = pt_B.position.y
        xR = robot_pose.translation.x
        yR = robot_pose.translation.y
        AB = np.array([xB - xA, yB - yA])
        AR = np.array([xR - xA, yR - yA])
        t = np.dot(AB, AR) / np.dot(AB, AB)
        t = max(0, min(1, t))
        return (xA + t * AB[0], yA + t * AB[1])

    def calculate_look_ahead_point_on_segment(
        self,
        path_seg: Tuple[Union[Pose, PoseStamped], Union[Pose, PoseStamped]],
        robot_pose: Transform,
        look_ahead_distance: float,
    ) -> Tuple[float, float]:
        """
        path_seg: 2 points that form a line segment
        robot_pose: pose of robot from tf tree
        look_ahead_distance: circle centered on the robot
        return: the point on the path segment intersecting the circle formed by the look ahead distance
        """
        if type(path_seg[0]) == PoseStamped:
            pt_A = path_seg[0].pose
            pt_B = path_seg[1].pose
        else:
            pt_A = path_seg[0]
            pt_B = path_seg[1]
        xA = pt_A.position.x
        yA = pt_A.position.y
        xB = pt_B.position.x
        yB = pt_B.position.y
        xR = robot_pose.translation.x
        yR = robot_pose.translation.y
        AB = np.array([xB - xA, yB - yA])
        P_closest = self.calculate_closest_point([pt_A, pt_B], robot_pose)
        dist_AB = np.linalg.norm([xB - xA, yB - yA], 2)
        dist_AP_closest = np.linalg.norm([P_closest[0] - xA, P_closest[1] - yA], 2)
        dist_RP_closest = np.linalg.norm([P_closest[0] - xR, P_closest[1] - yR], 2)
        dist_look_forward_sq = look_ahead_distance**2 - dist_RP_closest**2
        if dist_look_forward_sq < 0:
            return None
        dist_look_forward = np.sqrt(dist_look_forward_sq)
        if dist_AP_closest + dist_look_forward > dist_AB:
            return None
        else:
            return tuple(
                (dist_AP_closest + dist_look_forward) / dist_AB * AB
                + np.array([xA, yA])
            )

    def calculate_look_ahead_point_on_path(
        self,
        path: List[Tuple[PoseStamped, PoseStamped]],
        robot_pose: Transform,
        look_ahead_distance: float,
    ) -> Tuple[Tuple[float, float], float, float]:
        """
        path_seg: 2 points that form a line segment
        robot_pose: pose of robot from tf tree
        look_ahead_distance: circle centered on the robot
        return: the point on the path segment intersecting the circle formed by the look ahead distance,
                cross track error and orientation error
        """

        # 3 cases: 2 path segments, 1 path segment and 1 path point
        if len(path) == 1:
            # one point remaining, go to the end
            curr_cte = 0.0
            final_node = self.path[0]
            dx = final_node.pose.position.x - robot_pose.translation.x
            dy = final_node.pose.position.y - robot_pose.translation.y
            yaw_des = np.arctan2(dy, dx)
            qx = robot_pose.rotation.x
            qy = robot_pose.rotation.y
            qz = robot_pose.rotation.z
            qw = robot_pose.rotation.w
            r = R.from_quat([qx, qy, qz, qw])
            yaw_r = r.as_euler("zyx", degrees=False)[0]
            curr_oe = np.mod(yaw_des - yaw_r + np.pi, 2 * np.pi) - np.pi
            return None, curr_cte, curr_oe

        curr_seg, next_seg = None, None
        for i in range(len(path) - 1):
            curr_seg = path[i : i + 2]
            curr_cte = self.calculate_cross_track_error(curr_seg, robot_pose)
            next_seg = None
            if len(path) >= i + 3:
                # check if there is a next segment
                next_seg = path[i + 1 : i + 3]
                next_cte = self.calculate_cross_track_error(next_seg, robot_pose)
                if next_cte > curr_cte:
                    # if next segment is closer, then we repeat and check the following segment
                    # otherwise, we take it
                    break
        
        print(f"Taking segment {[(g.pose.position.x, g.pose.position.y) for g in curr_seg]}")
        look_ahead_pt, cte, oe = None, None, None
        for _ in range(5):
            if next_seg is not None:
                next_look_ahead_pt = self.calculate_look_ahead_point_on_segment(
                    next_seg, robot_pose, look_ahead_distance
                )
                if next_look_ahead_pt is not None:
                    look_ahead_pt = next_look_ahead_pt
                    cte = self.calculate_cross_track_error(
                        (next_seg[0], look_ahead_pt), robot_pose
                    )
                    oe = self.calculate_look_ahead_based_orientation_error(
                        look_ahead_pt, robot_pose
                    )
                    return look_ahead_pt, cte, oe
            curr_look_ahead_pt = self.calculate_look_ahead_point_on_segment(
                curr_seg, robot_pose, look_ahead_distance
            )
            if curr_look_ahead_pt is not None:
                look_ahead_pt = curr_look_ahead_pt
                cte = self.calculate_cross_track_error(
                    (curr_seg[0], look_ahead_pt), robot_pose
                )
                oe = self.calculate_look_ahead_based_orientation_error(
                    look_ahead_pt, robot_pose
                )
                return look_ahead_pt, cte, oe
            if look_ahead_pt is not None:
                break
            look_ahead_distance *= 1.3
        else:
            # if cannot be found, pick the last point
            look_ahead_pt = (curr_seg[1].pose.position.x, curr_seg[1].pose.position.y)
            cte = self.calculate_cross_track_error(
                (curr_seg[0], look_ahead_pt), robot_pose
            )
            oe = self.calculate_look_ahead_based_orientation_error(
                look_ahead_pt, robot_pose
            )
            return look_ahead_pt, cte, oe
        
    def pub_twist(self, v: float, omega:float) -> None:
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

            look_ahead_point, cte, oe = self.calculate_look_ahead_point_on_path(
                self.path, robot_pose, self.look_ahead_distance
            )
            # print(f"At: {robot_pose.translation.x:.2f}, {robot_pose.translation.y:.2f}, Moving towards: {look_ahead_point[0]:.2f},{look_ahead_point[1]:.2f}")

            # w = 1 / (1 / np.exp(-self.k_w * curr_cte - self.d_blend))
            # v_move = self.v_max * np.exp(-self.k_v * curr_cte)
            # omega = self.K_omega * curr_oe
            # omega = np.clip(omega, -self.omega_max, self.omega_max)
            # v = w * v_move + (1 - w) * self.v_max

            # To control movement, first we move to the right orientation, then push forward
            if not -self.orientation_error_bounds < oe < self.orientation_error_bounds:
                omega = self.K_omega * oe
                omega = np.clip(omega, -self.omega_max, self.omega_max)
                v = 0.0
            else:
                w = 1 / (1 / np.exp(-self.k_w * cte - self.d_blend))
                v_move = self.v_max * np.exp(-self.k_v * cte)
                omega = self.K_omega * oe
                omega = np.clip(omega, -self.omega_max, self.omega_max)
                v = w * v_move + (1 - w) * self.v_max

            if omega is None or v is None:
                return
            
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
