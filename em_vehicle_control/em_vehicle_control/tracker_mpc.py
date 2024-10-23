from scipy.spatial.transform import Rotation as R
import networkx as nx
from shapely import LineString, Point, MultiLineString, MultiPoint
import threading
import numpy as np
from typing import Tuple, Union, List
from enum import Enum
from time import time

from em_vehicle_control.helper_classes.mpc_tracker_theta import MPCTracker
from em_vehicle_control.helper_classes.segment import *

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Transform, Twist
from em_vehicle_control_msgs.msg import Pose2D, Path2D
import tf2_ros
from rclpy.duration import Duration

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values

class Direction(Enum):
    FORWARD = 1
    BACKWARD = -1

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
        

        self.path = None
        self.path_msg_lock = threading.Lock()

        self.path_subscription = self.create_subscription(
            Path2D, f"path", self.path_subscription, 1
        )
        self.path_subscription
        self.timer = self.create_timer(0.125, self.timer_callback)  # 8Hz

        self.tracker = MPCTracker()

        #########
        # WRITE #
        #########
        self.write = False
        if self.write:
            with open("src/data.txt", "w") as file:
                file.write(
                    f"xR, yR, yaw_R, xA, yA, xB, yB, curr_cte, curr_oe, v, omega \n"
                )

    def path_subscription(self, msg):
        with self.path_msg_lock:
            self.path = msg.poses
            self.get_logger().info("New path")
            self.tracker.initialise_new_path()

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

    def get_robot_pose(self) -> PosePt2D:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "world",  # Target frame
                f"{self.robot_name}/base_link",  # Source frame
                rclpy.time.Time(),
                Duration(seconds=1.0),
            )
        except:
            return None
        if transform is None or transform.transform is None:
            return None
        robot_pose = transform.transform
        qx = robot_pose.rotation.x
        qy = robot_pose.rotation.y
        qz = robot_pose.rotation.z
        qw = robot_pose.rotation.w
        r = R.from_quat([qx, qy, qz, qw])
        robot_yaw = r.as_euler("zyx", degrees=False)[0]
        robot_pose = (
            robot_pose.translation.x,
            robot_pose.translation.y,
            robot_yaw
        )
        return robot_pose

    def control_loop(self):
        """
        Control loop that manages and runs the MPC tracker,
        and publishes velocity commands
        """
        if self.path is None:
            return
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        
        segments = create_segments(self.path)
        tic = time()
        v, omega = self.tracker.track(robot_pose, segments)
        toc = time()
        # self.get_logger().info(f"{toc-tic}")

        self.pub_twist(v, omega)

    def timer_callback(self) -> None:
        """
        Callback that runs every tick of the ros timer. 
        Locks path object for thread safety.
        """
        with self.path_msg_lock:
            self.control_loop()


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
