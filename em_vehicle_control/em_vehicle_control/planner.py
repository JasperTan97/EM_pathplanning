from scipy.spatial.transform import Rotation as R
import networkx as nx
import numpy as np

from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadGraph
from em_vehicle_control.helper_classes.global_plannar import GlobalPlannar
from em_vehicle_control.helper_classes.vehicles import EdyMobile

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
import tf2_ros
from rclpy.duration import Duration


class Planner(Node):
    """
    Node that runs periodically
    Takes current poses, passes them through the planner,
    outputs path message with poses and the timestamp when the vehicle is allowed to pass the node
    """

    def __init__(self):
        super().__init__("planner")
        timer_period = 2  # seconds

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        test_roads = [
            RoadSegment((2.87, 1.67), (3.52, 13.67)),
            RoadSegment((6.9, 0.3), (7.55, 15.67)),
            RoadSegment((4.72, 1.67), (5.7, 8.64)),
            RoadSegment((2.87, 8), (7.55, 8.64)),
            RoadSegment((2.87, 1.67), (7.55, 2.32)),
            # RoadSegment((0,0), (10,10))
        ]
        test_map = RoadMap(test_roads)
        test_graph = RoadGraph(test_map)
        test_graph.make_vertices(0.4, 0.3, True)
        self.test_global_plannar = GlobalPlannar(test_graph)
        # test_global_plannar.CAstar([(3.1, 1.85)], [(3.3, 3.06)], [EdyMobile()], None)
        self.robot_names = ["robot_0"]
        self.goals = [(7.23, 14.9)]
        self.timer = self.create_timer(timer_period, self.plan_callback)
        self.pubbers = []
        for robot_name in self.robot_names:
            pub = self.create_publisher(Path, f"{robot_name}/path", 1)
            self.pubbers.append(pub)

    def plan_callback(self):
        current_poses = []
        for robot_name in self.robot_names:
            try:
                # Lookup transform from source_frame to target_frame
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    "world",  # Target frame
                    f"{robot_name}/base_link",  # Source frame
                    rclpy.time.Time(),
                    Duration(seconds=1.0),
                )
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                r = R.from_quat([qx, qy, qz, qw])
                yaw = r.as_euler("zyx", degrees=False)[0]
                current_poses.append((x, y, yaw))
            except tf2_ros.LookupException as e:
                self.get_logger().error(f"Transform lookup failed: {e}")

            except tf2_ros.ExtrapolationException as e:
                self.get_logger().error(f"Extrapolation exception: {e}")

            except tf2_ros.TransformException as e:
                self.get_logger().error(f"Transform exception: {e}")
        if current_poses == []:
            return
        current_pos = [(x[0], x[1]) for x in current_poses]
        self.test_global_plannar.CAstar(current_pos, self.goals, [EdyMobile()])
        current_time = rclpy.time.Time()
        paths = self.test_global_plannar.generate_paths()
        # paths = [[
        #     ((3.1,2.0), 3*np.pi/4, 0.0),
        #     ((2.1,3.0), np.pi/4, 0.0),
        #     ((3.1,4.0), np.pi/2, 0.0),
        #     ((3.1,5.0), 0.0, 0.0),
        #     ((4.0,5.0), 0.0, 0.0)
        #     ]]
        for path, pub in zip(paths, self.pubbers):
            path_msg = Path()
            path_msg.header.frame_id = "world"
            for node in path:
                posestamped_msg = PoseStamped()
                posestamped_msg.pose.position.x = node[0][0]
                posestamped_msg.pose.position.y = node[0][1]
                posestamped_msg.pose.position.z = 0.0
                r = R.from_euler("z", node[1], degrees=False)
                (
                    posestamped_msg.pose.orientation.x,
                    posestamped_msg.pose.orientation.y,
                    posestamped_msg.pose.orientation.z,
                    posestamped_msg.pose.orientation.w,
                ) = r.as_quat()
                time_in_nanoseconds = rclpy.time.Duration(nanoseconds=int(node[2] * 1e9))
                posestamped_msg.header.stamp = (time_in_nanoseconds + current_time).to_msg()
                posestamped_msg.header.frame_id = "world"
                path_msg.poses.append(posestamped_msg)
            pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Planner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
