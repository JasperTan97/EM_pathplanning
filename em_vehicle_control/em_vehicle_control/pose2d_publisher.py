# pose2d_publisher.py
import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Pose2D
from std_msgs.msg import Header

class Pose2DPublisher(Node):
    def __init__(self):
        super().__init__('pose2d_publisher')
        self.publisher_ = self.create_publisher(Pose2D, 'pose2d_topic', 10)
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_pose)

    def publish_pose(self):
        # Pose 1: FORWARD
        msg_forward = Pose2D()
        msg_forward.x = 1.0
        msg_forward.y = 2.0
        msg_forward.direction_flag = Pose2D.FORWARD
        self.publisher_.publish(msg_forward)
        self.get_logger().info(f'Published Pose2D: x={msg_forward.x}, y={msg_forward.y}, direction=FORWARD')

        # Pose 2: BACKWARD
        msg_backward = Pose2D()
        msg_backward.x = 3.0
        msg_backward.y = 4.0
        msg_backward.direction_flag = Pose2D.BACKWARD
        self.publisher_.publish(msg_backward)
        self.get_logger().info(f'Published Pose2D: x={msg_backward.x}, y={msg_backward.y}, direction=BACKWARD')

def main(args=None):
    rclpy.init(args=args)
    publisher = Pose2DPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()