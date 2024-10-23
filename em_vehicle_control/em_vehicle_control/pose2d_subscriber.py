# pose2d_subscriber.py
import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Pose2D

class Pose2DSubscriber(Node):
    def __init__(self):
        super().__init__('pose2d_subscriber')
        self.subscription = self.create_subscription(
            Pose2D,
            'pose2d_topic',
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        direction = "FORWARD" if msg.direction_flag == Pose2D.FORWARD else "BACKWARD"
        self.get_logger().info(
            f'Received Pose2D: x={msg.x}, y={msg.y}, direction={direction}'
        )

def main(args=None):
    rclpy.init(args=args)
    subscriber = Pose2DSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
