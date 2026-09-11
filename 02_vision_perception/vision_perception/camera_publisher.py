import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera')
            raise RuntimeError('Camera not available')

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.timer_callback
        )

        self.get_logger().info('Camera publisher started')

    def timer_callback(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning('Failed to read frame')
            return

        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        msg.header.stamp = self.get_clock().now().to_msg()

        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()