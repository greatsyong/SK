import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import cv2
import torch


class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.bridge = CvBridge()

        self.model = YOLO('yolo11n.pt')

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/vision/detection_image',
            10
        )

        self.get_logger().info(
            f'YOLO detector started | device: {self.device}'
        )

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        results = self.model.track(
            frame,
            device=self.device,
            persist=True,
            verbose=False
        )

        annotated_frame = results[0].plot()

        output_msg = self.bridge.cv2_to_imgmsg(
            annotated_frame,
            encoding='bgr8'
        )

        output_msg.header = msg.header

        self.publisher.publish(output_msg)


def main(args=None):

    rclpy.init(args=args)

    node = YoloDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()