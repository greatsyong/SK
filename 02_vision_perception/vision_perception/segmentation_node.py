import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import torch


class SegmentationNode(Node):

    def __init__(self):
        super().__init__('segmentation_node')

        self.bridge = CvBridge()

        self.model = YOLO('yolo11n-seg.pt')

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/vision/segmentation_image',
            10
        )

        self.get_logger().info(
            f'Segmentation node started | device: {self.device}'
        )

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        results = self.model(
            frame,
            device=self.device,
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

    node = SegmentationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()