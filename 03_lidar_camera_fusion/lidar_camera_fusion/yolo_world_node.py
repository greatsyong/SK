import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLOWorld

import torch
import cv2


class YoloWorldNode(Node):

    def __init__(self):
        super().__init__('yolo_world_node')

        self.bridge = CvBridge()

        self.model = YOLOWorld(
            'yolov8s-worldv2.pt'
        )

        self.model.set_classes([
            'cardboard box',
            'pallet',
            'warehouse rack',
            'traffic cone',
            'warning sign',
            'forklift',
            'warehouse cart',
            'truck',
            'barrel',
            'person',
        ])

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        self.subscription = self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/vision/world_image',
            10
        )

        self.get_logger().info(
            f'YOLO-World started | device={self.device}'
        )

        self.get_logger().info(
            f'Classes: {self.model.names}'
        )

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        results = self.model.predict(
            frame,
            device=self.device,
            conf=0.15,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        object_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        cv2.putText(
            annotated,
            f'Objects: {object_count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        output_msg = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding='bgr8'
        )

        output_msg.header = msg.header

        self.publisher.publish(
            output_msg
        )


def main(args=None):

    rclpy.init(args=args)

    node = YoloWorldNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
