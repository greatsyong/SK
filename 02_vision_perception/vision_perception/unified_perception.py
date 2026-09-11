import csv
import os
import time

import cv2
import rclpy
import torch

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class UnifiedPerception(Node):

    def __init__(self):
        super().__init__('unified_perception')

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
            '/vision/unified_image',
            10
        )

        self.prev_time = time.perf_counter()

        # ---------------------------------
        # CSV logging
        # ---------------------------------
        self.csv_path = os.path.expanduser(
            '~/robotics_ws/src/vision_perception/results/vision_results.csv'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'timestamp',
            'fps',
            'processing_latency_ms',
            'inference_latency_ms',
            'object_count'
        ])

        self.csv_file.flush()

        self.get_logger().info(
            f'Unified perception started | device: {self.device}'
        )

        self.get_logger().info(
            f'Logging results to: {self.csv_path}'
        )

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # ---------------------------------
        # Processing latency start
        # ---------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        processing_start = time.perf_counter()

        # Detection + Segmentation + Tracking
        results = self.model.track(
            frame,
            device=self.device,
            persist=True,
            verbose=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        processing_end = time.perf_counter()

        processing_latency_ms = (
            processing_end - processing_start
        ) * 1000.0

        result = results[0]

        # Ultralytics internal inference timing
        inference_latency_ms = result.speed.get(
            'inference',
            0.0
        )

        # Visualization
        annotated_frame = result.plot()

        # ---------------------------------
        # FPS
        # ---------------------------------
        current_time = time.perf_counter()

        dt = current_time - self.prev_time

        if dt > 0:
            fps = 1.0 / dt
        else:
            fps = 0.0

        self.prev_time = current_time

        # ---------------------------------
        # Object count
        # ---------------------------------
        if result.boxes is not None:
            object_count = len(result.boxes)
        else:
            object_count = 0

        # ---------------------------------
        # Overlay
        # ---------------------------------
        cv2.putText(
            annotated_frame,
            f'FPS: {fps:.1f}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f'Objects: {object_count}',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f'Latency: {processing_latency_ms:.1f} ms',
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f'Device: {"GPU" if torch.cuda.is_available() else "CPU"}',
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ---------------------------------
        # CSV
        # ---------------------------------
        timestamp = self.get_clock().now().nanoseconds / 1e9

        self.csv_writer.writerow([
            f'{timestamp:.6f}',
            f'{fps:.3f}',
            f'{processing_latency_ms:.3f}',
            f'{inference_latency_ms:.3f}',
            object_count
        ])

        self.csv_file.flush()

        # ---------------------------------
        # Publish
        # ---------------------------------
        output_msg = self.bridge.cv2_to_imgmsg(
            annotated_frame,
            encoding='bgr8'
        )

        output_msg.header = msg.header

        self.publisher.publish(output_msg)

    def destroy_node(self):

        if hasattr(self, 'csv_file'):
            self.csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = UnifiedPerception()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()