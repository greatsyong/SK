import csv
import math
import os

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    return math.atan2(siny_cosp, cosy_cosp)


class EvaluationNode(Node):

    def __init__(self):
        super().__init__('evaluation_node')

        self.gt = None
        self.raw = None
        self.ekf = None

        self.start_time = self.get_clock().now()

        results_dir = os.path.expanduser(
            '~/robotics_ws/src/ekf_sensor_fusion/results'
        )

        os.makedirs(results_dir, exist_ok=True)

        self.csv_path = os.path.join(
            results_dir,
            'ekf_results.csv'
        )

        self.file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            'time',
            'gt_x',
            'gt_y',
            'gt_yaw',
            'raw_x',
            'raw_y',
            'raw_yaw',
            'ekf_x',
            'ekf_y',
            'ekf_yaw'
        ])

        self.create_subscription(
            Odometry,
            '/ground_truth',
            self.gt_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.raw_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.ekf_callback,
            10
        )

        self.timer = self.create_timer(
            0.1,
            self.record
        )

        self.get_logger().info(
            f'Recording results to {self.csv_path}'
        )

    def gt_callback(self, msg):
        self.gt = msg

    def raw_callback(self, msg):
        self.raw = msg

    def ekf_callback(self, msg):
        self.ekf = msg

    def record(self):

        if (
            self.gt is None
            or self.raw is None
            or self.ekf is None
        ):
            return

        now = self.get_clock().now()

        t = (
            now - self.start_time
        ).nanoseconds / 1e9

        gt_yaw = quaternion_to_yaw(
            self.gt.pose.pose.orientation
        )

        raw_yaw = quaternion_to_yaw(
            self.raw.pose.pose.orientation
        )

        ekf_yaw = quaternion_to_yaw(
            self.ekf.pose.pose.orientation
        )

        self.writer.writerow([
            t,

            self.gt.pose.pose.position.x,
            self.gt.pose.pose.position.y,
            gt_yaw,

            self.raw.pose.pose.position.x,
            self.raw.pose.pose.position.y,
            raw_yaw,

            self.ekf.pose.pose.position.x,
            self.ekf.pose.pose.position.y,
            ekf_yaw
        ])

        self.file.flush()

    def destroy_node(self):

        if not self.file.closed:
            self.file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = EvaluationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()