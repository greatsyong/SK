import math
import random

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def yaw_to_quaternion(yaw):
    return (
        0.0,
        0.0,
        math.sin(yaw / 2.0),
        math.cos(yaw / 2.0)
    )


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SyntheticSensorNode(Node):

    def __init__(self):
        super().__init__('synthetic_sensor_node')

        random.seed(42)

        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        self.imu_pub = self.create_publisher(
            Imu,
            '/imu',
            10
        )

        self.gt_pub = self.create_publisher(
            Odometry,
            '/ground_truth',
            10
        )

        self.dt = 0.02

        self.timer = self.create_timer(
            self.dt,
            self.timer_callback
        )

        self.start_time = self.get_clock().now()

        # Ground truth state
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_yaw = 0.0

        # Raw wheel odometry state
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0

        self.get_logger().info(
            'Synthetic sensors + ground truth started'
        )

    def timer_callback(self):

        now = self.get_clock().now()

        t = (
            now - self.start_time
        ).nanoseconds / 1e9

        # Ground-truth motion
        true_v = 1.0

        true_w = (
            0.25 * math.sin(0.3 * t)
            + 0.05
        )

        # Integrate ground truth
        self.true_x += (
            true_v
            * math.cos(self.true_yaw)
            * self.dt
        )

        self.true_y += (
            true_v
            * math.sin(self.true_yaw)
            * self.dt
        )

        self.true_yaw = normalize_angle(
            self.true_yaw
            + true_w * self.dt
        )

        # Simulated wheel odometry
        wheel_v = (
            true_v
            + random.gauss(0.0, 0.15)
        )

        wheel_w = (
            true_w
            + 0.03
            + random.gauss(0.0, 0.15)
        )

        # Raw wheel dead-reckoning
        self.raw_x += (
            wheel_v
            * math.cos(self.raw_yaw)
            * self.dt
        )

        self.raw_y += (
            wheel_v
            * math.sin(self.raw_yaw)
            * self.dt
        )

        self.raw_yaw = normalize_angle(
            self.raw_yaw
            + wheel_w * self.dt
        )

        # Simulated IMU
        imu_w = (
            true_w
            + random.gauss(0.0, 0.04)
        )

        # Publish ground truth
        gt = Odometry()

        gt.header.stamp = now.to_msg()
        gt.header.frame_id = 'odom'
        gt.child_frame_id = 'base_link_gt'

        gt.pose.pose.position.x = self.true_x
        gt.pose.pose.position.y = self.true_y

        qx, qy, qz, qw = yaw_to_quaternion(
            self.true_yaw
        )

        gt.pose.pose.orientation.x = qx
        gt.pose.pose.orientation.y = qy
        gt.pose.pose.orientation.z = qz
        gt.pose.pose.orientation.w = qw

        gt.twist.twist.linear.x = true_v
        gt.twist.twist.angular.z = true_w

        self.gt_pub.publish(gt)

        # Publish raw wheel odometry
        odom = Odometry()

        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link_raw'

        odom.pose.pose.position.x = self.raw_x
        odom.pose.pose.position.y = self.raw_y

        qx, qy, qz, qw = yaw_to_quaternion(
            self.raw_yaw
        )

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = wheel_v
        odom.twist.twist.angular.z = wheel_w

        self.odom_pub.publish(odom)

        # Publish IMU
        imu = Imu()

        imu.header.stamp = now.to_msg()
        imu.header.frame_id = 'imu_link'

        imu.angular_velocity.z = imu_w

        self.imu_pub.publish(imu)


def main(args=None):

    rclpy.init(args=args)

    node = SyntheticSensorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()