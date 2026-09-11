import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from ekf_sensor_fusion.ekf_core import EKF


def yaw_to_quaternion(yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    return 0.0, 0.0, qz, qw


class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_sensor_fusion')

        self.ekf = EKF()
        self.last_time = self.get_clock().now()

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )

        self.publisher = self.create_publisher(
            Odometry,
            '/odometry/filtered',
            10
        )

        self.timer = self.create_timer(
            0.02,
            self.timer_callback
        )

        self.get_logger().info(
            'EKF Sensor Fusion node started'
        )

    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x

        self.ekf.update_velocity(
            v_measured=v,
            variance=0.05
        )

    def imu_callback(self, msg):
        w = msg.angular_velocity.z

        self.ekf.update_yaw_rate(
            w_measured=w,
            variance=0.02
        )

    def timer_callback(self):
        now = self.get_clock().now()

        dt = (
            now - self.last_time
        ).nanoseconds / 1e9

        self.last_time = now

        self.ekf.predict(dt)

        self.publish_odometry(now)

    def publish_odometry(self, timestamp):
        px, py, yaw, v, w = self.ekf.x.flatten()

        msg = Odometry()

        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = float(px)
        msg.pose.pose.position.y = float(py)
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(w)

        msg.pose.covariance[0] = float(self.ekf.P[0, 0])
        msg.pose.covariance[7] = float(self.ekf.P[1, 1])
        msg.pose.covariance[35] = float(self.ekf.P[2, 2])

        msg.twist.covariance[0] = float(self.ekf.P[3, 3])
        msg.twist.covariance[35] = float(self.ekf.P[4, 4])

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = EKFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()