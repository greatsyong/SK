import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import GetEntityState
from nav_msgs.msg import Odometry


class GazeboGroundTruthNode(Node):

    def __init__(self):
        super().__init__('gazebo_ground_truth_node')

        self.publisher = self.create_publisher(
            Odometry,
            '/ground_truth',
            10
        )

        self.client = self.create_client(
            GetEntityState,
            '/gazebo/get_entity_state'
        )

        self.robot_name = 'ekf_robot'
        self.future = None

        self.timer = self.create_timer(
            0.02,
            self.timer_callback
        )

        self.get_logger().info(
            'Gazebo ground truth node started'
        )

    def timer_callback(self):

        if not self.client.service_is_ready():
            return

        if self.future is not None and not self.future.done():
            return

        request = GetEntityState.Request()
        request.name = self.robot_name
        request.reference_frame = 'world'

        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(
                f'GetEntityState service call failed: {e}'
            )
            return

        if not response.success:
            self.get_logger().warn(
                f'Could not get state of {self.robot_name}'
            )
            return

        odom = Odometry()

        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'base_link_gt'

        odom.pose.pose = response.state.pose
        odom.twist.twist = response.state.twist

        self.publisher.publish(odom)


def main(args=None):

    rclpy.init(args=args)

    node = GazeboGroundTruthNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()