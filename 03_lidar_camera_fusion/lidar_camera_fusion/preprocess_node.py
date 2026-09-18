import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PreprocessNode(Node):

    def __init__(self):
        super().__init__('lidar_preprocess_node')

        self.sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            PointCloud2,
            '/lidar/points_filtered',
            10
        )

        self.get_logger().info('LiDAR preprocessing node started.')

    def callback(self, msg):

        points = point_cloud2.read_points_numpy(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        ).astype(np.float32, copy=False)

        if points.size == 0:
            return

        # ROI crop + ground removal
        mask = (
            (points[:, 0] > -15.0) &
            (points[:, 0] < 15.0) &
            (points[:, 1] > -15.0) &
            (points[:, 1] < 15.0) &
            (points[:, 2] > -0.45) &
            (points[:, 2] < 3.0)
        )

        filtered = points[mask]

        if filtered.shape[0] == 0:
            return

        # Voxel downsampling
        voxel_size = 0.05

        voxel_indices = np.floor(
            filtered / voxel_size
        ).astype(np.int32)

        _, unique_indices = np.unique(
            voxel_indices,
            axis=0,
            return_index=True
        )

        filtered = filtered[unique_indices]

        if filtered.shape[0] == 0:
            return

        output = point_cloud2.create_cloud_xyz32(
            msg.header,
            filtered.tolist()
        )

        self.pub.publish(output)

        self.get_logger().info(
            f'raw={points.shape[0]} filtered={filtered.shape[0]}',
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = PreprocessNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
