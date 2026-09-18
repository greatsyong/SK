import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class CloudStats(Node):

    def __init__(self):
        super().__init__('cloud_stats')

        self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.callback,
            10
        )

        self.done = False

    def callback(self, msg):

        if self.done:
            return

        points = point_cloud2.read_points_numpy(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        ).astype(np.float32, copy=False)

        if points.size == 0:
            return

        print()
        print(f'Number of points: {len(points)}')

        for i, axis in enumerate(['X', 'Y', 'Z']):
            values = points[:, i]

            print(
                f'{axis}: '
                f'min={np.min(values):.3f}, '
                f'p1={np.percentile(values, 1):.3f}, '
                f'p10={np.percentile(values, 10):.3f}, '
                f'p50={np.percentile(values, 50):.3f}, '
                f'p90={np.percentile(values, 90):.3f}, '
                f'p99={np.percentile(values, 99):.3f}, '
                f'max={np.max(values):.3f}'
            )

        self.done = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CloudStats()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
