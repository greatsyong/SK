import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from visualization_msgs.msg import Marker, MarkerArray

from scipy.spatial import cKDTree


class ClusteringNode(Node):

    def __init__(self):
        super().__init__('lidar_clustering_node')

        self.sub = self.create_subscription(
            PointCloud2,
            '/lidar/points_filtered',
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/lidar/clusters',
            10
        )

        # Initial clustering parameters
        self.cluster_tolerance = 0.30
        self.min_cluster_size = 20
        self.max_cluster_size = 4000

        self.get_logger().info(
            'LiDAR clustering node started.'
        )

    def euclidean_clustering(self, points):

        tree = cKDTree(points)

        visited = np.zeros(
            len(points),
            dtype=bool
        )

        clusters = []

        for i in range(len(points)):

            if visited[i]:
                continue

            queue = [i]
            visited[i] = True

            cluster = []

            while queue:

                idx = queue.pop()
                cluster.append(idx)

                neighbors = tree.query_ball_point(
                    points[idx],
                    self.cluster_tolerance
                )

                for neighbor in neighbors:

                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            size = len(cluster)

            if (
                self.min_cluster_size <= size
                <= self.max_cluster_size
            ):
                clusters.append(cluster)

        return clusters

    def callback(self, msg):

        points = point_cloud2.read_points_numpy(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        ).astype(np.float32, copy=False)

        if points.size == 0:
            return

        clusters = self.euclidean_clustering(points)

        marker_array = MarkerArray()

        # Clear previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        valid_count = 0

        for cluster_indices in clusters:

            cluster_points = points[cluster_indices]

            p_min = np.min(
                cluster_points,
                axis=0
            )

            p_max = np.max(
                cluster_points,
                axis=0
            )

            size = p_max - p_min
            center = (p_min + p_max) / 2.0

            # Reject large structures such as walls
            if (
                size[0] > 4.0 or
                size[1] > 4.0 or
                size[2] > 3.0
            ):
                continue

            # Reject extremely flat/noisy structures
            if size[2] < 0.08:
                continue

            marker = Marker()

            marker.header = msg.header
            marker.ns = 'clusters'
            marker.id = valid_count

            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            marker.pose.orientation.w = 1.0

            marker.scale.x = max(
                float(size[0]),
                0.05
            )

            marker.scale.y = max(
                float(size[1]),
                0.05
            )

            marker.scale.z = max(
                float(size[2]),
                0.05
            )

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.35

            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

            valid_count += 1

        self.marker_pub.publish(marker_array)

        self.get_logger().info(
            f'points={len(points)} '
            f'raw_clusters={len(clusters)} '
            f'object_clusters={valid_count}',
            throttle_duration_sec=2.0
        )


def main(args=None):

    rclpy.init(args=args)

    node = ClusteringNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
