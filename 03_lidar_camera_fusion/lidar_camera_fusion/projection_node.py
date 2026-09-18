import rclpy
from rclpy.node import Node
from rclpy.time import Time

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener, TransformException


class LidarCameraProjectionNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        self.bridge = CvBridge()

        self.camera_frame = 'front_stereo_camera:left_rgb'
        self.lidar_frame = 'front_3d_lidar'

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_image = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo,
            '/front_camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.lidar_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/fusion/lidar_projection',
            10
        )

        self.get_logger().info(
            'Optimized LiDAR-Camera Projection Node started.'
        )

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])

            self.get_logger().info(
                f'Camera intrinsics loaded: '
                f'fx={self.fx:.3f}, fy={self.fy:.3f}, '
                f'cx={self.cx:.3f}, cy={self.cy:.3f}'
            )

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )

    @staticmethod
    def quaternion_to_rotation_matrix(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array([
            [
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy)
            ],
            [
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx)
            ],
            [
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy)
            ]
        ], dtype=np.float32)

    def lidar_callback(self, msg):

        if self.latest_image is None:
            return

        if self.fx is None:
            return

        # Read PointCloud2 directly into NumPy
        try:
            points = point_cloud2.read_points_numpy(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )
        except Exception as e:
            self.get_logger().error(
                f'PointCloud conversion failed: {e}'
            )
            return

        if points.size == 0:
            return

        points = np.asarray(points, dtype=np.float32)

        # Optional downsampling
        # Keep every second point.
        points = points[::2]

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.lidar_frame,
                Time()
            )

        except TransformException as ex:
            self.get_logger().warning(
                f'Could not get transform '
                f'{self.lidar_frame} -> {self.camera_frame}: {ex}'
            )
            return

        t = transform.transform.translation
        q = transform.transform.rotation

        translation = np.array(
            [t.x, t.y, t.z],
            dtype=np.float32
        )

        rotation = self.quaternion_to_rotation_matrix(
            q.x,
            q.y,
            q.z,
            q.w
        )

        # LiDAR -> camera frame
        points_camera = points @ rotation.T
        points_camera += translation

        # Keep points in front of camera
        front_mask = points_camera[:, 2] > 0.1
        points_camera = points_camera[front_mask]

        if points_camera.shape[0] == 0:
            return

        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]

        # Vectorized pinhole projection
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        image = self.latest_image.copy()
        height, width = image.shape[:2]

        valid = (
            (u >= 0.0) &
            (u < width) &
            (v >= 0.0) &
            (v < height)
        )

        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
        z = z[valid]

        if z.size == 0:
            return

        # Depth range
        min_depth = 0.5
        max_depth = 30.0

        depth_norm = np.clip(
            (z - min_depth) / (max_depth - min_depth),
            0.0,
            1.0
        )

        depth_values = (
            (1.0 - depth_norm) * 255.0
        ).astype(np.uint8)

        # Apply colormap to all points at once
        colors = cv2.applyColorMap(
            depth_values.reshape(-1, 1),
            cv2.COLORMAP_JET
        ).reshape(-1, 3)

        # Fast direct-pixel overlay
        image[v, u] = colors

        output_msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        output_msg.header = msg.header
        output_msg.header.frame_id = self.camera_frame

        self.image_pub.publish(output_msg)


def main(args=None):

    rclpy.init(args=args)

    node = LidarCameraProjectionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
