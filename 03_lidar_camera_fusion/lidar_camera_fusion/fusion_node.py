import csv
import os

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import FastSAM, YOLOWorld


class FusionNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_fusion_node')

        self.bridge = CvBridge()

        self.camera_frame = 'front_stereo_camera:left_rgb'
        self.lidar_frame = 'front_3d_lidar'

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_image = None
        self.latest_header = None

        # ---------------------------------
        # Fusion parameters
        # ---------------------------------
        self.depth_gap_threshold = 0.40
        self.min_lidar_points = 5

        # ---------------------------------
        # YOLO-World
        # ---------------------------------
        self.detector = YOLOWorld(
            'yolov8s-worldv2.pt'
        )

        self.detector.set_classes([
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

        # ---------------------------------
        # FastSAM
        # ---------------------------------
        self.sam = FastSAM(
            'FastSAM-s.pt'
        )

        self.device = (
            0 if torch.cuda.is_available()
            else 'cpu'
        )

        # ---------------------------------
        # TF
        # ---------------------------------
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ---------------------------------
        # ROS subscribers
        # ---------------------------------
        self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/front_camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.lidar_callback,
            10
        )

        # ---------------------------------
        # Publisher
        # ---------------------------------
        self.publisher = self.create_publisher(
            Image,
            '/fusion/fused_image',
            10
        )

        # ---------------------------------
        # CSV logging
        # ---------------------------------
        results_dir = os.path.expanduser(
            '~/robotics_ws/src/lidar_camera_fusion/results'
        )

        os.makedirs(
            results_dir,
            exist_ok=True
        )

        self.csv_path = os.path.join(
            results_dir,
            'fusion_comparison.csv'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow([
            'timestamp',
            'class',
            'confidence',

            'bbox_raw_points',
            'bbox_filtered_points',

            'mask_raw_points',
            'mask_filtered_points',

            'bbox_distance_m',
            'mask_distance_m',

            'bbox_centroid_x',
            'bbox_centroid_y',
            'bbox_centroid_z',

            'mask_centroid_x',
            'mask_centroid_y',
            'mask_centroid_z',

            'bbox_extent_x',
            'bbox_extent_y',
            'bbox_extent_z',

            'mask_extent_x',
            'mask_extent_y',
            'mask_extent_z',
        ])

        self.csv_file.flush()

        self.get_logger().info(
            f'Fusion node started | device={self.device}'
        )

        self.get_logger().info(
            f'CSV logging: {self.csv_path}'
        )

    # =================================================
    # Camera callbacks
    # =================================================

    def image_callback(self, msg):

        self.latest_image = (
            self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        )

        self.latest_header = msg.header

    def camera_info_callback(self, msg):

        if self.fx is not None:
            return

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        self.get_logger().info(
            f'Camera intrinsics loaded | '
            f'fx={self.fx:.2f}, '
            f'fy={self.fy:.2f}, '
            f'cx={self.cx:.2f}, '
            f'cy={self.cy:.2f}'
        )

    # =================================================
    # Quaternion -> rotation matrix
    # =================================================

    @staticmethod
    def quaternion_to_matrix(x, y, z, w):

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

    # =================================================
    # Depth clustering
    # =================================================

    def select_object_depth_cluster(
        self,
        points_camera
    ):

        if len(points_camera) < self.min_lidar_points:
            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        depth = points_camera[:, 2]

        order = np.argsort(depth)
        sorted_depth = depth[order]

        depth_diff = np.diff(
            sorted_depth
        )

        split_locations = (
            np.where(
                depth_diff >
                self.depth_gap_threshold
            )[0] + 1
        )

        clusters = np.split(
            order,
            split_locations
        )

        adaptive_min_points = max(
            self.min_lidar_points,
            int(
                0.05 *
                len(points_camera)
            )
        )

        valid_clusters = [
            cluster
            for cluster in clusters
            if len(cluster) >= adaptive_min_points
        ]

        if not valid_clusters:
            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        selected_cluster = min(
            valid_clusters,
            key=lambda cluster:
                np.median(
                    depth[cluster]
                )
        )

        selected_points = points_camera[
            selected_cluster
        ]

        selected_depth = (
            selected_points[:, 2]
        )

        median_depth = np.median(
            selected_depth
        )

        mad = np.median(
            np.abs(
                selected_depth -
                median_depth
            )
        )

        tolerance = max(
            0.20,
            2.5 * mad
        )

        mask = (
            np.abs(
                selected_depth -
                median_depth
            ) < tolerance
        )

        return selected_points[mask]

    # =================================================
    # 3D statistics
    # =================================================

    @staticmethod
    def calculate_statistics(points):

        if len(points) == 0:

            centroid = np.array([
                np.nan,
                np.nan,
                np.nan
            ])

            extent = np.array([
                np.nan,
                np.nan,
                np.nan
            ])

            distance = np.nan

            return (
                centroid,
                extent,
                distance
            )

        centroid = np.median(
            points,
            axis=0
        )

        distance = float(
            np.linalg.norm(
                centroid
            )
        )

        p_min = np.percentile(
            points,
            5,
            axis=0
        )

        p_max = np.percentile(
            points,
            95,
            axis=0
        )

        extent = (
            p_max -
            p_min
        )

        return (
            centroid,
            extent,
            distance
        )

    # =================================================
    # Main fusion callback
    # =================================================

    def lidar_callback(self, msg):

        if self.latest_image is None:
            return

        if self.fx is None:
            return

        frame = self.latest_image.copy()

        # ---------------------------------
        # YOLO-World
        # ---------------------------------
        det_results = self.detector.predict(
            frame,
            device=self.device,
            conf=0.15,
            verbose=False
        )

        det_result = det_results[0]

        # ---------------------------------
        # LiDAR
        # ---------------------------------
        points_lidar = (
            point_cloud2.read_points_numpy(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )
            .astype(
                np.float32,
                copy=False
            )
        )

        if points_lidar.size == 0:
            return

        # ---------------------------------
        # TF
        # ---------------------------------
        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    self.lidar_frame,
                    Time()
                )
            )

        except TransformException as ex:

            self.get_logger().warning(
                f'TF failed: {ex}'
            )

            return

        t = transform.transform.translation
        q = transform.transform.rotation

        rotation = (
            self.quaternion_to_matrix(
                q.x,
                q.y,
                q.z,
                q.w
            )
        )

        translation = np.array(
            [
                t.x,
                t.y,
                t.z
            ],
            dtype=np.float32
        )

        points_camera = (
            points_lidar @ rotation.T
        ) + translation

        # ---------------------------------
        # Points in front of camera
        # ---------------------------------
        front_mask = (
            points_camera[:, 2] > 0.1
        )

        points_camera = points_camera[
            front_mask
        ]

        if len(points_camera) == 0:
            return

        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]

        # ---------------------------------
        # 3D -> 2D projection
        # ---------------------------------
        u = (
            self.fx *
            x / z +
            self.cx
        )

        v = (
            self.fy *
            y / z +
            self.cy
        )

        height, width = (
            frame.shape[:2]
        )

        valid = (
            (u >= 0) &
            (u < width) &
            (v >= 0) &
            (v < height)
        )

        u = u[valid]
        v = v[valid]

        projected_points = (
            points_camera[valid]
        )

        fused_count = 0

        # ---------------------------------
        # Process YOLO detections
        # ---------------------------------
        if (
            det_result.boxes is not None
            and len(det_result.boxes) > 0
        ):

            boxes = (
                det_result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                det_result.boxes.cls
                .cpu()
                .numpy()
            )

            confidences = (
                det_result.boxes.conf
                .cpu()
                .numpy()
            )

            for (
                bbox,
                cls_id,
                confidence
            ) in zip(
                boxes,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = (
                    bbox.astype(int)
                )

                class_name = (
                    self.detector.names[
                        int(cls_id)
                    ]
                )

                # =================================
                # A. BBox association
                # =================================

                bbox_inside = (
                    (u >= x1) &
                    (u <= x2) &
                    (v >= y1) &
                    (v <= y2)
                )

                bbox_raw_points = (
                    projected_points[
                        bbox_inside
                    ]
                )

                bbox_filtered_points = (
                    self.select_object_depth_cluster(
                        bbox_raw_points
                    )
                )

                (
                    bbox_centroid,
                    bbox_extent,
                    bbox_distance
                ) = self.calculate_statistics(
                    bbox_filtered_points
                )

                # =================================
                # B. FastSAM mask
                # =================================

                sam_results = (
                    self.sam.predict(
                        frame,
                        device=self.device,
                        bboxes=[
                            [
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2)
                            ]
                        ],
                        verbose=False
                    )
                )

                mask_image = None

                if len(sam_results) > 0:

                    sam_result = (
                        sam_results[0]
                    )

                    if (
                        sam_result.masks
                        is not None
                        and len(
                            sam_result.masks.data
                        ) > 0
                    ):

                        mask_image = (
                            sam_result
                            .masks
                            .data[0]
                            .cpu()
                            .numpy()
                        )

                        mask_image = cv2.resize(
                            mask_image,
                            (width, height),
                            interpolation=(
                                cv2.INTER_NEAREST
                            )
                        )

                mask_raw_points = np.empty(
                    (0, 3),
                    dtype=np.float32
                )

                mask_filtered_points = np.empty(
                    (0, 3),
                    dtype=np.float32
                )

                if mask_image is not None:

                    ui = u.astype(
                        np.int32
                    )

                    vi = v.astype(
                        np.int32
                    )

                    mask_inside = (
                        mask_image[
                            vi,
                            ui
                        ] > 0.5
                    )

                    mask_raw_points = (
                        projected_points[
                            mask_inside
                        ]
                    )

                    mask_filtered_points = (
                        self.select_object_depth_cluster(
                            mask_raw_points
                        )
                    )

                (
                    mask_centroid,
                    mask_extent,
                    mask_distance
                ) = self.calculate_statistics(
                    mask_filtered_points
                )

                # ---------------------------------
                # Count fused object
                # ---------------------------------
                if (
                    len(mask_filtered_points)
                    >= self.min_lidar_points
                ):
                    fused_count += 1

                elif (
                    len(bbox_filtered_points)
                    >= self.min_lidar_points
                ):
                    fused_count += 1

                # =================================
                # Visualization
                # =================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                if np.isfinite(mask_distance):

                    label = (
                        f'{class_name} '
                        f'{confidence:.2f} | '
                        f'{mask_distance:.2f} m'
                    )

                elif np.isfinite(bbox_distance):

                    label = (
                        f'{class_name} '
                        f'{confidence:.2f} | '
                        f'{bbox_distance:.2f} m '
                        f'(BBox)'
                    )

                else:

                    label = (
                        f'{class_name} '
                        f'{confidence:.2f} | '
                        f'NO LIDAR'
                    )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            y1 - 8,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # ---------------------------------
                # FastSAM contour
                # ---------------------------------
                if mask_image is not None:

                    mask_uint8 = (
                        (
                            mask_image > 0.5
                        ).astype(
                            np.uint8
                        ) * 255
                    )

                    contours, _ = (
                        cv2.findContours(
                            mask_uint8,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                    )

                    cv2.drawContours(
                        frame,
                        contours,
                        -1,
                        (255, 0, 255),
                        1
                    )

                # =================================
                # CSV
                # =================================

                timestamp = (
                    self.get_clock()
                    .now()
                    .nanoseconds
                    / 1e9
                )

                self.csv_writer.writerow([
                    f'{timestamp:.6f}',
                    class_name,
                    f'{confidence:.4f}',

                    len(bbox_raw_points),
                    len(bbox_filtered_points),

                    len(mask_raw_points),
                    len(mask_filtered_points),

                    (
                        f'{bbox_distance:.4f}'
                        if np.isfinite(
                            bbox_distance
                        )
                        else ''
                    ),

                    (
                        f'{mask_distance:.4f}'
                        if np.isfinite(
                            mask_distance
                        )
                        else ''
                    ),

                    bbox_centroid[0],
                    bbox_centroid[1],
                    bbox_centroid[2],

                    mask_centroid[0],
                    mask_centroid[1],
                    mask_centroid[2],

                    bbox_extent[0],
                    bbox_extent[1],
                    bbox_extent[2],

                    mask_extent[0],
                    mask_extent[1],
                    mask_extent[2],
                ])

                self.csv_file.flush()

        # ---------------------------------
        # Global overlay
        # ---------------------------------
        cv2.putText(
            frame,
            f'Fused objects: {fused_count}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ---------------------------------
        # Publish
        # ---------------------------------
        output_msg = (
            self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )
        )

        if self.latest_header is not None:
            output_msg.header = (
                self.latest_header
            )

        self.publisher.publish(
            output_msg
        )

    # =================================================
    # Clean shutdown
    # =================================================

    def destroy_node(self):

        if hasattr(
            self,
            'csv_file'
        ):

            self.csv_file.flush()
            self.csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = FusionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
