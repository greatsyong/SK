from pathlib import Path as FilePath



import math

import time



import numpy as np

import rclpy



from rclpy.node import Node

from rclpy.qos import (

    QoSProfile,

    DurabilityPolicy,

    ReliabilityPolicy,

)



from geometry_msgs.msg import (

    Point,

    PoseStamped,

)



from nav_msgs.msg import (

    OccupancyGrid,

    Path as NavPath,

)





from visualization_msgs.msg import (

    Marker,

    MarkerArray,

)



from benchmark.metric_map_loader import (

    load_ros_metric_map,

)





# ============================================================

# CONFIG

# ============================================================



MAP_YAML = FilePath(

    "data/metric_maps/stech_lab/stech_lab_completed.yaml"

)



RESULT_DIR = FilePath(

    "results/stech_lab/scenario01"

)



WAFFLE_PATH_CSV = (

    RESULT_DIR

    / "scenario01_waffle_smoothed.csv"

)



RIDGEBACK_PATH_CSV = (

    RESULT_DIR

    / "scenario01_ridgeback_smoothed.csv"

)



WAFFLE_PROFILE_CSV = (

    RESULT_DIR

    / "scenario01_waffle_time_profile.csv"

)



RIDGEBACK_PROFILE_CSV = (

    RESULT_DIR

    / "scenario01_ridgeback_time_profile.csv"

)



FRAME_ID = "map"



PLAYBACK_SPEED = 8.0



LOOP_ANIMATION = True





ROBOTS = {



    "waffle": {

        "name":

            "TurtleBot3 Waffle Pi",



        "length_m":

            0.281,



        "width_m":

            0.306,



        "height_m":

            0.141,

    },



    "ridgeback": {

        "name":

            "Clearpath Ridgeback",



        "length_m":

            0.960,



        "width_m":

            0.793,



        "height_m":

            0.311,

    },

}





# ============================================================

# UTILITIES

# ============================================================



def quaternion_from_yaw(

    yaw,

):

    """

    Return quaternion tuple (x, y, z, w)

    for planar yaw.

    """



    half = (

        0.5

        * yaw

    )



    return (

        0.0,

        0.0,

        math.sin(half),

        math.cos(half),

    )





def image_cell_to_world(

    x_cell,

    y_cell,

    width,

    height,

    resolution,

    origin,

):

    """

    Convert image coordinates:



        x -> right

        y -> down



    to ROS map coordinates:



        x -> right

        y -> up



    YAML origin corresponds to lower-left map origin.

    """



    origin_x = origin[0]

    origin_y = origin[1]



    x_world = (

        origin_x

        +

        (

            x_cell

            + 0.5

        )

        * resolution

    )



    y_world = (

        origin_y

        +

        (

            height

            - y_cell

            - 0.5

        )

        * resolution

    )



    return (

        x_world,

        y_world,

    )





def load_xy_csv(

    path,

):

    return np.loadtxt(

        path,

        delimiter=",",

        skiprows=1,

    )





def load_time_profile(

    path,

):

    return np.genfromtxt(

        path,

        delimiter=",",

        names=True,

    )





def compute_arc_length(

    points,

    resolution,

):

    points_m = (

        np.asarray(

            points,

            dtype=float,

        )

        * resolution

    )



    delta = np.diff(

        points_m,

        axis=0,

    )



    ds = np.linalg.norm(

        delta,

        axis=1,

    )



    s = np.concatenate(

        (

            [0.0],

            np.cumsum(ds),

        )

    )



    return s





def interpolate_polyline_at_s(

    points,

    s_values,

    target_s,

):

    """

    Interpolate cell-space polyline by arc length.

    """



    if target_s <= s_values[0]:

        return np.array(

            points[0],

            dtype=float,

        )



    if target_s >= s_values[-1]:

        return np.array(

            points[-1],

            dtype=float,

        )



    index = np.searchsorted(

        s_values,

        target_s,

    )



    i0 = index - 1

    i1 = index



    s0 = s_values[

        i0

    ]



    s1 = s_values[

        i1

    ]



    if (

        s1

        - s0

        < 1e-12

    ):

        return np.array(

            points[i0],

            dtype=float,

        )



    ratio = (

        target_s

        - s0

    ) / (

        s1

        - s0

    )



    p0 = np.asarray(

        points[i0],

        dtype=float,

    )



    p1 = np.asarray(

        points[i1],

        dtype=float,

    )



    return (

        p0

        +

        ratio

        * (

            p1

            - p0

        )

    )





def tangent_yaw(

    points,

    s_values,

    target_s,

):

    """

    Estimate tangent orientation at current arc length.



    Important:

    image y increases downward,

    ROS world y increases upward,

    therefore dy sign is flipped.

    """



    index = np.searchsorted(

        s_values,

        target_s,

    )



    index = int(

        np.clip(

            index,

            1,

            len(points) - 1,

        )

    )



    p0 = points[

        index - 1

    ]



    p1 = points[

        index

    ]



    dx = (

        p1[0]

        - p0[0]

    )



    dy_ros = -(

        p1[1]

        - p0[1]

    )



    return math.atan2(

        dy_ros,

        dx,

    )





# ============================================================

# NODE

# ============================================================



class Scenario01RVizNode(

    Node

):



    def __init__(

        self,

    ):



        super().__init__(

            "scenario01_rviz"

        )



        self.get_logger().info(

            "Loading Scenario 01..."

        )



        # ----------------------------------------------------

        # Load map

        # ----------------------------------------------------



        self.map_data = (

            load_ros_metric_map(

                MAP_YAML

            )

        )



        self.grid = (

            self.map_data[

                "grid"

            ]

        )



        self.unknown_mask = (

            self.map_data[

                "unknown_mask"

            ]

        )



        self.resolution = (

            self.map_data[

                "resolution"

            ]

        )



        self.origin = (

            self.map_data[

                "origin"

            ]

        )



        self.width = (

            self.map_data[

                "width"

            ]

        )



        self.height = (

            self.map_data[

                "height"

            ]

        )



        # ----------------------------------------------------

        # Load trajectories

        # ----------------------------------------------------



        self.waffle_path = (

            load_xy_csv(

                WAFFLE_PATH_CSV

            )

        )



        self.ridge_path = (

            load_xy_csv(

                RIDGEBACK_PATH_CSV

            )

        )



        self.waffle_profile = (

            load_time_profile(

                WAFFLE_PROFILE_CSV

            )

        )



        self.ridge_profile = (

            load_time_profile(

                RIDGEBACK_PROFILE_CSV

            )

        )



        self.waffle_s = (

            compute_arc_length(

                self.waffle_path,

                self.resolution,

            )

        )



        self.ridge_s = (

            compute_arc_length(

                self.ridge_path,

                self.resolution,

            )

        )



        self.waffle_total_time = float(

            self.waffle_profile[

                "time_s"

            ][-1]

        )



        self.ridge_total_time = float(

            self.ridge_profile[

                "time_s"

            ][-1]

        )



        # ----------------------------------------------------

        # QoS

        # ----------------------------------------------------



        static_qos = QoSProfile(

            depth=1,

        )



        static_qos.reliability = (

            ReliabilityPolicy.RELIABLE

        )



        static_qos.durability = (

            DurabilityPolicy.TRANSIENT_LOCAL

        )



        # ----------------------------------------------------

        # Publishers

        # ----------------------------------------------------



        self.map_pub = (

            self.create_publisher(

                OccupancyGrid,

                "/stech_lab/map",

                static_qos,

            )

        )



        self.waffle_path_pub = (

            self.create_publisher(

                NavPath,

                "/scenario01/waffle/path",

                static_qos,

            )

        )



        self.ridge_path_pub = (

            self.create_publisher(

                NavPath,

                "/scenario01/ridgeback/path",

                static_qos,

            )

        )



        self.static_marker_pub = (

            self.create_publisher(

                MarkerArray,

                "/scenario01/static_markers",

                static_qos,

            )

        )



        self.robot_marker_pub = (

            self.create_publisher(

                MarkerArray,

                "/scenario01/robots",

                10,

            )

        )



        # ----------------------------------------------------

        # Publish static content

        # ----------------------------------------------------



        self.publish_map()



        self.publish_paths()



        self.publish_static_markers()



        # ----------------------------------------------------

        # Animation

        # ----------------------------------------------------



        self.animation_start_wall = (

            time.monotonic()

        )



        self.timer = (

            self.create_timer(

                0.05,

                self.animation_callback,

            )

        )



        self.get_logger().info(

            "Scenario 01 RViz publisher ready."

        )



        self.get_logger().info(

            f"Waffle mission time: "

            f"{self.waffle_total_time:.3f} s"

        )



        self.get_logger().info(

            f"Ridgeback mission time: "

            f"{self.ridge_total_time:.3f} s"

        )



        self.get_logger().info(

            f"Playback speed: "

            f"{PLAYBACK_SPEED:.1f}x"

        )





    # ========================================================

    # MAP

    # ========================================================



    def publish_map(

        self,

    ):



        msg = OccupancyGrid()



        msg.header.frame_id = (

            FRAME_ID

        )



        msg.header.stamp = (

            self.get_clock()

            .now()

            .to_msg()

        )



        msg.info.resolution = (

            float(

                self.resolution

            )

        )



        msg.info.width = (

            int(

                self.width

            )

        )



        msg.info.height = (

            int(

                self.height

            )

        )



        msg.info.origin.position.x = (

            float(

                self.origin[0]

            )

        )



        msg.info.origin.position.y = (

            float(

                self.origin[1]

            )

        )



        msg.info.origin.position.z = (

            0.0

        )



        msg.info.origin.orientation.w = (

            1.0

        )



        # ROS OccupancyGrid row 0 corresponds to

        # bottom map row, so image grid must be flipped.



        occupied = np.zeros(

            self.grid.shape,

            dtype=np.int8,

        )



        occupied[

            self.grid != 0

        ] = 100



        occupied[

            self.unknown_mask

        ] = -1



        ros_grid = np.flipud(

            occupied

        )



        msg.data = (

            ros_grid

            .flatten()

            .astype(int)

            .tolist()

        )



        self.map_pub.publish(

            msg

        )





    # ========================================================

    # PATH

    # ========================================================



    def build_path_msg(

        self,

        points,

    ):



        msg = NavPath()



        msg.header.frame_id = (

            FRAME_ID

        )



        msg.header.stamp = (

            self.get_clock()

            .now()

            .to_msg()

        )



        for i, point in enumerate(

            points

        ):



            x_world, y_world = (

                image_cell_to_world(

                    point[0],

                    point[1],

                    self.width,

                    self.height,

                    self.resolution,

                    self.origin,

                )

            )



            pose = PoseStamped()



            pose.header = (

                msg.header

            )



            pose.pose.position.x = (

                float(

                    x_world

                )

            )



            pose.pose.position.y = (

                float(

                    y_world

                )

            )



            pose.pose.position.z = (

                0.03

            )



            if i < (

                len(points)

                - 1

            ):



                dx = (

                    points[

                        i + 1

                    ][0]

                    -

                    point[0]

                )



                dy = -(

                    points[

                        i + 1

                    ][1]

                    -

                    point[1]

                )



                yaw = math.atan2(

                    dy,

                    dx,

                )



            elif i > 0:



                dx = (

                    point[0]

                    -

                    points[

                        i - 1

                    ][0]

                )



                dy = -(

                    point[1]

                    -

                    points[

                        i - 1

                    ][1]

                )



                yaw = math.atan2(

                    dy,

                    dx,

                )



            else:



                yaw = 0.0



            qx, qy, qz, qw = (

                quaternion_from_yaw(

                    yaw

                )

            )



            pose.pose.orientation.x = qx

            pose.pose.orientation.y = qy

            pose.pose.orientation.z = qz

            pose.pose.orientation.w = qw



            msg.poses.append(

                pose

            )



        return msg





    def publish_paths(

        self,

    ):



        self.waffle_path_pub.publish(

            self.build_path_msg(

                self.waffle_path

            )

        )



        self.ridge_path_pub.publish(

            self.build_path_msg(

                self.ridge_path

            )

        )





    # ========================================================

    # STATIC MARKERS

    # ========================================================



    def make_sphere_marker(

        self,

        marker_id,

        x,

        y,

        z,

        scale,

        r,

        g,

        b,

        namespace,

    ):



        marker = Marker()



        marker.header.frame_id = (

            FRAME_ID

        )



        marker.header.stamp = (

            self.get_clock()

            .now()

            .to_msg()

        )



        marker.ns = namespace

        marker.id = marker_id



        marker.type = (

            Marker.SPHERE

        )



        marker.action = (

            Marker.ADD

        )



        marker.pose.position.x = (

            float(x)

        )



        marker.pose.position.y = (

            float(y)

        )



        marker.pose.position.z = (

            float(z)

        )



        marker.pose.orientation.w = (

            1.0

        )



        marker.scale.x = scale

        marker.scale.y = scale

        marker.scale.z = scale



        marker.color.r = r

        marker.color.g = g

        marker.color.b = b

        marker.color.a = 1.0



        return marker





    def make_text_marker(

        self,

        marker_id,

        text,

        x,

        y,

        z,

    ):



        marker = Marker()



        marker.header.frame_id = (

            FRAME_ID

        )



        marker.header.stamp = (

            self.get_clock()

            .now()

            .to_msg()

        )



        marker.ns = (

            "scenario_metrics"

        )



        marker.id = marker_id



        marker.type = (

            Marker.TEXT_VIEW_FACING

        )



        marker.action = (

            Marker.ADD

        )



        marker.pose.position.x = x

        marker.pose.position.y = y

        marker.pose.position.z = z



        marker.pose.orientation.w = (

            1.0

        )



        marker.scale.z = (

            0.65

        )



        marker.color.r = (

            1.0

        )



        marker.color.g = (

            1.0

        )



        marker.color.b = (

            1.0

        )



        marker.color.a = (

            1.0

        )



        marker.text = text



        return marker





    def publish_static_markers(

        self,

    ):



        markers = MarkerArray()



        # Start

        start = (

            self.waffle_path[0]

        )



        sx, sy = (

            image_cell_to_world(

                start[0],

                start[1],

                self.width,

                self.height,

                self.resolution,

                self.origin,

            )

        )



        markers.markers.append(

            self.make_sphere_marker(

                marker_id=0,

                x=sx,

                y=sy,

                z=0.20,

                scale=0.45,

                r=0.1,

                g=1.0,

                b=0.1,

                namespace="start_goal",

            )

        )



        # Goal

        goal = (

            self.waffle_path[-1]

        )



        gx, gy = (

            image_cell_to_world(

                goal[0],

                goal[1],

                self.width,

                self.height,

                self.resolution,

                self.origin,

            )

        )



        markers.markers.append(

            self.make_sphere_marker(

                marker_id=1,

                x=gx,

                y=gy,

                z=0.20,

                scale=0.55,

                r=1.0,

                g=0.1,

                b=0.1,

                namespace="start_goal",

            )

        )



        # ----------------------------------------------------

        # Metrics text

        # ----------------------------------------------------



        text = (

            "Scenario 01\n"

            "\n"

            f"Waffle\n"

            f"Path: {self.waffle_s[-1]:.2f} m\n"

            f"Time: {self.waffle_total_time:.1f} s\n"

            "\n"

            f"Ridgeback\n"

            f"Path: {self.ridge_s[-1]:.2f} m\n"

            f"Time: {self.ridge_total_time:.1f} s"

        )



        text_x = (

            self.origin[0]

            + 4.0

        )



        text_y = (

            self.origin[1]

            + self.height

            * self.resolution

            - 3.0

        )



        markers.markers.append(

            self.make_text_marker(

                marker_id=0,

                text=text,

                x=text_x,

                y=text_y,

                z=1.5,

            )

        )



        self.static_marker_pub.publish(

            markers

        )





    # ========================================================

    # ROBOT FOOTPRINT MARKER

    # ========================================================



    def make_robot_marker(

        self,

        marker_id,

        namespace,

        robot,

        x,

        y,

        yaw,

        r,

        g,

        b,

    ):



        marker = Marker()



        marker.header.frame_id = (

            FRAME_ID

        )



        marker.header.stamp = (

            self.get_clock()

            .now()

            .to_msg()

        )



        marker.ns = namespace

        marker.id = marker_id



        marker.type = (

            Marker.CUBE

        )



        marker.action = (

            Marker.ADD

        )



        marker.pose.position.x = (

            float(x)

        )



        marker.pose.position.y = (

            float(y)

        )



        marker.pose.position.z = (

            robot[

                "height_m"

            ]

            / 2.0

        )



        (

            qx,

            qy,

            qz,

            qw,

        ) = quaternion_from_yaw(

            yaw

        )



        marker.pose.orientation.x = qx

        marker.pose.orientation.y = qy

        marker.pose.orientation.z = qz

        marker.pose.orientation.w = qw



        marker.scale.x = (

            robot[

                "length_m"

            ]

        )



        marker.scale.y = (

            robot[

                "width_m"

            ]

        )



        marker.scale.z = (

            robot[

                "height_m"

            ]

        )



        marker.color.r = r

        marker.color.g = g

        marker.color.b = b

        marker.color.a = 0.80



        return marker





    # ========================================================

    # TIME -> ARC LENGTH

    # ========================================================



    def profile_s_at_time(

        self,

        profile,

        target_time,

    ):



        times = profile[

            "time_s"

        ]



        distances = profile[

            "s_m"

        ]



        if target_time <= 0.0:

            return 0.0



        if target_time >= times[-1]:

            return float(

                distances[-1]

            )



        return float(

            np.interp(

                target_time,

                times,

                distances,

            )

        )





    # ========================================================

    # ANIMATION

    # ========================================================



    def animation_callback(

        self,

    ):



        elapsed_wall = (

            time.monotonic()

            -

            self.animation_start_wall

        )



        simulated_time = (

            elapsed_wall

            * PLAYBACK_SPEED

        )



        max_time = max(

            self.waffle_total_time,

            self.ridge_total_time,

        )



        if (

            LOOP_ANIMATION

            and

            simulated_time

            > max_time + 3.0

        ):



            self.animation_start_wall = (

                time.monotonic()

            )



            simulated_time = 0.0



        waffle_time = min(

            simulated_time,

            self.waffle_total_time,

        )



        ridge_time = min(

            simulated_time,

            self.ridge_total_time,

        )



        waffle_s = (

            self.profile_s_at_time(

                self.waffle_profile,

                waffle_time,

            )

        )



        ridge_s = (

            self.profile_s_at_time(

                self.ridge_profile,

                ridge_time,

            )

        )



        waffle_cell = (

            interpolate_polyline_at_s(

                self.waffle_path,

                self.waffle_s,

                waffle_s,

            )

        )



        ridge_cell = (

            interpolate_polyline_at_s(

                self.ridge_path,

                self.ridge_s,

                ridge_s,

            )

        )



        waffle_yaw = (

            tangent_yaw(

                self.waffle_path,

                self.waffle_s,

                waffle_s,

            )

        )



        ridge_yaw = (

            tangent_yaw(

                self.ridge_path,

                self.ridge_s,

                ridge_s,

            )

        )



        waffle_x, waffle_y = (

            image_cell_to_world(

                waffle_cell[0],

                waffle_cell[1],

                self.width,

                self.height,

                self.resolution,

                self.origin,

            )

        )



        ridge_x, ridge_y = (

            image_cell_to_world(

                ridge_cell[0],

                ridge_cell[1],

                self.width,

                self.height,

                self.resolution,

                self.origin,

            )

        )



        markers = MarkerArray()



        # Waffle

        markers.markers.append(

            self.make_robot_marker(

                marker_id=0,

                namespace="waffle_robot",

                robot=ROBOTS[

                    "waffle"

                ],

                x=waffle_x,

                y=waffle_y,

                yaw=waffle_yaw,

                r=0.1,

                g=0.45,

                b=1.0,

            )

        )



        # Ridgeback

        markers.markers.append(

            self.make_robot_marker(

                marker_id=0,

                namespace="ridgeback_robot",

                robot=ROBOTS[

                    "ridgeback"

                ],

                x=ridge_x,

                y=ridge_y,

                yaw=ridge_yaw,

                r=1.0,

                g=0.55,

                b=0.1,

            )

        )



        self.robot_marker_pub.publish(

            markers

        )





# ============================================================

# MAIN

# ============================================================



def main(

    args=None,

):



    rclpy.init(

        args=args

    )



    node = (

        Scenario01RVizNode()

    )



    try:



        rclpy.spin(

            node

        )



    except KeyboardInterrupt:



        pass



    finally:



        node.destroy_node()



        rclpy.shutdown()





if __name__ == "__main__":

    main()