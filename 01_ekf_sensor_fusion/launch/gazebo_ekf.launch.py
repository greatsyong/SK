import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

import xacro


def generate_launch_description():

    pkg_dir = get_package_share_directory('ekf_sensor_fusion')

    xacro_file = os.path.join(
        pkg_dir,
        'description',
        'ekf_robot.urdf.xacro'
    )

    world_file = os.path.join(
        pkg_dir,
        'worlds',
        'ekf_test.world'
    )

    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world_file
        }.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True
            }
        ]
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',
            'robot_description',
            '-entity',
            'ekf_robot',
            '-x',
            '0.0',
            '-y',
            '0.0',
            '-z',
            '0.15'
        ],
        output='screen'
    )

    ekf = Node(
        package='ekf_sensor_fusion',
        executable='ekf_node',
        name='ekf_sensor_fusion',
        output='screen',
        parameters=[
            {
                'use_sim_time': True
            }
        ]
    )
    ground_truth = Node(
        package='ekf_sensor_fusion',
        executable='gazebo_ground_truth',
        name='gazebo_ground_truth',
        output='screen',
        parameters=[
            {
                'use_sim_time': True
            }
        ]
    )

    evaluation = Node(
        package='ekf_sensor_fusion',
        executable='evaluate',
        name='evaluation_node',
        output='screen',
        parameters=[
            {
                'use_sim_time': True
            }
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        ekf,
        ground_truth,
        evaluation
    ])