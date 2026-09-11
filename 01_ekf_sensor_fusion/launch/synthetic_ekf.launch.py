from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    synthetic_sensor = Node(
        package='ekf_sensor_fusion',
        executable='synthetic_sensor',
        name='synthetic_sensor',
        output='screen'
    )

    ekf_node = Node(
        package='ekf_sensor_fusion',
        executable='ekf_node',
        name='ekf_sensor_fusion',
        output='screen'
    )

    evaluation_node = Node(
        package='ekf_sensor_fusion',
        executable='evaluate',
        name='evaluation_node',
        output='screen'
    )

    return LaunchDescription([
        synthetic_sensor,
        ekf_node,
        evaluation_node
    ])