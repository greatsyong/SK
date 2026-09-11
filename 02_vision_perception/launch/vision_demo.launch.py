from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    camera_node = Node(
        package='vision_perception',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen'
    )

    perception_node = Node(
        package='vision_perception',
        executable='unified_perception',
        name='unified_perception',
        output='screen'
    )

    return LaunchDescription([
        camera_node,
        perception_node
    ])