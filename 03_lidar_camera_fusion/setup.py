from setuptools import find_packages, setup

package_name = 'lidar_camera_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sooyong Kim',
    maintainer_email='sooyong@example.com',
    description='LiDAR-camera 3D perception and fusion using ROS 2 and Isaac Sim.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'projection_node = lidar_camera_fusion.projection_node:main',
            'cloud_stats = lidar_camera_fusion.cloud_stats:main',
            'preprocess_node = lidar_camera_fusion.preprocess_node:main',
            'clustering_node = lidar_camera_fusion.clustering_node:main',
            'yolo_world_node = lidar_camera_fusion.yolo_world_node:main',
        ],
    },
)
