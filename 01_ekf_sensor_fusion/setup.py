import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ekf_sensor_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'description'),
            glob('description/*')
        ),
        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')
        ),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sooyong',
    maintainer_email='TODO',
    description='EKF sensor fusion project',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_sensor_fusion.ekf_node:main',
            'synthetic_sensor = ekf_sensor_fusion.synthetic_sensor_node:main',
            'evaluate = ekf_sensor_fusion.evaluation_node:main',
            'gazebo_ground_truth = ekf_sensor_fusion.gazebo_ground_truth_node:main',
        ],
    },
)