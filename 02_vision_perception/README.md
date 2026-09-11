# ROS2 Vision Perception Pipeline

A real-time ROS2 perception pipeline integrating object detection, multi-object tracking, and instance segmentation using a pretrained YOLO model with GPU acceleration.

## Overview

This project demonstrates a ROS2-based vision perception pipeline for robotics applications.

The pipeline processes live camera images and performs:

- Object detection
- Multi-object tracking
- Instance segmentation
- Real-time visualization
- FPS measurement
- Processing latency measurement
- Inference latency measurement
- Object count logging

The system was developed and tested on ROS2 Humble with GPU acceleration.

## System Architecture

```text
Camera
  |
  v
/camera/image_raw
  |
  v
Unified Perception Node
  |
  +--> Object Detection
  |
  +--> Multi-Object Tracking
  |
  +--> Instance Segmentation
  |
  +--> Performance Measurement
  |
  v
/vision/unified_image

Main Components
Camera Publisher

Publishes webcam frames as ROS2 sensor_msgs/Image messages.

Topic:

/camera/image_raw
YOLO Detection and Tracking

Uses a pretrained YOLO model for object detection and multi-object tracking.

Tracking state is maintained between frames using persistent tracking.

Instance Segmentation

Uses a pretrained YOLO segmentation model to generate object masks in addition to bounding boxes and class predictions.

Unified Perception Node

Combines:

Detection
Tracking
Instance segmentation
FPS monitoring
Processing latency monitoring
Inference latency monitoring
Object counting

Output topic:

/vision/unified_image
ROS2 Nodes
camera_publisher
yolo_detector
segmentation_node
unified_perception
Launch

The complete pipeline can be launched with:

ros2 launch vision_perception vision_demo.launch.py
Performance Results

Test duration: approximately 70 seconds

Total analyzed frames:

2136

Measured performance:

Metric	Result
Average FPS	31.95
Average Processing Latency	18.26 ms
P95 Processing Latency	24.87 ms
Average YOLO Inference Latency	8.70 ms
P95 YOLO Inference Latency	12.52 ms
Average Objects per Frame	1.83
Maximum Objects per Frame	11

The camera publisher was configured for approximately 30 Hz input.

The measured FPS therefore represents stable real-time processing of a 30 Hz image stream rather than the maximum theoretical model throughput.

Approximately 95% of frames were processed within 24.87 ms, which is below the 33.3 ms frame period of a 30 Hz input stream.

Hardware and Software
Hardware
NVIDIA GeForce RTX 5060 Laptop GPU
Software
Ubuntu 22.04
ROS2 Humble
Python 3.10
PyTorch 2.11
CUDA 12.8
Ultralytics YOLO
OpenCV
cv_bridge
pandas
matplotlib
JupyterLab
Project Structure
vision_perception/
├── launch/
│   └── vision_demo.launch.py
├── results/
│   └── vision_results.csv
├── vision_perception/
│   ├── __init__.py
│   ├── camera_publisher.py
│   ├── segmentation_node.py
│   ├── unified_perception.py
│   └── yolo_detector.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
Benchmark Analysis

Performance data is logged to:

results/vision_results.csv

Logged parameters include:

timestamp
fps
processing_latency_ms
inference_latency_ms
object_count

The recorded data was analyzed using JupyterLab.

Key observations:

Stable real-time operation at approximately 30 Hz
Average end-to-end processing latency of approximately 18 ms
Average YOLO inference latency below 10 ms
P95 processing latency below the 30 Hz frame interval
Successful tracking and segmentation under varying object counts
Notes

The project uses pretrained YOLO models and focuses on robotics perception system integration rather than model training.

The objective is to demonstrate practical ROS2 integration, real-time inference, tracking, segmentation, and quantitative performance evaluation.

Author

Sooyong Kim