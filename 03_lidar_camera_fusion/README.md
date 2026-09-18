# LiDAR-Camera 3D Perception & Fusion

A ROS 2 multi-modal perception pipeline combining RGB semantic perception with 3D LiDAR geometry in an NVIDIA Isaac Sim warehouse environment.

The project integrates open-vocabulary object detection, prompted instance segmentation, calibrated LiDAR-to-camera projection, object-level point association, depth clustering, and robust 3D estimation. The system was evaluated under controlled viewpoint, range, and extrinsic-calibration perturbations.

## 1. Project Objective

RGB cameras provide dense semantic information but do not directly measure metric depth. LiDAR provides accurate 3D geometry but lacks rich semantic interpretation.

This project combines the complementary strengths of both sensors to answer three practical robotics questions:

1. Can LiDAR points be projected reliably into the camera image using calibrated sensor geometry?
2. Can camera detections be associated with geometrically consistent LiDAR returns to estimate object-level range and 3D structure?
3. How does fusion quality change under different viewpoints, sensing ranges, and extrinsic-calibration errors?

The goal was not simply to visualize projected points, but to build and evaluate a complete object-level LiDAR-camera fusion pipeline.

## 2. System Overview

### Camera branch

```text
RGB Camera
    ↓
YOLO-World
    ↓
Open-vocabulary object detections
    ↓
FastSAM
    ↓
Prompted instance masks
```

### LiDAR branch

```text
3D LiDAR
    ↓
PointCloud2
    ↓
LiDAR-to-camera TF transform
    ↓
Camera-frame 3D points
    ↓
Image-plane projection
```

The two branches are combined through object-level geometric association:

```text
Camera detections / masks
            +
Projected LiDAR points
            ↓
BBox or mask association
            ↓
Depth-gap clustering
            ↓
MAD-based robust filtering
            ↓
Object centroid / extent / range
```

## 3. Platform and Environment

### Simulation
- NVIDIA Isaac Sim 6.1
- Nova Carter
- Warehouse environment
- Front RGB camera
- Front 3D LiDAR

### Software
- Ubuntu 22.04
- ROS 2 Humble
- Python
- NumPy
- OpenCV
- PyTorch
- CUDA
- Ultralytics YOLO-World
- FastSAM
- TF2
- `sensor_msgs/msg/PointCloud2`

## 4. Camera-LiDAR Geometry

A LiDAR point expressed in the LiDAR coordinate frame is transformed into the camera frame using the rigid-body extrinsic transform:

```text
p_C = R_CL p_L + t_CL
```

Only points with positive camera-frame depth are retained.

The transformed point is projected into the image plane using the camera intrinsic parameters:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

This establishes the geometric relationship between 3D LiDAR measurements and 2D camera detections.

## 5. Perception Strategy

### 5.1 YOLO-World

A standard COCO-based detector was initially evaluated, but warehouse-specific objects were not represented reliably by the fixed class set.

YOLO-World was therefore used with an application-specific open vocabulary including cardboard boxes, pallets, warehouse racks, traffic cones, warning signs, forklifts, warehouse carts, trucks, barrels, and people.

### 5.2 FastSAM

YOLO-World bounding boxes are used as prompts for FastSAM. Two LiDAR association strategies are evaluated:

- **BBox association** — projected LiDAR points inside the detector bounding box
- **Mask association** — projected LiDAR points inside the FastSAM object mask

Mask-based association can reduce background contamination when segmentation is accurate, but it is not universally superior to bounding-box association.

## 6. Depth Association

A 2D detection can contain LiDAR returns from multiple physical surfaces. The implemented method:

1. collects candidate LiDAR points inside the detection region,
2. sorts points by camera-frame depth,
3. separates clusters when consecutive depth gaps exceed a threshold,
4. rejects insufficiently supported clusters,
5. selects the nearest valid depth cluster,
6. applies median-absolute-deviation filtering,
7. computes robust object statistics.

The final measurements include object range, 3D centroid, object extent, raw LiDAR support, and filtered LiDAR support.

## 7. Experimental Evaluation

The fusion pipeline was evaluated under controlled geometric changes.

### 7.1 Baseline

The nominal robot pose and calibrated LiDAR-camera transform were used as the reference condition.

### 7.2 Viewpoint Variation

The robot was rotated relative to the scene at yaw -30 deg and yaw +30 deg around the baseline heading.

### 7.3 Range Variation

The robot was returned to the baseline heading and moved approximately 3 m farther from the observed scene.

### 7.4 Extrinsic Calibration Perturbation

Artificial LiDAR-to-camera yaw error was introduced at 0 deg, 1 deg, 3 deg, and 5 deg while all other processing parameters were held constant.

## 8. Main Observations

- The fusion pipeline continued to operate under controlled viewpoint changes.
- Increasing range substantially reduced the available LiDAR support per object.
- FastSAM masks improved association for some compact objects but were not universally superior to bounding boxes.
- Extrinsic yaw error reduced geometrically consistent LiDAR support.
- Larger calibration errors could produce incorrect surface reassociation rather than only reducing point count.

## 9. Perception Failure Case

During testing, warehouse shelving was occasionally classified as `truck` by YOLO-World. These detections were excluded from the quantitative fusion conclusions because the upstream semantic label was incorrect.

This demonstrates that an otherwise geometrically valid fusion result can still be semantically incorrect if the upstream detector provides the wrong object hypothesis.

## 10. Package Structure

```text
03_lidar_camera_fusion/
├── lidar_camera_fusion/
│   ├── __init__.py
│   ├── cloud_stats.py
│   ├── clustering_node.py
│   ├── fusion_node.py
│   ├── fusion_node_extrinsic.py
│   ├── preprocess_node.py
│   ├── projection_node.py
│   └── yolo_world_node.py
├── results/
│   ├── fusion_comparison.csv
│   ├── fusion_yaw_minus30.csv
│   ├── fusion_extrinsic_yaw_p0p0deg.csv
│   ├── fusion_extrinsic_yaw_p1p0deg.csv
│   ├── fusion_extrinsic_yaw_p3p0deg.csv
│   └── fusion_extrinsic_yaw_p5p0deg.csv
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## 11. Experimental Data

The repository includes representative CSV logs generated during the fusion experiments.

Available logs include:

- baseline / comparison run,
- yaw -30 deg viewpoint test,
- extrinsic yaw perturbation at 0 deg,
- extrinsic yaw perturbation at 1 deg,
- extrinsic yaw perturbation at 3 deg,
- extrinsic yaw perturbation at 5 deg.

Additional controlled experiments were performed during development, including yaw +30 deg and the +3 m range test. Some intermediate CSV files were overwritten during iterative testing and are therefore not included in the repository.

## 12. Running the Fusion Node

Activate ROS 2 and the Python virtual environment:

```bash
source /opt/ros/humble/setup.bash
source ~/robotics_ws/vision_venv/bin/activate
cd ~/robotics_ws
source install/setup.bash
```

Run the standard fusion node:

```bash
python ~/robotics_ws/src/lidar_camera_fusion/lidar_camera_fusion/fusion_node.py
```

Visualize the output with `rqt_image_view` and select `/fusion/fused_image`.

## 13. Extrinsic Perturbation Experiment

Example:

```bash
python ~/robotics_ws/src/lidar_camera_fusion/lidar_camera_fusion/fusion_node_extrinsic.py \
  --ros-args -p extrinsic_yaw_error_deg:=0.0
```

The tested perturbations were 0.0, 1.0, 3.0, and 5.0 degrees.

## 14. Engineering Lessons

This project highlighted several practical sensor-fusion lessons:

- accurate geometry does not guarantee correct semantics,
- a 2D bounding box can contain multiple 3D surfaces,
- segmentation masks can improve or degrade LiDAR association depending on mask quality,
- range directly affects available LiDAR support,
- extrinsic calibration error can create incorrect surface reassociation,
- robustness should be evaluated under controlled perturbations rather than only through a visually correct baseline demonstration.

## 15. Related Material

Portfolio: https://sooyongtech.dev

GitHub: https://github.com/greatsyong/SK

A detailed technical report and demonstration video are also prepared as part of the portfolio project.

## Author

**Sooyong Kim**

Robotics & Autonomous Systems
