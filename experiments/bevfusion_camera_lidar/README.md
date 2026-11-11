# BEVFusion Camera-LiDAR ROS 2 Integration

## Overview

This directory contains Python ROS 2 publisher scripts used to integrate
the nuScenes Mini dataset with Autoware's pretrained camera-LiDAR BEVFusion
3D object detection pipeline.

The scripts publish camera images, CameraInfo, LiDAR point clouds, and
static camera transforms. The sequence publisher also monitors detection
messages and records experiment results.

## Files

| File | Description |
|---|---|
| publish_nuscenes_lidar.py | Publishes a single nuScenes LiDAR sample repeatedly |
| publish_nuscenes_sequence.py | Publishes a 10-sample camera-LiDAR sequence and records detections |
| publish_camera_tf.py | Publishes static transforms for six camera optical frames |

## Environment

- Ubuntu 24.04.5 LTS under WSL2
- ROS 2 Jazzy
- Autoware Universe
- CUDA 13.0.3
- TensorRT 10.13.3.9
- nuScenes Mini (v1.0-mini)

Python dependencies include ROS 2 Python packages, NumPy, PyYAML,
Pillow, and the nuScenes devkit.

## Dataset

The scripts expect nuScenes Mini at:

~/datasets/nuscenes

Dataset version: v1.0-mini

The dataset is not included in this repository. Download it separately
and ensure its metadata and sensor files are installed correctly.

## ROS 2 Topics

### LiDAR

Topic:
`/sensing/lidar/concatenated/pointcloud`

Message type: `sensor_msgs/msg/PointCloud2`

Frame: `lidar_top`

The single-frame publisher loads `nusc.sample[0]` and republishes its
LIDAR_TOP point cloud at one-second intervals.

### Cameras

The sequence publisher publishes six camera image topics:

- `/sensing/camera/camera0/image_rect_color`
- `/sensing/camera/camera1/image_rect_color`
- `/sensing/camera/camera2/image_rect_color`
- `/sensing/camera/camera3/image_rect_color`
- `/sensing/camera/camera4/image_rect_color`
- `/sensing/camera/camera5/image_rect_color`

Corresponding CameraInfo topics use the same camera IDs:

`/sensing/camera/cameraN/camera_info`

Camera mapping:

| Camera ID | nuScenes camera |
|---|---|
| camera0 | CAM_FRONT |
| camera1 | CAM_FRONT_RIGHT |
| camera2 | CAM_FRONT_LEFT |
| camera3 | CAM_BACK |
| camera4 | CAM_BACK_LEFT |
| camera5 | CAM_BACK_RIGHT |

### Detection Output

Topic: `/perception/detection/objects`

Message type: `autoware_perception_msgs/msg/DetectedObjects`

## Camera Transforms

`publish_camera_tf.py` publishes six static transforms from `lidar_top`
to the camera optical frames.

Translation and quaternion values are derived from nuScenes calibrated
sensor records for the first nuScenes Mini sample.
## Running the Camera-LiDAR Experiment

Use three separate WSL terminals.

### Terminal 1: Launch BEVFusion

```bash
cd ~/autoware/autoware
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch autoware_bevfusion bevfusion.launch.xml \
  model_name:=bevfusion_camera_lidar \
  model_path:=$HOME/autoware_data/ml_models/bevfusion

### Terminal 2: Publish Camera Transforms
cd ~/CMPE249-Team2/experiments/bevfusion_camera_lidar

source /opt/ros/jazzy/setup.bash
source ~/autoware/autoware/install/setup.bash

python publish_camera_tf.py

### Terminal 3: Publish nuScenes Sequence
cd ~/CMPE249-Team2/experiments/bevfusion_camera_lidar

source /opt/ros/jazzy/setup.bash
source ~/autoware/autoware/install/setup.bash
source ~/nuscenes_venv/bin/activate

python publish_nuscenes_sequence.py

The sequence publisher:
1. Loads nuScenes Mini.
2. Follows the first scene's sample-token chain.
3. Processes the first 10 samples.
4. Publishes six camera images and CameraInfo messages.
5. Publishes the corresponding LiDAR point clouds.
6. Waits for camera readiness and allows a 1-second settling interval.
7. Waits for detections associated with the LiDAR timestamps.
8. Records predictions and frame execution status.

Sequence Configuration
Parameter	                Value
Number of samples	        10
Camera readiness timeout	5 seconds
Camera settling interval	1 second
Detection timeout	        10 seconds
LiDAR subscriber timeout	10 seconds
Final detection grace period	10 seconds
Detection cache maximum entries	10
Detection cache TTL	        30 seconds

Output Files
Each sequence execution creates a unique directory under:

~/nuscenes_bevfusion/sequence_runs/

Each run generates:
- sequence_predictions.yaml — serialized detection predictions.
- sequence_frame_status.csv — per-frame execution status and timing information.
Existing run directories are not overwritten.

## Baseline Experiment Results

The camera-LiDAR integration was tested on 10 samples from nuScenes Mini
scene-0061 across three consecutive runs.

| Measurement | Result |
|---|---:|
| Total frames across three runs | 30 |
| Successful frames | 30 |
| Timeouts | 0 |
| Late detections | 0 |
| Serialization errors | 0 |
| Predicted objects per run | 59 |
| Average predicted objects per frame | 5.9 |
| Average publisher-observed detection wait | Approximately 169.7 ms |

The 169.7 ms measurement is the publisher-observed wait for detection
messages. It is not an isolated neural-network inference latency measurement.

These results demonstrate repeatable runtime integration on the tested
sequence. They do not establish detection accuracy.

## Known Limitations

- Quantitative detection accuracy has not yet been evaluated against
  nuScenes ground-truth annotations.
- Some camera frames can be considered stale by the BEVFusion synchronization
  logic and masked during fusion. Successful detection output does not prove
  that all six camera views contributed to every frame.
- Camera preprocessing was adapted for 1600x900 nuScenes images.
  Equivalence to the original training preprocessing has not been established.
- Camera transforms are based on the first nuScenes Mini sample's calibration.
- The evaluation currently covers only 10 samples from one scene.
- Publisher-observed detection wait includes more than neural-network
  inference time.

## Reproducibility

The scripts in this directory are the versions used for the recorded baseline
experiment. Corresponding prediction and status files are preserved in:

`results/bevfusion_camera_lidar_baseline/`

See that directory's README and SHA256SUMS.txt for baseline details and
integrity verification.
