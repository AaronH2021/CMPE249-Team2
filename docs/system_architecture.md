# BEVFusion System Architecture

## 1. Project Overview

This project integrates pretrained BEVFusion 3D object detection models
into ROS 2 using Autoware Universe.

The system uses the nuScenes Mini dataset to provide camera images and
LiDAR point clouds to the BEVFusion perception pipeline.

Two configurations have been deployed:

1. LiDAR-only BEVFusion.
2. Camera-LiDAR BEVFusion.

The primary objective is to evaluate pretrained multimodal 3D object
detection models in a ROS 2 environment and investigate their detection
performance and runtime characteristics.

The implementation and experiments are conducted using Ubuntu 24.04
under WSL2, ROS 2 Jazzy, Autoware Universe, CUDA, and TensorRT.

## 2. High-Level Architecture

The camera-LiDAR processing pipeline consists of the following stages:

1. Dataset input: Load camera images and LiDAR point clouds from nuScenes Mini.
2. ROS 2 publishing: Convert dataset sensor data into ROS 2 messages.
3. Coordinate transforms: Publish static transforms between the LiDAR
   frame and six camera optical frames.
4. Sensor synchronization: Associate camera messages with corresponding
   LiDAR timestamps.
5. BEVFusion inference: Process sensor inputs using the pretrained
   BEVFusion model and TensorRT inference engines.
6. Detection output: Publish 3D object detections through ROS 2.

### 2.1 Camera-LiDAR Data Flow

```mermaid
flowchart TD
    A["nuScenes Mini Dataset"] --> B["Camera Images"]
    A --> C["LIDAR_TOP Point Clouds"]

    B --> D["ROS 2 Camera Publisher"]
    C --> E["ROS 2 LiDAR Publisher"]

    D --> F["Camera Images + CameraInfo"]
    E --> G["PointCloud2"]

    H["Static Camera TF Publisher"] --> I["TF Transforms"]

    F --> J["Autoware BEVFusion Node"]
    G --> J
    I --> J

    J --> K["Camera-LiDAR Preprocessing and Fusion"]
    K --> L["TensorRT Inference"]
    L --> M["3D Object Detection"]
    M --> N["/perception/detection/objects"]

The camera and LiDAR publishers operate as separate ROS 2 data sources.
The BEVFusion node receives the sensor messages and performs inference.

### Development Environment Component	Configuration
Operating system	Ubuntu 24.04.5 LTS
Host environment	Windows 10 with WSL2
ROS 2	Jazzy
Perception framework	Autoware Universe
CUDA	13.0.3
TensorRT	10.13.3.9
Dataset	nuScenes Mini (v1.0-mini)
Inference framework	TensorRT

The development environment uses NVIDIA GPU acceleration through WSL2.

### Dataset and Sensor Inputs
The project uses the nuScenes Mini dataset, version v1.0-mini.

The dataset is stored locally at:

~/datasets/nuscenes

### LiDAR Input

The LiDAR input is obtained from the nuScenes LIDAR_TOP sensor.

The original point-cloud records contain five float32 values per point:
- X coordinate
- Y coordinate
- Z coordinate
- Intensity
- Ring index
The ROS 2 publisher converts the point cloud into a
sensor_msgs/msg/PointCloud2 message.

The published frame is:

lidar_top

### Camera Inputs
The camera-LiDAR configuration uses six nuScenes camera sensors:
ROS camera ID	nuScenes sensor
camera0	CAM_FRONT
camera1	CAM_FRONT_RIGHT
camera2	CAM_FRONT_LEFT
camera3	CAM_BACK
camera4	CAM_BACK_LEFT
camera5	CAM_BACK_RIGHT

Each camera publishes:
- Image messages (sensor_msgs/msg/Image)
- Camera calibration messages (sensor_msgs/msg/CameraInfo)
The camera images used in the experiment have a resolution of 1600 x 900.

The camera preprocessing configuration was adapted for these image
dimensions. Equivalence to the original model training preprocessing
has not yet been established.

## 5. ROS 2 Integration

The system uses ROS 2 topics to transfer sensor data to the Autoware
BEVFusion perception node.

### 5.1 LiDAR Topic

| Property | Value |
|---|---|
| Topic | `/sensing/lidar/concatenated/pointcloud` |
| Message type | `sensor_msgs/msg/PointCloud2` |
| Frame | `lidar_top` |

### 5.2 Camera Topics

Camera images are published using the following topic pattern:

`/sensing/camera/cameraN/image_rect_color`

Camera calibration information is published using:

`/sensing/camera/cameraN/camera_info`

Here, N ranges from 0 to 5.

### 5.3 Coordinate Transforms

A static transform publisher provides six camera transforms.

The transform hierarchy is:

`lidar_top` -> camera optical frames

The translation and quaternion values are derived from the calibrated
sensor records of the first nuScenes Mini sample.

These transforms allow the perception system to relate camera
coordinates to the LiDAR coordinate frame.

### 5.4 Detection Output

The BEVFusion node publishes detected objects on:

`/perception/detection/objects`

Message type:

`autoware_perception_msgs/msg/DetectedObjects`

The sequence publisher subscribes to this topic to collect detection
messages and associate them with the corresponding LiDAR timestamps.

## 6. BEVFusion Inference Architecture

The project deploys pretrained BEVFusion models through Autoware Universe.

Two model configurations have been successfully deployed:

### 6.1 LiDAR-Only BEVFusion

The LiDAR-only configuration processes LiDAR point clouds without
camera image features.

This configuration provides a baseline for comparing the effect of
adding camera information to the perception pipeline.

### 6.2 Camera-LiDAR BEVFusion

The multimodal configuration receives:

- Six camera image streams.
- Six corresponding CameraInfo streams.
- LiDAR point clouds.
- Camera-to-LiDAR coordinate transforms.

The perception node performs sensor synchronization and preprocessing
before running the BEVFusion inference pipeline.

### 6.3 TensorRT

TensorRT is used to execute the pretrained BEVFusion inference models.

The camera-LiDAR model bundle includes:

- `bevfusion_camera_lidar.onnx`
- `bevfusion_image_backbone.onnx`
- Generated TensorRT engines.

The TensorRT engines were successfully generated and used for inference.

A detailed comparison of TensorRT precision modes, latency, throughput,
and resource utilization remains part of the planned optimization work.

## 7. Sensor Synchronization

Camera and LiDAR messages originate from different nuScenes sensor
timestamps.

The sequence publisher preserves the original sensor timestamps when
constructing ROS 2 messages.

The BEVFusion camera synchronization logic uses a maximum
camera-to-LiDAR timestamp difference of 120 milliseconds.

Camera validity is determined using the absolute timestamp difference:

`abs(lidar_timestamp - camera_timestamp) <= 120 ms`

Camera inputs outside this threshold can be masked during fusion.

The sequence publisher also implements:

- A 5-second camera readiness timeout.
- A 1-second camera settling interval.
- A 10-second detection timeout.

The 1-second settling interval allows ROS 2 callbacks to process camera
messages before the corresponding LiDAR sample is published.

Receiving all six camera messages at the publisher does not guarantee
that all six camera views contribute to every inference. The BEVFusion
node independently evaluates camera timestamp validity.

## 8. Experiment Execution

The system is executed using separate ROS 2 terminals.

Baseline Runtime Results
The camera-LiDAR configuration was tested using ten samples from
nuScenes Mini scene-0061 across three consecutive runs.
Measurement	Result
Total processed frames	30
Successful frames	30
Timeouts	0
Late detections	0
Serialization errors	0
Predicted objects per run	59
Average publisher-observed detection wait	169.7 ms

The detection wait is measured by the sequence publisher. It is not
an isolated measurement of neural-network inference latency.

These results demonstrate repeatable runtime execution on the tested
sequence. They do not establish detection accuracy.
10. Current Limitations and Future Work
The following work remains:
1. Quantitatively evaluate detection accuracy against nuScenes
   ground-truth annotations.
2. Compare LiDAR-only and camera-LiDAR detection performance.
3. Measure isolated inference latency and end-to-end processing latency.
4. Measure throughput, GPU utilization, GPU memory, and CPU utilization.
5. Investigate TensorRT optimization, including FP32 and FP16 comparisons.
6. Evaluate the impact of camera synchronization and preprocessing.
7. Expand evaluation beyond the initial ten samples.
The pretrained model's performance on nuScenes should be interpreted
as cross-dataset evaluation rather than reproduction of its original
training benchmark.
11. Related Repository Resources
- ROS 2 publisher scripts:
  experiments/bevfusion_camera_lidar/
- Camera-LiDAR baseline results:
  results/bevfusion_camera_lidar_baseline/
- Dataset setup:
  datasets/
- Model information:
  models/
