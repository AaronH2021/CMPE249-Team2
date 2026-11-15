# Models

This directory documents the pretrained models used in the BEVFusion deployment and evaluation experiments.

## BEVFusion

**BEVFusion** is a multi-modal 3D object detection model designed for autonomous driving.

This project investigates two configurations:

### 1. Camera–LiDAR Fusion

Uses both camera images and LiDAR point clouds for multi-modal 3D object detection.

Experiment documentation and results:
- [`experiments/bevfusion_camera_lidar/`](../experiments/bevfusion_camera_lidar/)
- [`results/bevfusion_camera_lidar_baseline/`](../results/bevfusion_camera_lidar_baseline/)

### 2. LiDAR-Only BEVFusion

Uses LiDAR point cloud data for 3D object detection without camera input.

Experiment documentation and results:
- [`experiments/bevfusion_lidar_only/`](../experiments/bevfusion_lidar_only/)
- [`results/bevfusion_lidar_only_minival/`](../results/bevfusion_lidar_only_minival/)

## Model Weights and Configuration

Pretrained model checkpoints, ONNX files, and TensorRT engine files are not included in this repository.

Users must obtain compatible pretrained model weights and configure the appropriate local paths before running the experiments.

The LiDAR-only Autoware deployment uses the `bevfusion_lidar` model configuration.

## Notes

- Model weights are excluded from Git tracking because of their size and distribution considerations.
- TensorRT engines may depend on the GPU architecture, TensorRT version, and CUDA environment.
- Refer to the experiment-specific documentation for model setup and deployment instructions.
