# ROS 2 / Autoware BEVFusion Workspace

This directory contains patches for the Autoware BEVFusion integration
developed as part of the CMPE249 Team 2 project.

The patches preserve local source-code and configuration changes required
for the tested BEVFusion camera-LiDAR and LiDAR-only workflows.

## Source Versions

The patches were generated against these official release revisions:

| Component | Version | Commit |
|---|---|---|
| Autoware | 2.0.0 | 6f91fff |
| Autoware Universe | 0.53.1 | 815d3f1b5c |
| CUDA Blackboard | 0.4.0 | d40f8d4 |

## Patch Files

### 1. Autoware Universe

`patches/autoware_universe.patch`

Contains modifications to:

- `bevfusion_node.cpp`
  - Camera extrinsics retry logic.
  - Sensor-fusion readiness and cloud callback diagnostics.
- `bevfusion_node_utils.cpp`
  - Camera-LiDAR timestamp synchronization changes.
  - Detection publishing diagnostics.
- `bevfusion_camera_lidar.param.yaml`
  - Disables point-cloud densification using past frames.
  - Disables image undistortion.
- `bevfusion_lidar.param.yaml`
  - Disables point-cloud densification using past frames.

### 2. CUDA Blackboard

`patches/cuda_blackboard.patch`

Contains diagnostic logging in:

- `src/cuda_blackboard_subscriber.cpp`

## Applying the Patches

Clone or check out the corresponding repositories at the versions
listed above before applying the patches.

### Autoware Universe

From the Autoware Universe repository root:

```bash
git checkout 0.53.1
git apply --check /path/to/CMPE249-Team2/ros2_ws/patches/autoware_universe.patch
git apply /path/to/CMPE249-Team2/ros2_ws/patches/autoware_universe.patch

### CUDA Blackboard From the CUDA Blackboard repository root:
git checkout 0.4.0
git apply --check /path/to/CMPE249-Team2/ros2_ws/patches/cuda_blackboard.patch
git apply /path/to/CMPE249-Team2/ros2_ws/patches/cuda_blackboard.patch


Replace /path/to/CMPE249-Team2 with the actual location of the project repository.

Important: Apply patches to clean checkouts of the specified versions.
Do not apply them repeatedly to an already modified working tree.

Rebuilding
After applying both patches, return to the Autoware workspace root:
cd ~/autoware/autoware

source /opt/ros/jazzy/setup.bash

colcon build \
  --base-paths src \
  --packages-up-to autoware_bevfusion \
  --parallel-workers 2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release


Then source the workspace:
source install/setup.bash


Reproducibility Notes
- These patches preserve source and configuration modifications, not pretrained model weights or TensorRT engine files.
- The nuScenes dataset is not included.
- CUDA, TensorRT, ROS 2, and Autoware dependencies must be installed separately.
- Model configuration and dataset preprocessing must also be configured for the target environment.
- The patches document the modifications used in this project; they do not independently guarantee identical results on different hardware or software environments.
  EOF
