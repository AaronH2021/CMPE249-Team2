
# BEVFusion Camera-LiDAR Baseline

## Experiment
- Model: Pretrained BEVFusion Camera-LiDAR
- Framework: ROS 2 Jazzy / Autoware Universe
- Dataset: nuScenes Mini
- Scene: scene-0061
- Samples per run: 10
- Runtime optimization: TensorRT

## Validation Results

| Run | Successful Frames | Timeouts |
|---|---:|---:|
| Run 1 | 10/10 | 0 |
| Run 2 | 10/10 | 0 |
| Run 3 | 10/10 | 0 |
| Total | 30/30 | 0 |

- Late detections: 0
- Serialization errors: 0
- Average publisher-observed detection wait: 169.7 ms

## Configuration Notes
- Camera settling interval: 1.0 second
- Camera-LiDAR synchronization threshold: 120 ms
- Camera resolution: 1600x900
- LiDAR input: nuScenes LIDAR_TOP

## Archived Files

- predictions_run1.yaml
- predictions_run2.yaml
- predictions_run3.yaml
- status_run1.csv
- status_run2.csv
- status_run3.csv

## Important Notes
Successful inference does not guarantee all six camera
views contributed to every detection. Camera masks may
disable stale camera inputs.

This experiment validates ROS 2 integration and inference
execution, not ground-truth detection accuracy.

