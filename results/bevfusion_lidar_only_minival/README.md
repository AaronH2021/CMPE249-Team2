# BEVFusion LiDAR-Only Evaluation — nuScenes Mini

## Overview

This directory contains evaluation artifacts for deploying a pretrained LiDAR-only BEVFusion model through the Autoware ROS 2 perception pipeline.

The experiment uses the nuScenes v1.0-mini dataset and evaluates 81 frames from two scenes. This is a custom sequence evaluation, not an official nuScenes benchmark submission.

## Experiment Summary

| Item | Value |
|---|---|
| Model | Pretrained LiDAR-only BEVFusion |
| Dataset | nuScenes v1.0-mini |
| Evaluated frames | 81 |
| Unique timestamp-matched samples | 81 |
| Scenes | 2 |
| Evaluation method | Custom class-aware greedy one-to-one IoU matching |
| IoU thresholds | 0.25 and 0.50 |
| Official nuScenes mAP/NDS | Not computed |

## Detection Results

Aggregate class-aware results using all supported ground-truth objects:

| Metric | IoU | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEV | 0.25 | 1577 | 266 | 2566 | 0.8557 | 0.3806 | 0.5269 |
| BEV | 0.50 | 1378 | 465 | 2765 | 0.7477 | 0.3326 | 0.4604 |
| 3D | 0.25 | 1516 | 327 | 2627 | 0.8226 | 0.3659 | 0.5065 |
| 3D | 0.50 | 1075 | 768 | 3068 | 0.5833 | 0.2595 | 0.3592 |

The JSON summary also contains class-agnostic results, in-range results, and per-class metrics.

Precision is higher than recall at both IoU thresholds. Increasing the IoU threshold reduces measured performance, particularly for 3D IoU.

These are custom evaluator results, not official nuScenes mAP or NDS metrics.

## Deployment Reliability

| Measure | Result |
|---|---:|
| Frames published | 81 |
| Successful detection callbacks | 81 |
| Timeouts | 0 |
| Late detections recovered | 0 |
| Serialization errors | 0 |
| Duplicate or unmatched timestamps | 0 |

Exact timestamp matching establishes the association between predictions and unique dataset samples; it does not independently validate bounding-box coordinate correctness.

## System Performance

| Measure | Observed value |
|---|---:|
| Sequence throughput | Approximately 2.55 frames/s |
| Sequence duration | Approximately 31.83 seconds |
| Peak sampled GPU utilization | 63% |
| Peak sampled GPU memory | 818 MiB |
| Peak sampled GPU temperature | 46–49 °C |
| Peak sampled GPU power | Approximately 45.8 W |
| Minimum observed free system RAM | Approximately 4.59 GiB |
| Swap usage | 0 |

Sequence throughput is not isolated neural-network inference FPS. Publisher-observed callback wait is not equivalent to pure TensorRT inference latency.

See `profiling/` for the recorded measurements.

## Qualitative Analysis

Representative visualizations are in `visualizations/representative/`:

- `01_strong_detections.png`
- `02_missed_pedestrians.png`
- `03_false_positives.png`

The recurring near-origin CAR prediction appeared in 78 of 81 frames. Its position and dimensions were highly consistent in the LiDAR coordinate frame. Cross-frame analysis found a repeated local point pattern despite ego-motion.

This is consistent with a sensor- or vehicle-frame-fixed source, but the physical origin has not been established. It is documented as a systematic false positive, not definitively attributed to ego-vehicle reflections.

See `visualizations/false_positive_investigation/` for the final 3D diagnostic plots.

## Files

- `metrics/`: prediction YAML, frame-status CSV, and evaluation summary JSON.
- `profiling/`: GPU and system profiling measurements.
- `visualizations/`: representative detections and false-positive diagnostics.

## Reproducibility

Experiment scripts are maintained in `experiments/bevfusion_lidar_only/scripts/`.

The original nuScenes dataset and pretrained model weights are not included in this repository. They must be obtained separately and configured locally.

## Limitations

- Evaluation covers 81 frames from two nuScenes mini scenes, not the full validation set.
- The evaluator uses custom greedy one-to-one IoU matching.
- Results are not official nuScenes metrics.
- Coordinate and bounding-box convention assumptions affect evaluation.
- Timestamp association does not prove geometric correctness.
- The recurring near-origin false positive has no confirmed physical explanation.
- Sequence throughput is not isolated TensorRT inference latency.
