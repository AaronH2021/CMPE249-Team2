# Datasets

This directory documents the datasets used in the BEVFusion deployment and evaluation experiments.

## nuScenes Dataset

The LiDAR-only BEVFusion evaluation uses the **nuScenes v1.0-mini** dataset.

- Dataset: [nuScenes](https://www.nuscenes.org/nuscenes)
- Version: `v1.0-mini`
- Usage: LiDAR point cloud input and 3D object detection evaluation.
- Dataset files are not included in this repository.

### Download and Setup

1. Register for access and download the nuScenes dataset from the official website.
2. Extract the dataset into a local directory.
3. Configure the dataset root path in the experiment scripts.

Expected directory structure:

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

The dataset should remain outside the Git repository. Refer to the LiDAR-only experiment documentation for detailed setup and execution instructions.

## Notes

- Dataset files are excluded from Git tracking because of their size and licensing considerations.
- Users must obtain the dataset independently.
- The evaluation results in this repository were generated using nuScenes v1.0-mini.
