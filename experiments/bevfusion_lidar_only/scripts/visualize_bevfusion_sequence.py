
#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import evaluate_bevfusion_nuscenes as base
import evaluate_bevfusion_sequence as seq


def evaluate_frames(records, nusc, timestamp_lookup, point_cloud_range,
                    iou_threshold):
    """Associate predictions with nuScenes samples and evaluate each frame."""

    frames = []
    matched_tokens = set()

    for record in records:
        timestamp_ns = record["timestamp_ns"]

        if timestamp_ns not in timestamp_lookup:
            raise ValueError(
                f"No exact nuScenes timestamp match: {timestamp_ns}"
            )

        sample_token = timestamp_lookup[timestamp_ns]

        if sample_token in matched_tokens:
            raise ValueError(
                f"Duplicate sample association: {sample_token}"
            )

        matched_tokens.add(sample_token)

        if record["frame_id"] != "lidar_top":
            raise ValueError(
                f"Unexpected frame_id: {record['frame_id']}"
            )

        (
            sample,
            gt_all,
            gt_supported,
            total_points,
            ranged_points,
        ) = base.load_ground_truth(
            nusc,
            sample_token,
            point_cloud_range,
        )

        predictions = record["predictions"]

        result = base.greedy_match(
            predictions,
            gt_supported,
            "BEV",
            iou_threshold,
            True,
        )

        matched_predictions = {
            match[0] for match in result["matches"]
        }

        matched_ground_truth = {
            match[1] for match in result["matches"]
        }

        false_negative_ground_truth = [
            (index, gt)
            for index, gt in enumerate(gt_supported)
            if index not in matched_ground_truth
        ]

        pedestrian_fn = sum(
            1 for _, gt in false_negative_ground_truth
            if gt["class"] == "PEDESTRIAN"
        )

        scene = nusc.get("scene", sample["scene_token"])

        frames.append({
            "sample_token": sample_token,
            "sample": sample,
            "scene_name": scene["name"],
            "document_index": record["document_index"],
            "timestamp_ns": timestamp_ns,
            "predictions": predictions,
            "ground_truth": gt_supported,
            "matched_predictions": matched_predictions,
            "matched_ground_truth": matched_ground_truth,
            "false_negative_ground_truth": false_negative_ground_truth,
            "pedestrian_fn": pedestrian_fn,
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "raw_lidar_points": total_points,
            "in_range_lidar_points": ranged_points,
        })

    if len(frames) != len(records):
        raise RuntimeError("Not every prediction frame was evaluated.")

    return frames


def select_visualization_frames(frames):
    """Select three distinct frames for qualitative inspection."""

    selected = []
    used_tokens = set()

    def choose(candidates):
        for frame in candidates:
            if frame["sample_token"] not in used_tokens:
                selected.append(frame)
                used_tokens.add(frame["sample_token"])
                return frame
        return None

    # Example 1: strongest frame by number of true positives.
    successful = sorted(
        [f for f in frames if f["tp"] > 0],
        key=lambda f: (f["tp"], f["f1"]),
        reverse=True,
    )

    choose(successful)

    # Example 2: frame with the most missed pedestrians.
    missed_pedestrians = sorted(
        [f for f in frames if f["pedestrian_fn"] > 0],
        key=lambda f: (f["pedestrian_fn"], f["fn"]),
        reverse=True,
    )

    choose(missed_pedestrians)

    # Example 3: frame with the most false-positive predictions.
    false_positives = sorted(
        [f for f in frames if f["fp"] > 0],
        key=lambda f: (f["fp"], -f["tp"]),
        reverse=True,
    )

    choose(false_positives)

    if len(selected) < 3:
        raise RuntimeError(
            "Could not find three distinct frames for visualization."
        )

    return selected


def draw_box(ax, box, color, linestyle, label=None, linewidth=1.3):
    """Draw a yaw-oriented BEV bounding box."""

    corners = base.rectangle_corners(
        box["x"],
        box["y"],
        box["length"],
        box["width"],
        box["yaw"],
    )

    polygon = Polygon(
        corners,
        closed=True,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    ax.add_patch(polygon)


def visualize_frame(nusc, frame, output_path, plot_range):
    """Render and save a top-down LiDAR detection visualization."""

    sample = frame["sample"]
    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_path, _, _ = nusc.get_sample_data(lidar_token)

    pointcloud = LidarPointCloud.from_file(lidar_path)
    points = pointcloud.points

    # Restrict the displayed point cloud to the selected XY region.
    mask = (
        (np.abs(points[0]) <= plot_range)
        & (np.abs(points[1]) <= plot_range)
    )

    visible_points = points[:, mask]

    fig, ax = plt.subplots(figsize=(12, 12))

    scatter = ax.scatter(
        visible_points[0],
        visible_points[1],
        c=visible_points[2],
        s=0.15,
        cmap="viridis",
        alpha=0.65,
        linewidths=0,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="LiDAR point height (m)",
    )

    # Matched predictions: true positives.
    for index, prediction in enumerate(frame["predictions"]):
        if index in frame["matched_predictions"]:
            draw_box(
                ax,
                prediction,
                color="lime",
                linestyle="-",
                label="Matched prediction (TP)" if index ==
                min(frame["matched_predictions"]) else None,
                linewidth=1.5,
            )
        else:
            draw_box(
                ax,
                prediction,
                color="red",
                linestyle="-",
                label="False-positive prediction (FP)" if index ==
                next(
                    (
                        i for i in range(len(frame["predictions"]))
                        if i not in frame["matched_predictions"]
                    ),
                    -1,
                ) else None,
                linewidth=1.5,
            )

    # Ground-truth boxes: matched and missed.
    for index, gt in enumerate(frame["ground_truth"]):
        if index in frame["matched_ground_truth"]:
            draw_box(
                ax,
                gt,
                color="cyan",
                linestyle="--",
                label="Matched ground truth" if index ==
                min(frame["matched_ground_truth"]) else None,
                linewidth=1.1,
            )
        else:
            missed_indices = {
                i for i, _ in frame["false_negative_ground_truth"]
            }

            draw_box(
                ax,
                gt,
                color="orange",
                linestyle="--",
                label="Missed ground truth (FN)" if index ==
                min(missed_indices) else None,
                linewidth=1.4,
            )

    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("LiDAR X (m)")
    ax.set_ylabel("LiDAR Y (m)")
    ax.grid(True, alpha=0.2)

    ax.set_title(
        f"{frame['scene_name']} | Sample {frame['document_index'] + 1}\n"
        f"BEV IoU 0.25 | "
        f"TP={frame['tp']}  FP={frame['fp']}  FN={frame['fn']}\n"
        f"Predictions={len(frame['predictions'])}  "
        f"Ground truth={len(frame['ground_truth'])}  "
        f"Missed pedestrians={frame['pedestrian_fn']}",
        fontsize=11,
    )

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize selected BEVFusion sequence predictions."
    )

    parser.add_argument(
        "--predictions",
        default=(
            "experiment_results/bevfusion_lidar_only_minival/"
            "minival_20261115_095443/sequence_predictions.yaml"
        ),
    )

    parser.add_argument(
        "--dataroot",
        default="data/nuscenes",
    )

    parser.add_argument(
        "--version",
        default="v1.0-mini",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "experiment_results/bevfusion_lidar_only_minival/"
            "minival_20261115_095443/visualizations"
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--plot-range",
        type=float,
        default=50.0,
        help="Displayed LiDAR XY range in meters.",
    )

    args = parser.parse_args()

    prediction_path = Path(args.predictions).expanduser()
    dataroot = Path(args.dataroot).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not prediction_path.is_file():
        parser.error(f"Prediction file not found: {prediction_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prediction documents...")

    records = seq.load_prediction_documents(
        prediction_path,
        args.score_threshold,
    )

    print(f"Loaded prediction frames: {len(records)}")

    print(f"Loading nuScenes {args.version}...")

    nusc = NuScenes(
        version=args.version,
        dataroot=str(dataroot),
        verbose=False,
    )

    timestamp_lookup = seq.build_sample_timestamp_lookup(nusc)

    print("Evaluating individual frames...")

    frames = evaluate_frames(
        records,
        nusc,
        timestamp_lookup,
        [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
        args.iou_threshold,
    )

    print(f"Successfully evaluated frames: {len(frames)}")

    selected = select_visualization_frames(frames)

    filenames = [
        "01_strong_detections.png",
        "02_missed_pedestrians.png",
        "03_false_positives.png",
    ]

    descriptions = [
        "Strong detections",
        "Missed pedestrians",
        "False positives",
    ]

    summary_path = output_dir / "visualization_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.writer(stream)

        writer.writerow([
            "example",
            "scene",
            "sample_number",
            "sample_token",
            "timestamp_ns",
            "tp",
            "fp",
            "fn",
            "missed_pedestrians",
            "precision",
            "recall",
            "f1",
            "raw_lidar_points",
            "in_range_lidar_points",
            "image",
        ])

        for frame, filename, description in zip(
            selected,
            filenames,
            descriptions,
        ):

            output_path = output_dir / filename

            print(f"Creating {filename}...")

            visualize_frame(
                nusc,
                frame,
                output_path,
                args.plot_range,
            )

            writer.writerow([
                description,
                frame["scene_name"],
                frame["document_index"] + 1,
                frame["sample_token"],
                frame["timestamp_ns"],
                frame["tp"],
                frame["fp"],
                frame["fn"],
                frame["pedestrian_fn"],
                frame["precision"],
                frame["recall"],
                frame["f1"],
                frame["raw_lidar_points"],
                frame["in_range_lidar_points"],
                filename,
            ])

    print("\nVISUALIZATION COMPLETE")
    print("----------------------")

    for filename, description, frame in zip(
        filenames,
        descriptions,
        selected,
    ):
        print(
            f"{description}: {filename} | "
            f"TP={frame['tp']} FP={frame['fp']} FN={frame['fn']} | "
            f"Scene={frame['scene_name']}"
        )

    print(f"\nOutput directory: {output_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
