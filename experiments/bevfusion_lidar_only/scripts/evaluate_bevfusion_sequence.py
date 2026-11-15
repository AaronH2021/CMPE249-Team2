#!/usr/bin/env python3
"""
Evaluate a sequence of Autoware BEVFusion DetectedObjects YAML documents
against timestamp-matched nuScenes ground truth.

This is a custom greedy IoU evaluation, not official nuScenes mAP or NDS.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml
from nuscenes.nuscenes import NuScenes

import evaluate_bevfusion_nuscenes as base


def prediction_timestamp_ns(document):
    """Return the ROS header timestamp as an exact integer nanosecond value."""
    stamp = document.get("header", {}).get("stamp", {})
    return int(stamp.get("sec", 0)) * 1_000_000_000 + int(
        stamp.get("nanosec", 0)
    )


def parse_prediction_document(document, score_threshold):
    """Parse one Autoware DetectedObjects YAML document."""
    if not isinstance(document, dict) or not isinstance(
        document.get("objects"), list
    ):
        raise ValueError("Invalid prediction document: missing objects list.")

    predictions = []

    for index, obj in enumerate(document["objects"]):
        score = float(obj.get("existence_probability", 0.0))

        if score < score_threshold:
            continue

        classifications = obj.get("classification", [])
        label = (
            int(classifications[0]["label"])
            if classifications else 0
        )

        cls = base.AUTOWARE_LABEL_TO_CLASS.get(
            label, f"LABEL_{label}"
        )

        pose = obj["kinematics"]["pose_with_covariance"]["pose"]
        position = pose["position"]
        orientation = pose["orientation"]
        dimensions = obj["shape"]["dimensions"]

        box = {
            "index": index,
            "class": cls,
            "label": label,
            "score": score,
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
            "length": float(dimensions["x"]),
            "width": float(dimensions["y"]),
            "height": float(dimensions["z"]),
            "yaw": base.yaw_from_quaternion(orientation),
        }

        if min(
            box["length"], box["width"], box["height"]
        ) <= 0:
            continue

        predictions.append(box)

    return predictions


def load_prediction_documents(path, score_threshold):
    """Load every YAML document and reject duplicate timestamps."""
    with open(path, "r", encoding="utf-8") as stream:
        documents = [
            doc for doc in yaml.safe_load_all(stream)
            if isinstance(doc, dict)
        ]

    if not documents:
        raise ValueError("No valid prediction documents found.")

    records = []
    seen_timestamps = set()

    for document_index, document in enumerate(documents):
        if "objects" not in document:
            raise ValueError(
                f"Document {document_index} does not contain objects."
            )

        timestamp_ns = prediction_timestamp_ns(document)

        if timestamp_ns in seen_timestamps:
            raise ValueError(
                f"Duplicate prediction timestamp: {timestamp_ns}"
            )

        seen_timestamps.add(timestamp_ns)

        records.append({
            "document_index": document_index,
            "timestamp_ns": timestamp_ns,
            "frame_id": document.get(
                "header", {}
            ).get("frame_id", "unknown"),
            "predictions": parse_prediction_document(
                document, score_threshold
            ),
        })

    return records


def build_sample_timestamp_lookup(nusc):
    """Map each nuScenes keyframe LIDAR_TOP timestamp to its sample."""
    lookup = {}

    for sample in nusc.sample:
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)

        timestamp_ns = int(lidar_data["timestamp"]) * 1000

        if timestamp_ns in lookup:
            raise ValueError(
                f"Duplicate nuScenes LiDAR timestamp: {timestamp_ns}"
            )

        lookup[timestamp_ns] = sample["token"]

    return lookup


def add_counts(total, result):
    total["tp"] += result["tp"]
    total["fp"] += result["fp"]
    total["fn"] += result["fn"]


def calculate_metrics(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all timestamp-matched BEVFusion predictions."
    )

    parser.add_argument(
        "--predictions",
        default=(
            "experiment_results/bevfusion_lidar_only_minival/"
            "minival_20261114_191323/sequence_predictions.yaml"
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
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.50],
    )
    parser.add_argument(
        "--output",
        default="experiment_results/bevfusion_lidar_only_minival/"
                "sequence_evaluation_summary.json",
    )

    args = parser.parse_args()

    prediction_path = Path(args.predictions).expanduser()
    dataroot = Path(args.dataroot).expanduser()

    if not prediction_path.is_file():
        parser.error(f"Prediction file not found: {prediction_path}")

    print("Loading prediction documents...")
    records = load_prediction_documents(
        prediction_path, args.score_threshold
    )

    print(f"Prediction documents: {len(records)}")

    print(f"Loading nuScenes {args.version} from {dataroot}...")
    nusc = NuScenes(
        version=args.version,
        dataroot=str(dataroot),
        verbose=False,
    )

    timestamp_lookup = build_sample_timestamp_lookup(nusc)

    matched_tokens = set()
    frame_results = []
    frame_id_counts = Counter()

    # Each metric has its own aggregate counts across frames.
    aggregate = {}

    for threshold in args.iou_thresholds:
        for metric in ("BEV", "3D"):
            for class_aware in (False, True):
                for population in ("all_supported", "in_range"):
                    key = (
                        f"{metric}_IoU_{threshold:.2f}_"
                        f"{'class_aware' if class_aware else 'class_agnostic'}_"
                        f"{population}"
                    )
                    aggregate[key] = {
                        "tp": 0, "fp": 0, "fn": 0
                    }

    classes = sorted(set(base.AUTOWARE_LABEL_TO_CLASS.values()))
    per_class = {}

    for threshold in args.iou_thresholds:
        for metric in ("BEV", "3D"):
            for cls in classes:
                per_class[
                    f"{metric}_IoU_{threshold:.2f}_{cls}"
                ] = {"tp": 0, "fp": 0, "fn": 0}

    print("Associating predictions with nuScenes samples...")

    for record in records:
        timestamp_ns = record["timestamp_ns"]

        if timestamp_ns not in timestamp_lookup:
            raise ValueError(
                "Prediction timestamp has no exact nuScenes LIDAR_TOP "
                f"keyframe match: {timestamp_ns}"
            )

        sample_token = timestamp_lookup[timestamp_ns]

        if sample_token in matched_tokens:
            raise ValueError(
                f"Multiple prediction documents matched sample {sample_token}"
            )

        matched_tokens.add(sample_token)

        if record["frame_id"] != "lidar_top":
            raise ValueError(
                f"Unexpected prediction frame_id: {record['frame_id']}"
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
            args.point_cloud_range,
        )

        in_range_gt = [
            gt for gt in gt_supported
            if gt["center_in_model_range"]
        ]

        predictions = record["predictions"]

        frame_id_counts[sample["scene_token"]] += 1

        frame_results.append({
            "sample_token": sample_token,
            "timestamp_ns": timestamp_ns,
            "prediction_count": len(predictions),
            "supported_gt_count": len(gt_supported),
            "in_range_gt_count": len(in_range_gt),
            "raw_lidar_points": total_points,
            "in_range_lidar_points": ranged_points,
        })

        populations = {
            "all_supported": gt_supported,
            "in_range": in_range_gt,
        }

        for threshold in args.iou_thresholds:
            for metric in ("BEV", "3D"):
                for class_aware in (False, True):
                    for population_name, ground_truth in populations.items():
                        result = base.greedy_match(
                            predictions,
                            ground_truth,
                            metric,
                            threshold,
                            class_aware,
                        )

                        key = (
                            f"{metric}_IoU_{threshold:.2f}_"
                            f"{'class_aware' if class_aware else 'class_agnostic'}_"
                            f"{population_name}"
                        )
                        add_counts(aggregate[key], result)

                # Per-class class-aware evaluation.
                for cls in classes:
                    class_predictions = [
                        p for p in predictions if p["class"] == cls
                    ]
                    class_ground_truth = [
                        g for g in gt_supported if g["class"] == cls
                    ]

                    result = base.greedy_match(
                        class_predictions,
                        class_ground_truth,
                        metric,
                        threshold,
                        False,
                    )

                    key = f"{metric}_IoU_{threshold:.2f}_{cls}"
                    add_counts(per_class[key], result)

    if len(records) != len(matched_tokens):
        raise RuntimeError("Not all prediction documents were uniquely matched.")

    # Convert aggregate counts into final metrics.
    aggregate_metrics = {
        key: calculate_metrics(counts)
        for key, counts in aggregate.items()
    }

    per_class_metrics = {
        key: calculate_metrics(counts)
        for key, counts in per_class.items()
    }

    summary = {
        "evaluation_type": "custom_sequence_greedy_iou",
        "official_nuscenes_map_or_nds": False,
        "dataset_version": args.version,
        "dataroot": str(dataroot),
        "prediction_file": str(prediction_path),
        "prediction_documents": len(records),
        "matched_unique_samples": len(matched_tokens),
        "score_threshold": args.score_threshold,
        "point_cloud_range": args.point_cloud_range,
        "iou_thresholds": args.iou_thresholds,
        "frame_id_counts": dict(frame_id_counts),
        "aggregate_metrics": aggregate_metrics,
        "per_class_metrics": per_class_metrics,
        "frames": frame_results,
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("\nSEQUENCE VALIDATION")
    print("-------------------")
    print(f"Prediction documents: {len(records)}")
    print(f"Unique matched samples: {len(matched_tokens)}")
    print(f"Scene sample counts: {dict(frame_id_counts)}")
    print(f"Output: {output_path}")

    print("\nAGGREGATE METRICS")
    print("-----------------")

    for key, result in aggregate_metrics.items():
        print(
            f"{key}: "
            f"TP={result['tp']} "
            f"FP={result['fp']} "
            f"FN={result['fn']} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    print("\nPER-CLASS METRICS (ALL SUPPORTED GT)")
    print("------------------------------------")

    for key, result in per_class_metrics.items():
        print(
            f"{key}: "
            f"TP={result['tp']} "
            f"FP={result['fp']} "
            f"FN={result['fn']} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    print(
        "\nNote: metrics use frame-local greedy one-to-one IoU matching. "
        "They are not official nuScenes mAP or NDS."
    )


if __name__ == "__main__":
    main()
