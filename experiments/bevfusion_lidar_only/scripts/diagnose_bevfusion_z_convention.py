#!/usr/bin/env python3

"""
Empirically compare BEVFusion prediction z against nuScenes GT
geometric-center z and bottom z.

Matching is class-aware, one-to-one, and based exclusively on BEV IoU.
This is a diagnostic, not an official nuScenes evaluation.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from nuscenes.nuscenes import NuScenes

import evaluate_bevfusion_nuscenes as base
import evaluate_bevfusion_sequence as sequence


def build_ground_truth(nusc, sample_token, point_cloud_range):
    """Build supported GT boxes in the LIDAR_TOP sensor coordinate frame."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    _, nuscenes_boxes, _ = nusc.get_sample_data(lidar_token)

    xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
    ground_truth = []

    for index, box in enumerate(nuscenes_boxes):
        cls = base.NUSCENES_CATEGORY_TO_CLASS.get(box.name)

        if cls is None:
            continue

        x, y, z = [float(v) for v in box.center]
        width, length, height = [float(v) for v in box.wlh]

        # Match the existing evaluator's in-range GT population:
        # annotation center must be inside the model XYZ range.
        if not (
            xmin <= x <= xmax
            and ymin <= y <= ymax
            and zmin <= z <= zmax
        ):
            continue

        if min(length, width, height) <= 0:
            continue

        ground_truth.append({
            "index": index,
            "class": cls,
            "x": x,
            "y": y,
            "z": z,
            "length": length,
            "width": width,
            "height": height,
            "yaw": float(box.orientation.yaw_pitch_roll[0]),
        })

    return ground_truth


def bev_greedy_match(predictions, ground_truth, threshold):
    """Class-aware one-to-one greedy matching using BEV IoU only."""
    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            if pred["class"] != gt["class"]:
                continue

            bev_iou, _ = base.box_iou_metrics(pred, gt)

            if bev_iou >= threshold:
                candidates.append((bev_iou, pi, gi))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_predictions = set()
    used_ground_truth = set()
    matches = []

    for bev_iou, pi, gi in candidates:
        if pi in used_predictions or gi in used_ground_truth:
            continue

        used_predictions.add(pi)
        used_ground_truth.add(gi)
        matches.append((pi, gi, bev_iou))

    return matches


def summarize(values):
    if not values:
        return None

    absolute = [abs(v) for v in values]

    return {
        "count": len(values),
        "mean_signed_error_m": statistics.mean(values),
        "median_signed_error_m": statistics.median(values),
        "mean_absolute_error_m": statistics.mean(absolute),
        "median_absolute_error_m": statistics.median(absolute),
        "rmse_m": math.sqrt(statistics.mean(v * v for v in values)),
        "p05_signed_error_m": sorted(values)[
            max(0, math.ceil(0.05 * len(values)) - 1)
        ],
        "p95_signed_error_m": sorted(values)[
            max(0, math.ceil(0.95 * len(values)) - 1)
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose BEVFusion prediction z-coordinate convention."
    )

    parser.add_argument(
        "--predictions",
        default=(
            "experiment_results/bevfusion_lidar_only_minival/"
            "minival_20261114_191323/sequence_predictions.yaml"
        ),
    )
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--bev-iou-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.50],
    )
    parser.add_argument(
        "--output",
        default=(
            "experiment_results/bevfusion_lidar_only_minival/"
            "z_convention_diagnostic.json"
        ),
    )

    args = parser.parse_args()

    prediction_path = Path(args.predictions).expanduser()
    dataroot = Path(args.dataroot).expanduser()

    if not prediction_path.is_file():
        parser.error(f"Prediction file not found: {prediction_path}")

    print("Loading predictions...")
    records = sequence.load_prediction_documents(
        prediction_path, args.score_threshold
    )
    print(f"Prediction documents: {len(records)}")

    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(
        version=args.version,
        dataroot=str(dataroot),
        verbose=False,
    )

    timestamp_lookup = sequence.build_sample_timestamp_lookup(nusc)

    matched_samples = set()
    matched_pairs_by_threshold = {
        threshold: [] for threshold in args.bev_iou_thresholds
    }

    print("Matching predictions to timestamp-associated GT...")

    for frame_number, record in enumerate(records, start=1):
        if record["frame_id"] != "lidar_top":
            raise ValueError(
                f"Unexpected frame_id in document {frame_number}: "
                f"{record['frame_id']}"
            )

        timestamp_ns = record["timestamp_ns"]

        if timestamp_ns not in timestamp_lookup:
            raise ValueError(
                f"No exact nuScenes LIDAR_TOP timestamp match: {timestamp_ns}"
            )

        sample_token = timestamp_lookup[timestamp_ns]

        if sample_token in matched_samples:
            raise ValueError(
                f"Duplicate matched nuScenes sample: {sample_token}"
            )

        matched_samples.add(sample_token)

        ground_truth = build_ground_truth(
            nusc, sample_token, args.point_cloud_range
        )

        predictions = record["predictions"]

        for threshold in args.bev_iou_thresholds:
            matches = bev_greedy_match(
                predictions, ground_truth, threshold
            )

            for pi, gi, bev_iou in matches:
                pred = predictions[pi]
                gt = ground_truth[gi]

                # Hypothesis A: prediction z represents GT geometric center.
                center_error = pred["z"] - gt["z"]

                # Hypothesis B: prediction z represents GT bottom face.
                gt_bottom_z = gt["z"] - gt["height"] / 2.0
                bottom_error = pred["z"] - gt_bottom_z

                matched_pairs_by_threshold[threshold].append({
                    "sample_token": sample_token,
                    "prediction_index": pred["index"],
                    "ground_truth_index": gt["index"],
                    "class": gt["class"],
                    "bev_iou": bev_iou,
                    "prediction_z": pred["z"],
                    "gt_center_z": gt["z"],
                    "gt_bottom_z": gt_bottom_z,
                    "gt_height": gt["height"],
                    "center_error_m": center_error,
                    "bottom_error_m": bottom_error,
                })

        if frame_number % 10 == 0 or frame_number == len(records):
            print(f"Processed {frame_number}/{len(records)} frames")

    results = {}

    for threshold, pairs in matched_pairs_by_threshold.items():
        center_errors = [p["center_error_m"] for p in pairs]
        bottom_errors = [p["bottom_error_m"] for p in pairs]

        class_results = {}

        for cls in sorted({p["class"] for p in pairs}):
            class_pairs = [p for p in pairs if p["class"] == cls]

            class_results[cls] = {
                "matched_pairs": len(class_pairs),
                "center_hypothesis": summarize(
                    [p["center_error_m"] for p in class_pairs]
                ),
                "bottom_hypothesis": summarize(
                    [p["bottom_error_m"] for p in class_pairs]
                ),
            }

        center_summary = summarize(center_errors)
        bottom_summary = summarize(bottom_errors)

        results[f"BEV_IoU_{threshold:.2f}"] = {
            "matched_pairs": len(pairs),
            "center_hypothesis": center_summary,
            "bottom_hypothesis": bottom_summary,
            "class_breakdown": class_results,
            "pairs": pairs,
        }

    output = {
        "diagnostic": "prediction_z_vs_nuscenes_gt_center_or_bottom",
        "matching": "class-aware one-to-one greedy BEV IoU",
        "dataset_version": args.version,
        "prediction_file": str(prediction_path),
        "prediction_documents": len(records),
        "uniquely_matched_samples": len(matched_samples),
        "point_cloud_range": args.point_cloud_range,
        "score_threshold": args.score_threshold,
        "results": results,
        "interpretation_note": (
            "Smaller absolute z errors support the corresponding hypothesis. "
            "This is empirical evidence, not proof of the model's training "
            "target convention. BEV matching can still produce incorrect "
            "object associations."
        ),
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)

    print("\nZ-CONVENTION DIAGNOSTIC")
    print("=======================")
    print(f"Prediction documents: {len(records)}")
    print(f"Unique matched samples: {len(matched_samples)}")

    for threshold, result in results.items():
        print(f"\n{threshold}")
        print(f"Matched pairs: {result['matched_pairs']}")

        for hypothesis in ("center_hypothesis", "bottom_hypothesis"):
            summary = result[hypothesis]

            if summary is None:
                print(f"{hypothesis}: no matches")
                continue

            print(
                f"{hypothesis}: "
                f"MAE={summary['mean_absolute_error_m']:.4f} m, "
                f"median AE={summary['median_absolute_error_m']:.4f} m, "
                f"RMSE={summary['rmse_m']:.4f} m, "
                f"mean signed={summary['mean_signed_error_m']:.4f} m"
            )

        center_mae = result["center_hypothesis"]["mean_absolute_error_m"]
        bottom_mae = result["bottom_hypothesis"]["mean_absolute_error_m"]

        if center_mae < bottom_mae:
            conclusion = "CENTER hypothesis has lower mean absolute error"
        elif bottom_mae < center_mae:
            conclusion = "BOTTOM hypothesis has lower mean absolute error"
        else:
            conclusion = "Equal mean absolute errors"

        print(f"Comparison: {conclusion}")

        print("Per-class MAE:")
        for cls, class_result in result["class_breakdown"].items():
            center = class_result["center_hypothesis"]["mean_absolute_error_m"]
            bottom = class_result["bottom_hypothesis"]["mean_absolute_error_m"]
            print(
                f"  {cls}: center={center:.4f} m, "
                f"bottom={bottom:.4f} m, "
                f"pairs={class_result['matched_pairs']}"
            )

    print(f"\nJSON report: {output_path}")


if __name__ == "__main__":
    main()
