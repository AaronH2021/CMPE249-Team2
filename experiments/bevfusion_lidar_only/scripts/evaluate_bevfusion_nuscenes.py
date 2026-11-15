
#!/usr/bin/env python3
"""
Evaluate one Autoware BEVFusion DetectedObjects message against nuScenes GT.

Expected prediction input:
  ros2 topic echo /perception/detection/objects --once > ~/bevfusion_predictions.txt

Run:
  source ~/nuscenes_venv/bin/activate
  python ~/nuscenes_bevfusion/evaluate_bevfusion_nuscenes.py

Dependencies:
  pip install nuscenes-devkit PyYAML

Assumptions:
- Predictions are in the nuScenes LIDAR_TOP sensor coordinate frame.
- DetectedObject.shape.dimensions.x/y/z are length/width/height.
- Prediction orientation is a yaw-only quaternion about +Z.
- nuScenes Box.wlh is width/length/height; it is reordered to length/width/height.
- nuScenes Box coordinates returned by get_sample_data() are in the LiDAR frame.
- Supported GT classes are limited to the five BEVFusion model classes.
- Greedy one-to-one matching is used (descending IoU), not official nuScenes mAP.
"""

import argparse
import math
import os
import sys

import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box


# Autoware ObjectClassification constants commonly used in DetectedObject:
# UNKNOWN=0, CAR=1, TRUCK=2, BUS=3, TRAILER=4, MOTORCYCLE=5,
# BICYCLE=6, PEDESTRIAN=7.
AUTOWARE_LABEL_TO_CLASS = {
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "TRAILER",
    6: "BICYCLE",
    7: "PEDESTRIAN",
}

NUSCENES_CATEGORY_TO_CLASS = {
    "vehicle.car": "CAR",
    "vehicle.truck": "TRUCK",
    "vehicle.bus": "BUS",
    "vehicle.bus.rigid": "BUS",
    "vehicle.bus.bendy": "BUS",
    "vehicle.bicycle": "BICYCLE",
    "vehicle.trailer": "TRAILER",
    "human.pedestrian.adult": "PEDESTRIAN",
    "human.pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
}


def yaw_from_quaternion(q):
    # Yaw for a normalized planar quaternion (x=y=0).
    return 2.0 * math.atan2(float(q["z"]), float(q["w"]))


def rectangle_corners(cx, cy, length, width, yaw):
    """Return CCW corners for a yaw-oriented rectangle in the XY plane."""
    local = [
        (-length / 2.0, -width / 2.0),
        ( length / 2.0, -width / 2.0),
        ( length / 2.0,  width / 2.0),
        (-length / 2.0,  width / 2.0),
    ]
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (cx + x * c - y * s, cy + x * s + y * c)
        for x, y in local
    ]


def signed_polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def line_intersection(p1, p2, q1, q2):
    """Intersection of infinite lines p1-p2 and q1-q2."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    return (
        (det1 * (x3 - x4) - (x1 - x2) * det2) / den,
        (det1 * (y3 - y4) - (y1 - y2) * det2) / den,
    )


def polygon_intersection(subject, clip):
    """Sutherland-Hodgman convex polygon clipping; both polygons CCW."""
    output = list(subject)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break

        def inside(p):
            return (
                (b[0] - a[0]) * (p[1] - a[1])
                - (b[1] - a[1]) * (p[0] - a[0])
            ) >= -1e-9

        prev = input_list[-1]
        prev_inside = inside(prev)
        for curr in input_list:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(line_intersection(prev, curr, a, b))
                output.append(curr)
            elif prev_inside:
                output.append(line_intersection(prev, curr, a, b))
            prev, prev_inside = curr, curr_inside
    return output


def box_iou_metrics(a, b):
    """Boxes use center x/y/z, length, width, height, yaw."""
    poly_a = rectangle_corners(
        a["x"], a["y"], a["length"], a["width"], a["yaw"]
    )
    poly_b = rectangle_corners(
        b["x"], b["y"], b["length"], b["width"], b["yaw"]
    )
    area_a = a["length"] * a["width"]
    area_b = b["length"] * b["width"]
    inter_poly = polygon_intersection(poly_a, poly_b)
    inter_area = abs(signed_polygon_area(inter_poly))
    bev_union = area_a + area_b - inter_area
    bev_iou = inter_area / bev_union if bev_union > 0 else 0.0

    zmin_a = a["z"] - a["height"] / 2
    zmax_a = a["z"] + a["height"] / 2
    zmin_b = b["z"] - b["height"] / 2
    zmax_b = b["z"] + b["height"] / 2

    inter_h = max(0.0, min(zmax_a, zmax_b) - max(zmin_a, zmin_b))
    inter_vol = inter_area * inter_h
    vol_a = area_a * a["height"]
    vol_b = area_b * b["height"]
    union_vol = vol_a + vol_b - inter_vol
    iou3d = inter_vol / union_vol if union_vol > 0 else 0.0

    return bev_iou, iou3d


def angle_error_deg(a, b):
    # Box heading is equivalent modulo pi (180 degrees).
    delta = abs((a - b + math.pi / 2) % math.pi - math.pi / 2)
    return math.degrees(delta)


def load_predictions(path, score_threshold):
    with open(path, "r", encoding="utf-8") as f:
        # Accept multiple YAML documents; ros2 topic echo may append "---".
        documents = list(yaml.safe_load_all(f))

    data = next(
        (doc for doc in documents if isinstance(doc, dict)),
        None
    )

    if not isinstance(data, dict) or "objects" not in data:
        raise ValueError(
            "Input YAML does not contain a top-level 'objects' list."
        )

    header = data.get("header", {})
    stamp = header.get("stamp", {})
    frame = header.get("frame_id", "unknown")
    timestamp = (
        int(stamp.get("sec", 0))
        + int(stamp.get("nanosec", 0)) * 1e-9
    )

    preds = []

    for index, obj in enumerate(data["objects"]):
        existence = float(obj.get("existence_probability", 0.0))

        if existence < score_threshold:
            continue

        classifications = obj.get("classification", [])
        label = (
            int(classifications[0]["label"])
            if classifications else 0
        )
        cls = AUTOWARE_LABEL_TO_CLASS.get(
            label, f"LABEL_{label}"
        )

        pose = obj["kinematics"]["pose_with_covariance"]["pose"]
        pos = pose["position"]
        quat = pose["orientation"]
        dims = obj["shape"]["dimensions"]

        box = {
            "index": index,
            "class": cls,
            "label": label,
            "score": existence,
            "x": float(pos["x"]),
            "y": float(pos["y"]),
            "z": float(pos["z"]),
            "length": float(dims["x"]),
            "width": float(dims["y"]),
            "height": float(dims["z"]),
            "yaw": yaw_from_quaternion(quat),
        }

        if min(
            box["length"],
            box["width"],
            box["height"]
        ) <= 0:
            print(
                f"WARNING: skipping prediction {index} "
                "with non-positive dimensions"
            )
            continue

        preds.append(box)

    return preds, frame, timestamp


def load_ground_truth(nusc, sample_token, point_cloud_range):
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_path, nuscenes_boxes, _ = nusc.get_sample_data(lidar_token)

    # Load raw nuScenes LIDAR_TOP points.
    # The point cloud and boxes returned by get_sample_data()
    # are both expressed in the LiDAR sensor coordinate frame.
    pointcloud = LidarPointCloud.from_file(lidar_path)
    points = pointcloud.points[:3, :]

    xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range

    # Apply the configured BEVFusion XYZ range to the raw points.
    in_model_range = (
        (points[0] >= xmin) & (points[0] <= xmax)
        & (points[1] >= ymin) & (points[1] <= ymax)
        & (points[2] >= zmin) & (points[2] <= zmax)
    )

    points_in_range = points[:, in_model_range]

    gt_all = []
    gt_supported = []

    for i, box in enumerate(nuscenes_boxes):
        cls = NUSCENES_CATEGORY_TO_CLASS.get(box.name)
        yaw = float(box.orientation.yaw_pitch_roll[0])

        # nuScenes Box.wlh is width, length, height.
        w, l, h = [float(v) for v in box.wlh]

        # Determine whether the annotation center is inside
        # the model's XYZ point-cloud range.
        center_in_range = (
            xmin <= float(box.center[0]) <= xmax
            and ymin <= float(box.center[1]) <= ymax
            and zmin <= float(box.center[2]) <= zmax
        )

        # Count raw LiDAR points inside the oriented GT box.
        raw_point_mask = points_in_box(box, points)

        # Count only points that are also inside the model range.
        ranged_point_mask = points_in_box(box, points_in_range)

        item = {
            "index": i,
            "name": box.name,
            "class": cls if cls else box.name,
            "supported": cls is not None,
            "center_in_model_range": center_in_range,
            "lidar_points_raw": int(raw_point_mask.sum()),
            "lidar_points_in_model_range": int(
                ranged_point_mask.sum()
            ),
            "x": float(box.center[0]),
            "y": float(box.center[1]),
            "z": float(box.center[2]),
            "length": l,
            "width": w,
            "height": h,
            "yaw": yaw,
        }

        gt_all.append(item)

        if cls is not None:
            gt_supported.append(item)

    return (
        sample,
        gt_all,
        gt_supported,
        len(points.T),
        int(in_model_range.sum())
    )


def greedy_match(preds, gts, metric, threshold, class_aware):
    """Greedy one-to-one matching sorted by descending IoU."""
    candidates = []

    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if class_aware and pred["class"] != gt["class"]:
                continue

            bev_iou, iou3d = box_iou_metrics(pred, gt)
            value = bev_iou if metric == "BEV" else iou3d

            if value >= threshold:
                candidates.append(
                    (value, pi, gi, bev_iou, iou3d)
                )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    used_p = set()
    used_g = set()
    matches = []

    for value, pi, gi, bev_iou, iou3d in candidates:
        if pi in used_p or gi in used_g:
            continue

        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, bev_iou, iou3d))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_match_table(title, result, preds, gts):
    print(f"\n{title}")
    print("-" * len(title))

    print(
        f"TP={result['tp']}  FP={result['fp']}  "
        f"FN={result['fn']}  "
        f"Precision={result['precision']:.3f}  "
        f"Recall={result['recall']:.3f}  "
        f"F1={result['f1']:.3f}"
    )

    if not result["matches"]:
        print("No matches.")
        return

    print(
        "Pred | Pred class | GT | GT category              "
        "| BEV IoU | 3D IoU | Center err"
    )

    for pi, gi, bev_iou, iou3d in result["matches"]:
        p = preds[pi]
        g = gts[gi]

        dist = math.sqrt(
            (p["x"] - g["x"])**2
            + (p["y"] - g["y"])**2
            + (p["z"] - g["z"])**2
        )

        print(
            f"{p['index']:>4} | {p['class']:<10} | "
            f"{g['index']:>2} | {g['name']:<24} | "
            f"{bev_iou:>7.3f} | {iou3d:>6.3f} | "
            f"{dist:>9.3f} m"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Autoware BEVFusion detections "
            "against nuScenes GT."
        )
    )

    parser.add_argument(
        "--predictions",
        default=os.path.expanduser("~/bevfusion_predictions.txt"),
        help=(
            "ROS DetectedObjects YAML file "
            "(default: ~/bevfusion_predictions.txt)"
        ),
    )

    parser.add_argument(
        "--dataroot",
        default=os.path.expanduser("~/datasets/nuscenes"),
        help="nuScenes dataset directory",
    )

    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version (default: v1.0-mini)",
    )

    parser.add_argument(
        "--sample-token",
        default="ca9a282c9e77460f8360f564131a8af5",
        help="nuScenes sample token",
    )

    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[
            -122.4, -122.4, -3.0,
            122.4, 122.4, 5.0
        ],
        metavar=(
            "XMIN", "YMIN", "ZMIN",
            "XMAX", "YMAX", "ZMAX"
        ),
        help=(
            "BEVFusion XYZ point-cloud range "
            "(default: -122.4 -122.4 -3.0 122.4 122.4 5.0)"
        ),
    )

    parser.add_argument(
        "--visibility-thresholds",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help=(
            "LiDAR returns required for visibility-count summaries "
            "(default: 1 5 10 20)"
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum existence_probability to retain a prediction "
            "(default: 0.0)"
        ),
    )

    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.50],
        help="IoU thresholds for reporting (default: 0.25 0.50)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.predictions):
        parser.error(
            f"Prediction file not found: {args.predictions}"
        )

    preds, frame, pred_timestamp = load_predictions(
        args.predictions,
        args.score_threshold
    )

    print(
        f"Loading nuScenes {args.version} "
        f"from {args.dataroot} ..."
    )

    nusc = NuScenes(
        version=args.version,
        dataroot=args.dataroot,
        verbose=False
    )

    (
        sample,
        gt_all,
        gt_supported,
        total_lidar_points,
        ranged_lidar_points
    ) = load_ground_truth(
        nusc,
        args.sample_token,
        args.point_cloud_range
    )

    lidar_sd = nusc.get(
        "sample_data",
        sample["data"]["LIDAR_TOP"]
    )

    gt_timestamp = float(lidar_sd["timestamp"]) * 1e-6

    print("\nDATASET / INPUT CHECK")
    print("---------------------")
    print(f"Sample token:             {sample['token']}")
    print(f"Prediction frame_id:      {frame}")
    print(f"Prediction stamp:         {pred_timestamp:.9f}")
    print(f"nuScenes LiDAR timestamp: {gt_timestamp:.9f}")
    print(
        f"Timestamp difference:     "
        f"{abs(pred_timestamp - gt_timestamp) * 1000:.6f} ms"
    )
    print(f"Predictions retained:     {len(preds)}")
    print(f"All nuScenes GT boxes:    {len(gt_all)}")
    print(f"Supported-class GT boxes: {len(gt_supported)}")
    print(f"Raw LIDAR_TOP points:     {total_lidar_points}")
    print(f"Points inside model range: {ranged_lidar_points}")
    print(f"Model XYZ range:          {args.point_cloud_range}")

    if frame != "lidar_top":
        print(
            "WARNING: prediction frame is not 'lidar_top'; "
            "verify coordinate-frame alignment."
        )

    if abs(pred_timestamp - gt_timestamp) > 1e-3:
        print(
            "WARNING: prediction and GT timestamps "
            "differ by more than 1 ms."
        )

    # ---------------------------------------------------------
    # Ground-truth range and LiDAR visibility diagnostics
    # ---------------------------------------------------------

    print("\nGROUND-TRUTH RANGE / LIDAR VISIBILITY")
    print("-------------------------------------")

    supported_in_range = [
        g for g in gt_supported
        if g["center_in_model_range"]
    ]

    print(
        f"Supported GT with center inside model range: "
        f"{len(supported_in_range)}/{len(gt_supported)}"
    )

    for threshold in args.visibility_thresholds:
        count = sum(
            g["lidar_points_in_model_range"] >= threshold
            for g in supported_in_range
        )

        print(
            f"Supported in-range GT with >= {threshold:>3} "
            f"LiDAR points: {count}/{len(supported_in_range)}"
        )

    print("\nPer-object supported GT:")
    print(
        "GT | Category                  | Center in range "
        "| Raw pts | In-range pts"
    )

    for g in gt_supported:
        print(
            f"{g['index']:>2} | {g['name']:<25} | "
            f"{str(g['center_in_model_range']):<15} | "
            f"{g['lidar_points_raw']:>7} | "
            f"{g['lidar_points_in_model_range']:>12}"
        )

    print(
        "\nNote: point counts are raw nuScenes returns inside each "
        "oriented 3D annotation box; in-range counts additionally "
        "apply the configured BEVFusion XYZ range. They are a "
        "visibility proxy, not a guarantee that every return "
        "survives the model's full preprocessing."
    )

    # ---------------------------------------------------------
    # Prediction summary
    # ---------------------------------------------------------

    print("\nPREDICTION SUMMARY")
    print("------------------")

    for p in preds:
        print(
            f"Pred {p['index']}: {p['class']:<10} "
            f"score={p['score']:.3f} "
            f"center=({p['x']:.2f},{p['y']:.2f},{p['z']:.2f}) "
            f"size=({p['length']:.2f},{p['width']:.2f},"
            f"{p['height']:.2f}) "
            f"yaw={math.degrees(p['yaw']):.1f} deg"
        )

    # ---------------------------------------------------------
    # Nearest ground-truth box
    # ---------------------------------------------------------

    print("\nNEAREST GROUND-TRUTH BOX (ALL 69 ANNOTATIONS)")
    print("---------------------------------------------")

    for p in preds:
        ranked = []

        for g in gt_all:
            d = math.sqrt(
                (p["x"] - g["x"])**2
                + (p["y"] - g["y"])**2
                + (p["z"] - g["z"])**2
            )

            bev_iou, iou3d = box_iou_metrics(p, g)
            ranked.append((d, g, bev_iou, iou3d))

        d, g, bev_iou, iou3d = min(
            ranked,
            key=lambda x: x[0]
        )

        print(
            f"Pred {p['index']} ({p['class']}) -> "
            f"GT {g['index']} ({g['name']}): "
            f"center={d:.3f} m, "
            f"BEV IoU={bev_iou:.3f}, "
            f"3D IoU={iou3d:.3f}, "
            f"class_supported={g['supported']}"
        )

    # ---------------------------------------------------------
    # Detection metrics
    # ---------------------------------------------------------

    print("\nMETRICS AGAINST SIX SUPPORTED PUBLISHED CLASSES")
    print("------------------------------------------------")
    print(
        "Classes: CAR, TRUCK, BUS, TRAILER, BICYCLE, PEDESTRIAN."
    )
    print(
        "These evaluate published Autoware detections after "
        "class remapping."
    )
    print(
        "GT categories outside these six classes are excluded."
    )
    print(
        "Class-agnostic scores measure localization; "
        "class-aware scores also require class agreement."
    )

    evaluation_populations = [
        ("ALL SUPPORTED GT", gt_supported),
        ("IN-RANGE SUPPORTED GT", supported_in_range),
    ]

    for population_name, evaluation_gt in evaluation_populations:
        print(f"\n\n===== {population_name} =====")
        print(f"Ground-truth objects: {len(evaluation_gt)}")

        for threshold in args.iou_thresholds:
            for metric in ("BEV", "3D"):
                for class_aware in (False, True):
                    result = greedy_match(
                        preds,
                        evaluation_gt,
                        metric,
                        threshold,
                        class_aware
                    )

                    mode = (
                        "class-aware"
                        if class_aware
                        else "class-agnostic"
                    )

                    title = (
                        f"{population_name}: "
                        f"{metric} IoU >= {threshold:.2f} ({mode})"
                    )

                    print_match_table(
                        title,
                        result,
                        preds,
                        evaluation_gt
                    )
    # ---------------------------------------------------------
    # Per-match error details
    # ---------------------------------------------------------

    print(
        "\nPER-MATCH ERROR DETAILS "
        "(BEST CLASS-AWARE BEV MATCHES AT 0.25)"
    )
    print(
        "---------------------------------------------------------------"
    )

    result = greedy_match(
        preds,
        gt_supported,
        "BEV",
        0.25,
        True
    )

    if not result["matches"]:
        print("No class-aware matches at BEV IoU >= 0.25.")

    for pi, gi, bev_iou, iou3d in result["matches"]:
        p = preds[pi]
        g = gt_supported[gi]

        center_err = math.sqrt(
            (p["x"] - g["x"])**2
            + (p["y"] - g["y"])**2
            + (p["z"] - g["z"])**2
        )

        dims_err = [
            p["length"] - g["length"],
            p["width"] - g["width"],
            p["height"] - g["height"],
        ]

        yaw_err = angle_error_deg(
            p["yaw"],
            g["yaw"]
        )

        print(
            f"Pred {p['index']} vs GT {g['index']} ({g['name']}): "
            f"center_err={center_err:.3f} m; "
            f"dimension_delta L/W/H=("
            f"{dims_err[0]:+.3f},"
            f"{dims_err[1]:+.3f},"
            f"{dims_err[2]:+.3f}) m; "
            f"yaw_error={yaw_err:.2f} deg; "
            f"BEV IoU={bev_iou:.3f}; "
            f"3D IoU={iou3d:.3f}"
        )

    # ---------------------------------------------------------
    # Interpretation notes
    # ---------------------------------------------------------

    print("\nINTERPRETATION NOTES")
    print("--------------------")
    print(
        "- These are single-frame, greedy-matching metrics, "
        "not official nuScenes mAP."
    )
    print(
        "- The primary custom evaluation uses six published "
        "classes: CAR/TRUCK/BUS/TRAILER/BICYCLE/PEDESTRIAN."
    )
    print(
        "- Published detections are evaluated after Autoware "
        "class remapping."
    )
    print(
        "- Both all-supported-GT and in-range-supported-GT "
        "populations are evaluated."
    )
    print(
        "- In-range GT membership is based on annotation center; "
        "a box may partially overlap the range even if its "
        "center is outside."
    )
    print(
        "- Class-agnostic evaluation measures localization; "
        "class-aware evaluation also requires class agreement."
    )
    print(
        "- Nearest-GT output includes unsupported categories "
        "(e.g. barriers) to expose likely confusions."
    )
    print(
        "- Verify the model's remapping configuration before "
        "interpreting CAR/TRUCK/TRAILER confusions."
    )
    print(
        "- LiDAR point counts are a raw-return visibility proxy; "
        "they do not reproduce every model preprocessing filter."
    )
    print(
        "- A single frame is insufficient for a reliable "
        "generalization or accuracy claim."
    )
    print(
        "- These custom metrics do not calculate confidence-ranked "
        "average precision or official nuScenes NDS."
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
