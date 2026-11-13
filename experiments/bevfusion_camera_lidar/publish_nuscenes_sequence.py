#!/usr/bin/env python3
"""Replay a nuScenes Mini scene through Autoware BEVFusion and record results.

For each of the first NUM_SAMPLES keyframes of scene 0 this node:

  1. publishes the six camera images + CameraInfo for that sample,
  2. waits until they have actually gone out and BEVFusion has cached them,
  3. publishes the matching LiDAR sweep (this triggers BEVFusion inference),
  4. waits for a DetectedObjects message whose header stamp equals the LiDAR
     stamp, and writes it to YAML plus a per-frame status row to CSV.

Frames are processed strictly one at a time so every detection can be tied
to exactly one input sample. Each run writes to a new, timestamped folder
under ~/nuscenes_bevfusion/sequence_runs/.
"""

import csv
import time
import numpy as np
import rclpy
import yaml

from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from autoware_perception_msgs.msg import DetectedObjects

from rosidl_runtime_py.convert import message_to_ordereddict

from nuscenes.nuscenes import NuScenes


# (nuScenes channel name, Autoware camera id, TF optical frame).
# The list order defines BEVFusion's camera_id 0-5 and must match both the
# BEVFusion launch configuration and the frames in publish_camera_tf.py.
CAMERAS = [
    ("CAM_FRONT", "camera0", "cam_front_optical_frame"),
    ("CAM_FRONT_RIGHT", "camera1", "cam_front_right_optical_frame"),
    ("CAM_FRONT_LEFT", "camera2", "cam_front_left_optical_frame"),
    ("CAM_BACK", "camera3", "cam_back_optical_frame"),
    ("CAM_BACK_LEFT", "camera4", "cam_back_left_optical_frame"),
    ("CAM_BACK_RIGHT", "camera5", "cam_back_right_optical_frame"),
]

NUM_SAMPLES = 10

# Maximum time to wait for all six camera images and CameraInfo messages.
CAMERA_READINESS_TIMEOUT_SECONDS = 5.0

# Allow callbacks to settle after all camera messages are received.
CAMERA_SETTLE_SECONDS = 1.0

# Maximum time to wait for a detection with the matching LiDAR timestamp.
DETECTION_TIMEOUT_SECONDS = 10.0

# Maximum time to wait for BEVFusion to subscribe to the LiDAR topic.
LIDAR_SUBSCRIBER_TIMEOUT_SECONDS = 10.0

# After the last sample, keep listening this long for late detections.
FINAL_DETECTION_GRACE_SECONDS = 10.0

# Bounds for the cache of detections that arrive before we expect them.
DETECTION_CACHE_MAX_ENTRIES = 10
DETECTION_CACHE_TTL_SECONDS = 30.0


class NuScenesSequencePublisher(Node):

    def __init__(self):
        super().__init__("nuscenes_sequence_publisher")

        self.dataroot = Path.home() / "datasets/nuscenes"

        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=str(self.dataroot),
            verbose=False,
        )

        # Create a unique output directory. Never overwrite previous results.
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path.home()
            / "nuscenes_bevfusion"
            / "sequence_runs"
            / f"scene0061_{run_timestamp}"
        )
        # exist_ok=False and open(..., "x") below both raise instead of
        # silently overwriting results from an earlier run.
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.predictions_path = (
            self.output_dir / "sequence_predictions.yaml"
        )
        self.status_path = (
            self.output_dir / "sequence_frame_status.csv"
        )

        self.predictions_file = open(
            self.predictions_path, "x", encoding="utf-8"
        )

        self.status_file = open(
            self.status_path, "x", newline="", encoding="utf-8"
        )

        self.status_writer = csv.DictWriter(
            self.status_file,
            fieldnames=[
                "sample_index",
                "scene",
                "sample_token",
                "lidar_timestamp_us",
                "lidar_timestamp_ns",
                "status",
                "detection_count",
                "wait_seconds",
                "camera_deltas_ms",
                "error",
            ],
        )
        self.status_writer.writeheader()
        self.status_file.flush()

        self.lidar_publisher = self.create_publisher(
            PointCloud2,
            "/sensing/lidar/concatenated/pointcloud",
            10,
        )

        self.image_publishers = []
        self.info_publishers = []

        self.camera_image_received = [False] * len(CAMERAS)
        self.camera_info_received = [False] * len(CAMERAS)
        self.camera_subscriptions = []

        for index, (_, camera_id, _) in enumerate(CAMERAS):
            image_topic = f"/sensing/camera/{camera_id}/image_rect_color"
            info_topic = f"/sensing/camera/{camera_id}/camera_info"

            self.image_publishers.append(
                self.create_publisher(
                    Image,
                    image_topic,
                    qos_profile_sensor_data,
                )
            )

            self.info_publishers.append(
                self.create_publisher(
                    CameraInfo,
                    info_topic,
                    qos_profile_sensor_data,
                )
            )

            # Subscribe to our *own* camera topics as a delivery check:
            # hearing a message back confirms it actually went out over
            # DDS before we move on to publishing the LiDAR cloud.
            # (i=index binds the loop value; a bare lambda would capture
            # only the final index.)
            self.camera_subscriptions.append(
                self.create_subscription(
                    Image,
                    image_topic,
                    lambda msg, i=index: self.camera_image_callback(msg, i),
                    qos_profile_sensor_data,
                )
            )

            self.camera_subscriptions.append(
                self.create_subscription(
                    CameraInfo,
                    info_topic,
                    lambda msg, i=index: self.camera_info_callback(msg, i),
                    qos_profile_sensor_data,
                )
            )

        self.detection_subscriber = self.create_subscription(
            DetectedObjects,
            "/perception/detection/objects",
            self.detection_callback,
            10,
        )

        # Autoware XYZIRC point layout (16 bytes per point) expected by
        # BEVFusion on /sensing/lidar/concatenated/pointcloud.
        self.fields = [
            PointField(
                name="x", offset=0,
                datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="y", offset=4,
                datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="z", offset=8,
                datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="intensity", offset=12,
                datatype=PointField.UINT8, count=1
            ),
            PointField(
                name="return_type", offset=13,
                datatype=PointField.UINT8, count=1
            ),
            PointField(
                name="channel", offset=14,
                datatype=PointField.UINT16, count=1
            ),
        ]

        # Follow the actual scene chain rather than assuming sample order.
        # nuScenes samples form a linked list: scene.first_sample_token ->
        # sample.next -> ... -> "" (empty string at the end of the scene).
        scene = self.nusc.scene[0]
        self.scene_name = scene["name"]

        self.sample_tokens = []
        sample_token = scene["first_sample_token"]

        while sample_token:
            self.sample_tokens.append(sample_token)

            sample = self.nusc.get("sample", sample_token)
            sample_token = sample["next"]

        self.sample_tokens = self.sample_tokens[:NUM_SAMPLES]

        # Map each exact LiDAR timestamp to its original sample.
        # BEVFusion copies the input cloud's header stamp onto its output,
        # so the LiDAR stamp (converted from us to ns to match ROS
        # sec/nanosec) is the key that ties a detection to its sample.
        self.sample_by_timestamp = {}
        self.completed_timestamps = set()
        self.timed_out_timestamps = set()
        self.lidar_publish_times = {}

        for index, token in enumerate(self.sample_tokens):
            sample = self.nusc.get("sample", token)
            lidar_data = self.nusc.get(
                "sample_data", sample["data"]["LIDAR_TOP"]
            )
            timestamp_ns = lidar_data["timestamp"] * 1000

            if timestamp_ns in self.sample_by_timestamp:
                raise RuntimeError(
                    f"Duplicate LiDAR timestamp: {timestamp_ns}"
                )

            self.sample_by_timestamp[timestamp_ns] = {
                "sample_index": index + 1,
                "sample_token": token,
                "lidar_timestamp_us": lidar_data["timestamp"],
            }

        if len(self.sample_tokens) != NUM_SAMPLES:
            self.close_output_files()
            raise RuntimeError(
                f"Expected {NUM_SAMPLES} samples, "
                f"found {len(self.sample_tokens)}"
            )

        self.sample_index = 0
        self.current_sample = None
        self.expected_lidar_timestamp_ns = None

        # Cache detections by their exact LiDAR timestamp.
        self.detection_cache = {}
        self.late_detection_count = 0

        # Per-sample state machine, advanced by process_sequence():
        #
        #   PUBLISH_CAMERAS -> load sample, publish 6 images + CameraInfo
        #   WAIT_CAMERA     -> wait until all 6 are echoed back, settle,
        #                      then publish the LiDAR cloud
        #   WAIT_RESULT     -> wait for the detection with the same stamp
        #   (then back to PUBLISH_CAMERAS for the next sample)
        #
        # Cameras go first because BEVFusion only runs inference when a
        # point cloud arrives, using whatever images it has cached.
        self.state = "PUBLISH_CAMERAS"
        self.state_start_time = time.monotonic()
        self.camera_ready_time = None
        self.result_received = False
        self.finished = False
        self.finalizing = False
        self.finalization_start_time = None
        self.output_files_closed = False

        self.success_count = 0
        self.timeout_count = 0
        self.serialization_error_count = 0

        # Drive the state machine by polling every 20 ms instead of
        # blocking, so ROS callbacks keep running in between.
        self.timer = self.create_timer(
            0.02, self.process_sequence
        )

        self.get_logger().info(
            f"Starting sequence: {self.scene_name}, "
            f"{len(self.sample_tokens)} samples"
        )

        self.get_logger().info(
            f"Prediction output: {self.predictions_path}"
        )
        self.get_logger().info(
            f"Frame status output: {self.status_path}"
        )

        for index, token in enumerate(self.sample_tokens):
            self.get_logger().info(
                f"Sample {index + 1}: {token}"
            )

    @staticmethod
    def make_stamp(timestamp_us):
        """Convert a nuScenes microsecond timestamp to a ROS Time stamp."""
        seconds, microseconds = divmod(
            timestamp_us, 1_000_000
        )

        header = Header()
        header.stamp.sec = int(seconds)
        header.stamp.nanosec = int(microseconds * 1000)

        return header.stamp

    @staticmethod
    def to_plain_data(value):
        """Convert ROS ordered dictionaries recursively to YAML-safe data."""
        if isinstance(value, dict):
            return {
                key: NuScenesSequencePublisher.to_plain_data(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                NuScenesSequencePublisher.to_plain_data(item)
                for item in value
            ]

        return value

    def camera_image_callback(self, msg, camera_index):
        """Mark a camera image as delivered if its stamp matches this sample.

        Stale messages from a previous sample are ignored by the exact
        timestamp comparison.
        """
        if self.state != "WAIT_CAMERA":
            return

        expected_timestamp = (
            self.current_sample["camera_timestamps_ns"][camera_index]
        )

        received_timestamp = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        if received_timestamp == expected_timestamp:
            self.camera_image_received[camera_index] = True

    def camera_info_callback(self, msg, camera_index):
        """Same delivery check as camera_image_callback, for CameraInfo."""
        if self.state != "WAIT_CAMERA":
            return

        expected_timestamp = (
            self.current_sample["camera_timestamps_ns"][camera_index]
        )

        received_timestamp = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        if received_timestamp == expected_timestamp:
            self.camera_info_received[camera_index] = True

    def prepare_sample(self):
        """Load the LiDAR sweep and build all ROS messages for one sample.

        Everything is prepared up front so publishing later is fast and
        all six cameras go out back-to-back.
        """
        token = self.sample_tokens[self.sample_index]
        sample = self.nusc.get("sample", token)

        # Load LiDAR data.
        lidar_data = self.nusc.get(
            "sample_data", sample["data"]["LIDAR_TOP"]
        )

        lidar_path = self.dataroot / lidar_data["filename"]

        # 5 float32 values per point: x, y, z, intensity, ring index.
        points = np.fromfile(
            lidar_path, dtype=np.float32
        ).reshape(-1, 5)

        # Load all six cameras belonging to this exact sample.
        camera_messages = []
        camera_deltas_ms = []
        camera_timestamps_ns = []

        for camera_name, camera_id, frame_id in CAMERAS:
            camera_data = self.nusc.get(
                "sample_data", sample["data"][camera_name]
            )

            calibration = self.nusc.get(
                "calibrated_sensor",
                camera_data["calibrated_sensor_token"],
            )

            image_path = self.dataroot / camera_data["filename"]

            with PILImage.open(image_path) as source:
                rgb_image = source.convert("RGB")
                width, height = rgb_image.size
                image_bytes = rgb_image.tobytes()

            if (
                width != camera_data["width"]
                or height != camera_data["height"]
            ):
                raise RuntimeError(
                    f"{camera_name}: image dimensions "
                    "do not match nuScenes metadata"
                )

            # Each camera keeps its own capture time, which differs from the
            # LiDAR time by tens of ms. BEVFusion uses that gap to decide
            # whether to mask a camera out (see computeCameraMasks).
            header = Header()
            header.stamp = self.make_stamp(camera_data["timestamp"])
            camera_timestamps_ns.append(
                header.stamp.sec * 1_000_000_000
                + header.stamp.nanosec
            )
            header.frame_id = frame_id

            image_msg = Image()
            image_msg.header = header
            image_msg.height = height
            image_msg.width = width
            image_msg.encoding = "rgb8"
            image_msg.is_bigendian = 0
            image_msg.step = width * 3
            image_msg.data = image_bytes

            # CameraInfo from the nuScenes 3x3 intrinsic matrix K:
            #   k = K flattened row-major
            #   r = identity (monocular, no rectification rotation)
            #   p = [K | 0], the 3x4 projection with no stereo baseline
            #   d = [] because nuScenes images are already undistorted
            #       (the patch sets run_image_undistortion: false).
            intrinsic = calibration["camera_intrinsic"]

            info_msg = CameraInfo()
            info_msg.header = header
            info_msg.width = width
            info_msg.height = height
            info_msg.distortion_model = "plumb_bob"
            info_msg.d = []

            info_msg.k = [
                value
                for row in intrinsic
                for value in row
            ]

            info_msg.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]

            info_msg.p = [
                intrinsic[0][0],
                intrinsic[0][1],
                intrinsic[0][2],
                0.0,
                intrinsic[1][0],
                intrinsic[1][1],
                intrinsic[1][2],
                0.0,
                intrinsic[2][0],
                intrinsic[2][1],
                intrinsic[2][2],
                0.0,
            ]

            camera_messages.append((image_msg, info_msg))

            delta_ms = (
                camera_data["timestamp"] - lidar_data["timestamp"]
            ) / 1000.0

            camera_deltas_ms.append(delta_ms)

        self.current_sample = {
            "token": token,
            "lidar_timestamp": lidar_data["timestamp"],
            "points": points,
            "camera_messages": camera_messages,
            "camera_deltas_ms": camera_deltas_ms,
            "camera_timestamps_ns": camera_timestamps_ns,
        }

    def publish_cameras(self):
        """Publish all six images + CameraInfo and enter WAIT_CAMERA."""
        self.camera_image_received = [False] * len(CAMERAS)
        self.camera_info_received = [False] * len(CAMERAS)

        for index, (image_msg, info_msg) in enumerate(
            self.current_sample["camera_messages"]
        ):
            self.image_publishers[index].publish(image_msg)
            self.info_publishers[index].publish(info_msg)

        deltas = self.current_sample["camera_deltas_ms"]

        self.get_logger().info(
            f"Sample {self.sample_index + 1}: "
            f"published six cameras; "
            f"camera-LiDAR deltas (ms)="
            f"{[round(value, 3) for value in deltas]}"
        )

        self.state = "WAIT_CAMERA"
        self.state_start_time = time.monotonic()
        self.camera_ready_time = None

    def publish_lidar(self):
        """Publish the LiDAR cloud, which triggers BEVFusion inference.

        Records the expected output stamp, enters WAIT_RESULT, then checks
        whether the detection for this stamp is already in the cache.
        """
        points = self.current_sample["points"]

        # Repack to XYZIRC: intensity clamped to UINT8, return_type is
        # unknown in nuScenes (0), ring index becomes the channel.
        packed_points = [
            (
                float(x),
                float(y),
                float(z),
                int(np.clip(intensity, 0, 255)),
                0,
                int(ring),
            )
            for x, y, z, intensity, ring in points
        ]

        header = Header()
        header.stamp = self.make_stamp(
            self.current_sample["lidar_timestamp"]
        )
        header.frame_id = "lidar_top"

        msg = point_cloud2.create_cloud(
            header=header,
            fields=self.fields,
            points=packed_points,
        )

        self.expected_lidar_timestamp_ns = (
            header.stamp.sec * 1_000_000_000
            + header.stamp.nanosec
        )

        self.result_received = False
        self.lidar_publish_times[self.expected_lidar_timestamp_ns] = (
            time.monotonic()
        )

        self.lidar_publisher.publish(msg)

        self.get_logger().info(
            f"Sample {self.sample_index + 1}: "
            f"published LiDAR cloud with {len(points)} points; "
            f"timestamp={self.current_sample['lidar_timestamp']} us"
        )

        self.state = "WAIT_RESULT"
        self.state_start_time = self.lidar_publish_times[
            self.expected_lidar_timestamp_ns
        ]

        self.cleanup_detection_cache()

        # Handle the case where the detection was received and cached
        # before this sample reached WAIT_RESULT.
        cached_entry = self.detection_cache.pop(
            self.expected_lidar_timestamp_ns, None
        )

        if cached_entry is not None:
            self.detection_callback(cached_entry["message"])

    def write_prediction(self, msg):
        """Write one timestamp-matched ROS detection message as a YAML document."""
        ros_data = message_to_ordereddict(msg)
        plain_data = self.to_plain_data(ros_data)

        yaml.safe_dump(
            plain_data,
            self.predictions_file,
            explicit_start=True,
            sort_keys=False,
        )

        self.predictions_file.flush()

    def write_status(
        self,
        status,
        detection_count=0,
        wait_seconds=None,
        error="",
    ):
        """Append one row for the current sample to the frame-status CSV.

        Flushed immediately so results survive a crash or Ctrl-C.
        """
        if self.current_sample is None:
            return

        timestamp_us = self.current_sample["lidar_timestamp"]
        timestamp_ns = timestamp_us * 1000

        self.status_writer.writerow({
            "sample_index": self.sample_index + 1,
            "scene": self.scene_name,
            "sample_token": self.current_sample["token"],
            "lidar_timestamp_us": timestamp_us,
            "lidar_timestamp_ns": timestamp_ns,
            "status": status,
            "detection_count": detection_count,
            "wait_seconds": (
                f"{wait_seconds:.6f}"
                if wait_seconds is not None else ""
            ),
            "camera_deltas_ms": ";".join(
                f"{value:.3f}"
                for value in self.current_sample["camera_deltas_ms"]
            ),
            "error": error,
        })

        self.status_file.flush()

    def write_late_status(self, timestamp_ns, detection_count, wait_seconds):
        """Record a late detection against its original nuScenes sample."""
        sample_info = self.sample_by_timestamp[timestamp_ns]

        self.status_writer.writerow({
            "sample_index": sample_info["sample_index"],
            "scene": self.scene_name,
            "sample_token": sample_info["sample_token"],
            "lidar_timestamp_us": sample_info["lidar_timestamp_us"],
            "lidar_timestamp_ns": timestamp_ns,
            "status": "late_success",
            "detection_count": detection_count,
            "wait_seconds": f"{wait_seconds:.6f}",
            "camera_deltas_ms": "",
            "error": "Detection arrived after the original sample timed out",
        })

        self.status_file.flush()

    def advance_sample(self):
        """Reset per-sample state and go back to PUBLISH_CAMERAS."""
        self.sample_index += 1
        self.current_sample = None
        self.expected_lidar_timestamp_ns = None
        self.result_received = False

        self.state = "PUBLISH_CAMERAS"
        self.state_start_time = time.monotonic()

    def cleanup_detection_cache(self):
        """Remove expired cached detections and enforce the cache size limit."""
        now = time.monotonic()

        expired_timestamps = [
            timestamp_ns
            for timestamp_ns, entry in self.detection_cache.items()
            if now - entry["received_at"] >= DETECTION_CACHE_TTL_SECONDS
        ]

        for timestamp_ns in expired_timestamps:
            self.detection_cache.pop(timestamp_ns, None)

            self.get_logger().warning(
                f"Expired cached detection: timestamp={timestamp_ns}"
            )

        while len(self.detection_cache) > DETECTION_CACHE_MAX_ENTRIES:
            oldest_timestamp = min(
                self.detection_cache,
                key=lambda timestamp: self.detection_cache[timestamp]["received_at"],
            )

            self.detection_cache.pop(oldest_timestamp)

            self.get_logger().warning(
                f"Evicted oldest cached detection: "
                f"timestamp={oldest_timestamp}"
            )

    def detection_callback(self, msg):
        """Match a BEVFusion detection to its sample by LiDAR timestamp.

        Four cases:
          * stamp not in this sequence, or already saved -> ignore
          * stamp is the sample we are waiting on       -> save as success
          * stamp belongs to a sample that timed out    -> save as late_success
          * stamp published but not yet awaited         -> cache it for
                                                           publish_lidar()
        """
        self.cleanup_detection_cache()

        output_timestamp_ns = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        # Ignore detections outside this sequence.
        if output_timestamp_ns not in self.sample_by_timestamp:
            return

        # Ignore duplicate results for a sample already saved.
        if output_timestamp_ns in self.completed_timestamps:
            return

        detection_count = len(msg.objects)
        sample_info = self.sample_by_timestamp[output_timestamp_ns]

        # The detection belongs to the currently active sample.
        if (
            self.state == "WAIT_RESULT"
            and output_timestamp_ns == self.expected_lidar_timestamp_ns
        ):
            elapsed = time.monotonic() - self.state_start_time

            try:
                self.write_prediction(msg)
            except Exception as exc:
                self.serialization_error_count += 1
                self.completed_timestamps.add(output_timestamp_ns)

                self.get_logger().error(
                    f"PREDICTION SERIALIZATION ERROR: "
                    f"sample={sample_info['sample_index']}, "
                    f"error={exc}"
                )

                self.write_status(
                    status="serialization_error",
                    detection_count=detection_count,
                    wait_seconds=elapsed,
                    error=str(exc),
                )

                self.result_received = True
                return

            self.completed_timestamps.add(output_timestamp_ns)
            self.result_received = True
            self.success_count += 1

            self.write_status(
                status="success",
                detection_count=detection_count,
                wait_seconds=elapsed,
            )

            self.get_logger().info(
                f"DETECTION SAVED: "
                f"sample={sample_info['sample_index']}, "
                f"boxes={detection_count}, "
                f"timestamp={msg.header.stamp.sec}."
                f"{msg.header.stamp.nanosec:09d}, "
                f"wait={elapsed:.3f}s"
            )
            return

        # A detection arrived after its sample had timed out.
        if output_timestamp_ns in self.timed_out_timestamps:
            try:
                self.write_prediction(msg)
            except Exception as exc:
                self.serialization_error_count += 1
                self.get_logger().error(
                    f"LATE PREDICTION SERIALIZATION ERROR: "
                    f"sample={sample_info['sample_index']}, "
                    f"error={exc}"
                )
                return

            self.completed_timestamps.add(output_timestamp_ns)
            self.late_detection_count += 1

            self.write_late_status(
                output_timestamp_ns,
                detection_count,
                time.monotonic()
                - self.lidar_publish_times[output_timestamp_ns],
            )

            self.get_logger().info(
                f"LATE DETECTION SAVED: "
                f"sample={sample_info['sample_index']}, "
                f"boxes={detection_count}, "
                f"timestamp={msg.header.stamp.sec}."
                f"{msg.header.stamp.nanosec:09d}"
            )
            return

        # Do not cache results for samples whose LiDAR cloud was never published.
        if output_timestamp_ns not in self.lidar_publish_times:
            return

        # Retain a result that arrived before its sample's WAIT_RESULT state.
        self.detection_cache[output_timestamp_ns] = {
            "message": msg,
            "received_at": time.monotonic(),
        }

        self.cleanup_detection_cache()

    def process_sequence(self):
        """Timer callback (every 20 ms) that advances the state machine.

        Every wait has a timeout, so a dropped message is recorded in the
        status CSV instead of stalling the whole run.
        """
        if self.finished:
            return

        if self.state == "PUBLISH_CAMERAS":
            # All samples done: keep spinning for a grace period so late
            # detections for timed-out samples can still be recorded.
            if self.sample_index >= len(self.sample_tokens):
                if not self.finalizing:
                    self.finalizing = True
                    self.finalization_start_time = time.monotonic()

                    self.get_logger().info(
                        "All samples processed; entering final "
                        f"{FINAL_DETECTION_GRACE_SECONDS:.1f}s "
                        "detection collection window"
                    )

                elif (
                    time.monotonic() - self.finalization_start_time
                    >= FINAL_DETECTION_GRACE_SECONDS
                ):
                    self.finish_sequence()

                return

            self.prepare_sample()
            self.publish_cameras()

        elif self.state == "WAIT_CAMERA":
            elapsed = time.monotonic() - self.state_start_time

            all_images_received = all(self.camera_image_received)
            all_info_received = all(self.camera_info_received)

            if all_images_received and all_info_received:
                if self.camera_ready_time is None:
                    self.camera_ready_time = time.monotonic()

                    self.get_logger().info(
                        f"Sample {self.sample_index + 1}: "
                        "all six camera images and CameraInfo "
                        "received with matching timestamps"
                    )

                settle_elapsed = (
                    time.monotonic() - self.camera_ready_time
                )

                # The settle delay gives BEVFusion's own callbacks time to
                # cache the images. Checking for a LiDAR subscriber avoids
                # publishing a cloud that nobody receives.
                if settle_elapsed >= CAMERA_SETTLE_SECONDS:
                    subscriber_count = (
                        self.lidar_publisher.get_subscription_count()
                    )

                    if subscriber_count > 0:
                        self.get_logger().info(
                            f"Sample {self.sample_index + 1}: "
                            f"LiDAR subscriber detected "
                            f"(count={subscriber_count})"
                        )
                        self.publish_lidar()

                    elif (
                        time.monotonic() - self.camera_ready_time
                        >= CAMERA_SETTLE_SECONDS
                        + LIDAR_SUBSCRIBER_TIMEOUT_SECONDS
                    ):
                        self.timeout_count += 1

                        self.get_logger().error(
                            f"LIDAR SUBSCRIBER TIMEOUT: "
                            f"sample={self.sample_index + 1}, "
                            "no subscribers detected"
                        )

                        self.write_status(
                            status="lidar_subscriber_timeout",
                            error=(
                                "No LiDAR subscribers detected within "
                                f"{LIDAR_SUBSCRIBER_TIMEOUT_SECONDS:.1f}s"
                            ),
                        )

                        self.advance_sample()
            elif elapsed >= CAMERA_READINESS_TIMEOUT_SECONDS:
                self.timeout_count += 1

                missing_images = [
                    CAMERAS[i][0]
                    for i, received in enumerate(
                        self.camera_image_received
                    )
                    if not received
                ]

                missing_info = [
                    CAMERAS[i][0]
                    for i, received in enumerate(
                        self.camera_info_received
                    )
                    if not received
                ]

                self.get_logger().error(
                    f"CAMERA READINESS TIMEOUT: "
                    f"sample={self.sample_index + 1}, "
                    f"missing_images={missing_images}, "
                    f"missing_camera_info={missing_info}"
                )

                self.write_status(
                    status="camera_readiness_timeout",
                    error=(
                        f"Missing images: {missing_images}; "
                        f"missing CameraInfo: {missing_info}"
                    ),
                )

                self.advance_sample()

        elif self.state == "WAIT_RESULT":
            if self.result_received:
                self.advance_sample()
                return

            elapsed = time.monotonic() - self.state_start_time

            if elapsed >= DETECTION_TIMEOUT_SECONDS:
                self.timeout_count += 1

                self.get_logger().error(
                    f"DETECTION TIMEOUT: "
                    f"sample={self.sample_index + 1}, "
                    f"token={self.current_sample['token']}, "
                    f"timestamp="
                    f"{self.current_sample['lidar_timestamp']} us"
                )

                self.write_status(
                    status="timeout",
                    detection_count=0,
                    wait_seconds=elapsed,
                    error="No matching detection received before timeout",
                )

                self.timed_out_timestamps.add(
                    self.expected_lidar_timestamp_ns
                )

                self.advance_sample()

    def close_output_files(self):
        if self.output_files_closed:
            return

        if hasattr(self, "predictions_file"):
            if not self.predictions_file.closed:
                self.predictions_file.close()

        if hasattr(self, "status_file"):
            if not self.status_file.closed:
                self.status_file.close()

        self.output_files_closed = True

    def finish_sequence(self):
        if self.finished:
            return

        self.finished = True
        self.timer.cancel()

        self.close_output_files()

        self.get_logger().info(
            "SEQUENCE COMPLETE"
        )

        self.get_logger().info(
            f"Successful frames: {self.success_count}"
        )

        self.get_logger().info(
            f"Timed-out frames: {self.timeout_count}"
        )

        self.get_logger().info(
            f"Late detections recovered: {self.late_detection_count}"
        )

        self.get_logger().info(
            f"Serialization errors: {self.serialization_error_count}"
        )

        self.get_logger().info(
            f"Prediction file: {self.predictions_path}"
        )

        self.get_logger().info(
            f"Status file: {self.status_path}"
        )

        # Stop spinning after all samples have been processed.
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = NuScenesSequencePublisher()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close_output_files()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

