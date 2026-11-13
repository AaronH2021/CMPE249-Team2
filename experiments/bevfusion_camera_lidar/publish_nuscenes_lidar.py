#!/usr/bin/env python3
"""Single-frame LiDAR smoke test for BEVFusion.

Republishes the first nuScenes Mini LiDAR sweep once per second, always with
the same original timestamp. Useful for checking that BEVFusion is wired up;
for real evaluation use ``publish_nuscenes_sequence.py``, which also
publishes cameras and records the detections.
"""

import numpy as np
import rclpy

from pathlib import Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from nuscenes.nuscenes import NuScenes


class NuScenesLidarPublisher(Node):
    def __init__(self):
        super().__init__("nuscenes_lidar_publisher")

        self.publisher = self.create_publisher(
            PointCloud2,
            "/sensing/lidar/concatenated/pointcloud",
            10,
        )

        # Load the exact nuScenes sample used by the camera publisher.
        dataroot = str(Path.home() / "datasets/nuscenes")
        nusc = NuScenes(
            version="v1.0-mini",
            dataroot=dataroot,
            verbose=False,
        )

        sample = nusc.sample[0]
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

        self.scan_path = Path(dataroot) / lidar_data["filename"]

        # Preserve the original nuScenes LiDAR timestamp.
        self.timestamp_us = lidar_data["timestamp"]

        # nuScenes .pcd.bin files are flat float32 arrays with 5 values per
        # point: x, y, z, intensity (0-255), ring index (0-31).
        self.points = np.fromfile(
            self.scan_path, dtype=np.float32
        ).reshape(-1, 5)

        # Autoware's XYZIRC point layout (16 bytes per point), which is what
        # BEVFusion's preprocessing expects on the concatenated cloud topic.
        self.fields = [
            PointField(name="x", offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.UINT8, count=1),
            PointField(name="return_type", offset=13,
                       datatype=PointField.UINT8, count=1),
            PointField(name="channel", offset=14,
                       datatype=PointField.UINT16, count=1),
        ]

        # Republish the same frame every second so a late-starting
        # BEVFusion node still receives it.
        self.timer = self.create_timer(1.0, self.publish_scan)

        self.get_logger().info(
            f"Loaded {len(self.points)} points from {self.scan_path.name}"
        )
        self.get_logger().info(
            f"Using nuScenes timestamp: {self.timestamp_us} us"
        )

    def publish_scan(self):
        """Convert the nuScenes sweep to XYZIRC and publish it.

        Intensity is clamped into UINT8 range, return_type is unknown in
        nuScenes so it is set to 0, and the ring index becomes ``channel``.
        """
        packed_points = [
            (
                float(x),
                float(y),
                float(z),
                int(np.clip(intensity, 0, 255)),
                0,
                int(ring),
            )
            for x, y, z, intensity, ring in self.points
        ]

        msg = point_cloud2.create_cloud(
            header=self._make_header(),
            fields=self.fields,
            points=packed_points,
        )

        self.publisher.publish(msg)

        self.get_logger().info(
            f"Published {len(self.points)} points to "
            "/sensing/lidar/concatenated/pointcloud"
        )

    def _make_header(self):
        """Build a header stamped with the original nuScenes time.

        nuScenes stores timestamps in microseconds; ROS splits time into
        whole seconds plus nanoseconds. Keeping the dataset time (instead
        of "now") lets the output detections be matched back to the sample.
        """
        header = Header()

        timestamp_sec, timestamp_remainder_us = divmod(
            self.timestamp_us, 1_000_000
        )

        header.stamp.sec = int(timestamp_sec)
        header.stamp.nanosec = int(timestamp_remainder_us * 1000)
        header.frame_id = "lidar_top"

        return header


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesLidarPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

