
#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class NuScenesCameraTFPublisher(Node):
    def __init__(self):
        super().__init__("nuscenes_camera_tf_publisher")

        self.broadcaster = StaticTransformBroadcaster(self)

        # Transform: lidar_top -> camera optical frame
        # Translation and quaternion values calculated from
        # the first nuScenes Mini sample's calibrated_sensor records.
        camera_transforms = [
            (
                "cam_front_optical_frame",
                [-0.0124633846, 0.7648668889, -0.3109103442],
                [-0.7005017073, -0.0036449450,
                 -0.0011340526, 0.7136405162],
            ),
            (
                "cam_front_right_optical_frame",
                [0.4966619828, 0.6142281759, -0.3267711275],
                [-0.6170008416, 0.3420364534,
                 -0.3273897702, 0.6285992081],
            ),
            (
                "cam_front_left_optical_frame",
                [-0.4914898682, 0.5890869724, -0.3196275959],
                [-0.6199902995, -0.3314581696,
                 0.3219698409, 0.6341000961],
            ),
            (
                "cam_back_optical_frame",
                [-0.0037719758, -0.9087816307, -0.2832560752],
                [0.0051135943, 0.7097400489,
                 -0.7044427580, 0.0018206560],
            ),
            (
                "cam_back_left_optical_frame",
                [-0.4831280315, 0.0990449361, -0.2497880224],
                [0.4121191501, 0.5871938533,
                 -0.5601146227, -0.4142858847],
            ),
            (
                "cam_back_right_optical_frame",
                [0.4823426875, 0.0768302392, -0.2732123125],
                [-0.3944526630, 0.5950676329,
                 -0.5695145201, 0.4073755273],
            ),
        ]

        transforms = []

        for child_frame, translation, rotation in camera_transforms:
            transform = TransformStamped()

            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "lidar_top"
            transform.child_frame_id = child_frame

            transform.transform.translation.x = translation[0]
            transform.transform.translation.y = translation[1]
            transform.transform.translation.z = translation[2]

            transform.transform.rotation.x = rotation[0]
            transform.transform.rotation.y = rotation[1]
            transform.transform.rotation.z = rotation[2]
            transform.transform.rotation.w = rotation[3]

            transforms.append(transform)

        self.broadcaster.sendTransform(transforms)

        self.get_logger().info(
            "Published six nuScenes camera transforms from lidar_top."
        )


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesCameraTFPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

