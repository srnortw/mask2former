"""
Republish a static image file as sensor_msgs/Image (for testing without a camera).

Example:
  ros2 run mask2former_ros publish_test_image --ros-args \
    -p image_path:=/path/to/frame.jpg -p rate_hz:=1.0
"""

from __future__ import annotations

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class PublishTestImageNode(Node):
    def __init__(self) -> None:
        super().__init__("publish_test_image")

        self.declare_parameter("image_path", "")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("frame_id", "camera")

        image_path = self.get_parameter("image_path").value
        if not image_path or not os.path.isfile(image_path):
            raise RuntimeError(
                f"Set parameter image_path to a valid file (got: {image_path!r})"
            )

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        self._bgr = bgr
        self._bridge = CvBridge()
        topic = self.get_parameter("topic").value
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(Image, topic, 10)
        period = 1.0 / max(rate_hz, 0.1)
        self.timer = self.create_timer(period, self._publish)
        self.get_logger().info(
            f"Publishing {image_path} ({bgr.shape[1]}x{bgr.shape[0]}) "
            f"on {topic} at {rate_hz:.2f} Hz"
        )

    def _publish(self) -> None:
        msg = self._bridge.cv2_to_imgmsg(self._bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PublishTestImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
