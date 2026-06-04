#!/usr/bin/env python3
"""Subscribe once to a sensor_msgs/Image topic and save as JPEG (for quick tests)."""

import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class SaveOnce(Node):
    def __init__(self, topic: str, out_path: str) -> None:
        super().__init__("save_ros_viz_once")
        self.out_path = out_path
        self.bridge = CvBridge()
        self.done = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f"Waiting for one message on {topic} ...")

    def _cb(self, msg: Image) -> None:
        if self.done:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(self.out_path, bgr)
        self.get_logger().info(f"Saved {self.out_path} ({bgr.shape[1]}x{bgr.shape[0]})")
        self.done = True
        raise SystemExit(0)


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "/perception/instance_masks/visualization"
    out = sys.argv[2] if len(sys.argv) > 2 else "reports/ros2_viz_from_topic.jpg"
    rclpy.init()
    node = SaveOnce(topic, out)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
