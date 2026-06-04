"""
Mode A — ROS2 thin client: subscribe to camera, POST /predict, publish masks.

Requires FastAPI server (docker compose or GHCR image) on the same network.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from mask2former_ros.api_utils import (
    build_combined_mask,
    build_visualization,
    decode_instance_masks,
)


class SegmentationClientNode(Node):
    def __init__(self) -> None:
        super().__init__("mask2former_client")

        self.declare_parameter("server_url", "http://localhost:8000")
        self.declare_parameter("predict_path", "/predict")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/perception/instance_masks")
        self.declare_parameter("publish_viz", True)
        self.declare_parameter("request_timeout_sec", 120.0)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("skip_if_busy", True)
        self.declare_parameter("conf_threshold", 0.5)

        server_url = self.get_parameter("server_url").value.rstrip("/")
        predict_path = self.get_parameter("predict_path").value
        self.predict_url = f"{server_url}{predict_path}"
        self.request_timeout = float(self.get_parameter("request_timeout_sec").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.skip_if_busy = bool(self.get_parameter("skip_if_busy").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        publish_viz = bool(self.get_parameter("publish_viz").value)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._in_flight = False
        self._latest_msg: Optional[Image] = None

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, input_topic, self._image_callback, qos)

        self.mask_pub = self.create_publisher(Image, output_topic, 10)
        self.viz_pub = None
        if publish_viz:
            self.viz_pub = self.create_publisher(
                Image,
                output_topic + "/visualization",
                10,
            )

        self._check_health(server_url)
        self.get_logger().info("Mode A HTTP client ready")
        self.get_logger().info(f"  POST {self.predict_url}")
        self.get_logger().info(f"  subscribe: {input_topic}")
        self.get_logger().info(f"  publish:   {output_topic}")
        self.get_logger().info(f"  conf_threshold: {self.conf_threshold}")

    def _check_health(self, server_url: str) -> None:
        try:
            resp = requests.get(f"{server_url}/health", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("loaded"):
                self.get_logger().warn("API reports model not loaded yet")
            else:
                self.get_logger().info(
                    f"API health OK — model={data.get('model')} loaded={data.get('loaded')}"
                )
        except requests.RequestException as exc:
            self.get_logger().warn(
                f"Could not reach API at {server_url}/health ({exc}). "
                "Start server: docker compose up -d"
            )

    def _image_callback(self, msg: Image) -> None:
        with self._lock:
            if self.skip_if_busy and self._in_flight:
                return
            self._latest_msg = msg
            self._in_flight = True

        try:
            self._process(msg)
        finally:
            with self._lock:
                self._in_flight = False

    def _process(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        ok, buffer = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().error("JPEG encode failed")
            return

        try:
            resp = requests.post(
                self.predict_url,
                params={"conf_threshold": self.conf_threshold},
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self.get_logger().error(f"POST {self.predict_url} failed: {exc}")
            return

        instances = decode_instance_masks(
            data["instances"],
            bgr.shape[0],
            bgr.shape[1],
        )
        combined = build_combined_mask(instances, bgr.shape[0], bgr.shape[1])

        mask_msg = self.bridge.cv2_to_imgmsg(combined, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        if self.viz_pub is not None:
            viz = build_visualization(bgr, instances)
            viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)

        self.get_logger().info(
            f"Detected {len(instances)} instances in "
            f"{data.get('inference_ms', 0.0):.1f} ms (server ONNX)"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
