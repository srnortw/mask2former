#!/usr/bin/env bash
# Mode A smoke test helper: docker API + instructions for ROS2 nodes (.venv build).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${REPO_ROOT}/ros2_ws"
IMAGE_PATH="${1:-}"
# Avoid ~/.local numpy 2.x breaking cv_bridge when using .venv nodes
export PYTHONNOUSERSITE=1

if [[ -z "${IMAGE_PATH}" || ! -f "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/test_image.jpg"
  echo "Example: $0 ~/Pictures/lane_frame.jpg"
  exit 1
fi

if [[ ! -f "${WS}/install/setup.bash" ]]; then
  "${REPO_ROOT}/scripts/build_ros2.sh"
fi

# shellcheck source=/dev/null
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
# shellcheck source=/dev/null
source "${WS}/install/setup.bash"

echo "Starting API (docker compose)..."
(cd "${REPO_ROOT}" && docker compose up -d)

echo "Waiting for /health..."
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 2
done
curl -s http://localhost:8000/health | head -c 200
echo

ROS_SETUP="source /opt/ros/jazzy/setup.bash && source ${WS}/install/setup.bash"

echo ""
echo "Open two terminals (ROS2 built with .venv — see scripts/build_ros2.sh):"
echo ""
echo "  # Terminal A — test camera"
echo "  ${ROS_SETUP}"
echo "  ros2 run mask2former_ros publish_test_image --ros-args -p image_path:=${IMAGE_PATH}"
echo ""
echo "  # Terminal B — HTTP client"
echo "  ${ROS_SETUP}"
echo "  ros2 launch mask2former_ros segmentation_client.launch.py"
echo ""
echo "RViz2: topic /perception/instance_masks/visualization"
