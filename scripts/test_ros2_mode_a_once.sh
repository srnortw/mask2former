#!/usr/bin/env bash
# One-frame Mode A test: publish image → HTTP client → save ROS visualization.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${REPO_ROOT}/ros2_ws"
IMAGE_PATH="${1:-}"

if [[ -z "${IMAGE_PATH}" || ! -f "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/test_image.jpg"
  exit 1
fi

export PYTHONNOUSERSITE=1
# shellcheck source=/dev/null
source /opt/ros/jazzy/setup.bash
[[ -f "${WS}/install/setup.bash" ]] || "${REPO_ROOT}/scripts/build_ros2.sh"
# shellcheck source=/dev/null
source "${WS}/install/setup.bash"

OUT_API="${REPO_ROOT}/reports/ros2_test_overlay.jpg"
OUT_ROS="${REPO_ROOT}/reports/ros2_viz_from_topic.jpg"

(cd "${REPO_ROOT}" && docker compose up -d)
curl -sf http://localhost:8000/health >/dev/null || {
  echo "API not ready on :8000"
  exit 1
}

echo "=== Direct API overlay ==="
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/visualize_predict.py" \
  "${IMAGE_PATH}" -o "${OUT_API}"

ros2 run mask2former_ros publish_test_image --ros-args \
  -p image_path:="${IMAGE_PATH}" -p rate_hz:=2.0 > /tmp/ros2_pub.log 2>&1 &
PUB_PID=$!

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/save_ros_viz_once.py" \
  /perception/instance_masks/visualization "${OUT_ROS}" > /tmp/ros2_save.log 2>&1 &
SAVE_PID=$!

sleep 2
timeout 30 ros2 run mask2former_ros segmentation_client 2>&1 | tee /tmp/ros2_client.log || true

kill $PUB_PID $SAVE_PID 2>/dev/null || true
wait $SAVE_PID 2>/dev/null || true

echo ""
grep "Detected" /tmp/ros2_client.log | tail -3 || tail -10 /tmp/ros2_client.log
echo ""
echo "Saved:"
echo "  ${OUT_API}  (direct POST /predict)"
echo "  ${OUT_ROS}  (ROS topic visualization)"
ls -la "${OUT_API}" "${OUT_ROS}"
