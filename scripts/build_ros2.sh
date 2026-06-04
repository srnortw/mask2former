#!/usr/bin/env bash
# Build mask2former_ros with project .venv (same pattern as run_drift_report.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${REPO_ROOT}/ros2_ws"
VENV="${REPO_ROOT}/.venv"

if ! command -v ros2 &>/dev/null; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash
fi

[[ -d "${VENV}" ]] || {
  echo "Create venv first: cd ${REPO_ROOT} && python3 -m venv .venv"
  exit 1
}

# cv_bridge needs numpy 1.x; pin in requirements-ros2-client.txt
"${VENV}/bin/pip" install -q -r "${REPO_ROOT}/requirements-ros2-client.txt"

export COLCON_PYTHON_EXECUTABLE="${VENV}/bin/python"
cd "${WS}"
colcon build --packages-select mask2former_ros "$@"

echo ""
echo "Done. Before ros2 run:"
echo "  export PYTHONNOUSERSITE=1   # avoids ~/.local numpy breaking cv_bridge"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source ${WS}/install/setup.bash"
