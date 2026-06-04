# ROS2 workspace — Mask2Former (Mode A)

**Mode A:** ROS2 Jazzy node → HTTP `POST /predict` → Docker FastAPI (Phase 06).

Python extras (`requests`, OpenCV) use the repo **`.venv`**, not system `pip` — same as monitoring and `visualize_predict.py`.

## Prerequisites

- Ubuntu 24.04 + [ROS2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- Project venv at `~/Desktop/mask2former/.venv`
- Inference API: `docker compose up -d` from repo root

## Build (uses `.venv`)

From repo root:

```bash
./scripts/build_ros2.sh
```

Or manually:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Desktop/mask2former
.venv/bin/pip install -r requirements-ros2-client.txt
cd ros2_ws
COLCON_PYTHON_EXECUTABLE=../.venv/bin/python colcon build --packages-select mask2former_ros
source install/setup.bash
```

`rclpy` / `cv_bridge` come from ROS2; `requests` and OpenCV come from **`.venv`**.

## Run (test without camera)

Terminal 1 — API:

```bash
cd ~/Desktop/mask2former && docker compose up -d
```

Terminal 2 — fake camera:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Desktop/mask2former/ros2_ws/install/setup.bash
ros2 run mask2former_ros publish_test_image --ros-args \
  -p image_path:=/path/to/your/frame.jpg
```

Terminal 3 — HTTP client:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Desktop/mask2former/ros2_ws/install/setup.bash
ros2 launch mask2former_ros segmentation_client.launch.py
```

Helper (starts API, prints commands):

```bash
./scripts/run_ros2_mode_a.sh /path/to/frame.jpg
```

## RViz2

```bash
rviz2
# Add → Image → Topic: /perception/instance_masks/visualization
```

## Topics

| Topic | Type | Notes |
|-------|------|--------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input |
| `/perception/instance_masks` | `sensor_msgs/Image` | `mono8`, pixel = instance_id+1 |
| `/perception/instance_masks/visualization` | `sensor_msgs/Image` | BGR overlay |

## Mode B (later)

Embedded ONNX on-robot — will reuse `src/inference.py`; not in this package yet.
