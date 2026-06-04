import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("mask2former_ros")
    default_params = os.path.join(pkg_share, "config", "params.yaml")

    server_url_arg = DeclareLaunchArgument(
        "server_url",
        default_value="http://localhost:8000",
        description="FastAPI base URL (docker compose or GHCR host)",
    )

    client_node = Node(
        package="mask2former_ros",
        executable="segmentation_client",
        name="mask2former_client",
        output="screen",
        parameters=[
            default_params,
            {
                "server_url": LaunchConfiguration("server_url"),
            },
        ],
    )

    return LaunchDescription([server_url_arg, client_node])
