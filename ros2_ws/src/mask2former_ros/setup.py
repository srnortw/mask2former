from setuptools import find_packages, setup

package_name = "mask2former_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/segmentation_client.launch.py"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
    ],
    install_requires=["setuptools", "requests", "opencv-python-headless", "numpy"],
    zip_safe=True,
    maintainer="srnortw",
    maintainer_email="serkan@example.com",
    description="Mask2Former HTTP client ROS2 node (Mode A)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "segmentation_client = mask2former_ros.segmentation_client:main",
            "publish_test_image = mask2former_ros.publish_test_image:main",
        ],
    },
)
