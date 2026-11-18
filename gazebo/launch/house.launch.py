"""Gazebo Fortress + ros_gz bridge + Nav2 for one exported house (Point 9e).

    source /opt/ros/humble/setup.bash
    ros2 launch gazebo/launch/house.launch.py house:=200000            # with the Gazebo window
    ros2 launch gazebo/launch/house.launch.py house:=200000 headless:=true

Export the house first: python -m trustmap.gazebo_export 200000

Frames: map = odom = the Gazebo world frame (the shared grid's coordinates).
The robot's pose comes from Gazebo's OdometryPublisher (ground truth), which
publishes odom -> base_link; there is no AMCL (OPEN_QUESTIONS Q11).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

HERE = os.path.dirname(os.path.abspath(__file__))
GAZEBO_DIR = os.path.dirname(HERE)

# Sensor mounts relative to base_link (the model origin on the floor); must match
# trustmap/gazebo_export.py robot_sdf().
LIDAR_XYZ = ("0", "0", "0.24")
CAMERA_XYZ = ("0.18", "0", "0.29")


def _setup(context):
    house = LaunchConfiguration("house").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    world_dir = os.path.join(GAZEBO_DIR, "worlds", f"house_{house}")
    if not os.path.isdir(world_dir):
        raise RuntimeError(f"{world_dir} not found; run: python -m trustmap.gazebo_export {house}")
    params = os.path.join(GAZEBO_DIR, "config", "nav2_params.yaml")

    gz_cmd = ["ign", "gazebo", "-r", os.path.join(world_dir, "world.sdf")]
    if headless:
        gz_cmd[3:3] = ["-s", "--headless-rendering"]

    def static_tf(name, parent, child, xyz):
        return Node(package="tf2_ros", executable="static_transform_publisher", name=name,
                    arguments=["--x", xyz[0], "--y", xyz[1], "--z", xyz[2],
                               "--frame-id", parent, "--child-frame-id", child],
                    parameters=[{"use_sim_time": True}])

    nav2 = os.path.join(get_package_share_directory("nav2_bringup"), "launch", "navigation_launch.py")
    return [
        ExecuteProcess(cmd=gz_cmd, output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="bridge", output="screen",
             parameters=[{"config_file": os.path.join(GAZEBO_DIR, "config", "bridge.yaml"),
                          "use_sim_time": True}]),
        static_tf("map_to_odom", "map", "odom", ("0", "0", "0")),
        static_tf("base_to_lidar", "base_link", "lidar_link", LIDAR_XYZ),
        static_tf("base_to_camera", "base_link", "camera_link", CAMERA_XYZ),
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[params, {"yaml_filename": os.path.join(world_dir, "map.yaml"), "use_sim_time": True}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map",
             output="screen",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2),
            launch_arguments={"use_sim_time": "true", "params_file": params, "autostart": "true",
                              "use_composition": "False"}.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("house", default_value="200000", description="test-house seed"),
        DeclareLaunchArgument("headless", default_value="false", description="run Gazebo without a window"),
        OpaqueFunction(function=_setup),
    ])
