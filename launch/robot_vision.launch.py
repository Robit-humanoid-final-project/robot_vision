from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_info_yaml_file_path = PathJoinSubstitution(
        [FindPackageShare("insta360_usb_cam"), "config", "camera_info_config.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="robot_vision",
                executable="camera_node",
                name="camera_node",
                output="screen",
                emulate_tty=True,
                parameters=[camera_info_yaml_file_path],
            ),
        ]
    )
