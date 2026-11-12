import os
import time

import cv2 as cv
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Best Effort, Volatile, Keep Last(5)
from sensor_msgs.msg import Image


class VisionObservPubNode(Node):
    def __init__(self):
        super().__init__("vision_observ_pub_node")


def main(args=None):
    rclpy.init(args=args)
    node = VisionObservPubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
