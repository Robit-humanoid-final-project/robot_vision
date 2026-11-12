import os

import cv2 as cv
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile


class ImageInfoSubNode(Node):
    def __init__(self):
        super().__init__("image_info_sub_node")

        self.declare_parameter("topic_name", "camera/info")
        self.declare_parameter("device_name", "AUTO")

        topic_name = self.get_parameter("topic_name").value
        device_name = self.get_parameter("device_name").value

        self.bridge = CvBridge()

    def image_info_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ImageInfoSubNode()
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
