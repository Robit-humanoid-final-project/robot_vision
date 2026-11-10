import os
import time

import cv2 as cv
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# 분리된 추론 모듈 import
from robot_vision.detector import Detector


class ImageSubNode(Node):
    def __init__(self):
        super().__init__("image_sub_node")

        # 설치된 패키지 경로에서 모델 가져오기
        package_share_dir = get_package_share_directory("robot_vision")
        default_model_path = os.path.join(package_share_dir, "model", "best.xml")

        self.declare_parameter("topic_name", "camera1/camera/image_raw")
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device_name", "AUTO")

        topic_name = self.get_parameter("topic_name").value
        model_path = self.get_parameter("model_path").value
        device_name = self.get_parameter("device_name").value

        # OpenVINO 추론 model
        self.detector = Detector(model_path=model_path, device_name=device_name)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, qos_profile_sensor_data
        )

    def image_callback(self, msg):
        start_time = time.time()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        results, display_img = self.detector.predict(cv_image)

        latency_ms = (time.time() - start_time) * 1000
        cv.putText(
            display_img,
            f"Latency: {latency_ms:.1f}ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv.imshow("Robot Vision OpenVINO", display_img)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubNode()
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
