import concurrent.futures
import os
import time

import cv2 as cv
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from robot_vision.detector import Detector
from robot_vision.hsv import HSV


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

        self.bridge = CvBridge()
        # OpenVINO 추론 model
        self.detector = Detector(model_path=model_path, device_name=device_name)
        # HSV
        self.hsv = HSV()
        # 병렬 처리
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, qos_profile_sensor_data
        )

    def letterbox_resize(self, img, target_w=640, target_h=640):
        orig_h, orig_w = img.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        # padding
        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded

    def image_callback(self, msg):
        start_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        # resize, padding
        padded_bgr = self.letterbox_resize(cv_image, self.detector.w, self.detector.h)

        # 병렬
        future_yolo = self.thread_pool.submit(self.detector.predict, padded_bgr)
        future_hsv = self.thread_pool.submit(self.hsv.process, padded_bgr)

        # yolo_gray_img는 바운딩 박스가 쳐진 1채널 흑백 이미지(640x640)
        results, yolo_gray_img = future_yolo.result()
        mask, color_extracted = future_hsv.result()

        # 흑백 이미지를 컬러(3채널) 캔버스로 확장
        final_display = cv.cvtColor(yolo_gray_img, cv.COLOR_GRAY2BGR)

        # mask 영역에 해당하는 픽셀만 color_extracted의 픽셀로 덮어씌움
        cv.copyTo(color_extracted, mask, final_display)

        time_diff = time.time() - start_time
        latency_ms = time_diff * 1000
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        cv.putText(
            final_display,
            f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv.imshow("Robot Vision (Fast Mode)", final_display)
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
