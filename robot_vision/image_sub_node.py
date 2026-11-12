import os
import threading
import time

import cv2 as cv
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
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
        # worker 구조 (연산 중일 때 프레임 드랍, 실시간성 유지)
        self.latest_frame = None
        self.frame_lock = threading.Lock()  # mutex
        self.frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self.process_latest_frame, daemon=True
        )
        self.worker_thread.start()
        # QoS profile keep last, depth 1
        camera_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, camera_qos
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        with self.frame_lock:
            self.latest_frame = cv_image
        self.frame_event.set()

    def process_latest_frame(self):
        while not self.stop_event.is_set():
            if not self.frame_event.wait(timeout=0.1):
                continue

            with self.frame_lock:
                cv_image = self.latest_frame
                self.latest_frame = None
                self.frame_event.clear()

            if cv_image is None:
                continue

            start_time = time.perf_counter()

            processed_bgr = self.letterbox_resize(
                cv_image, self.detector.w, self.detector.h
            )
            _, yolo_gray_img = self.detector.predict(processed_bgr)
            mask, color_extracted = self.hsv.process(processed_bgr)

            final_display = cv.cvtColor(yolo_gray_img, cv.COLOR_GRAY2BGR)
            cv.copyTo(color_extracted, mask, final_display)

            cv.imshow("Robot Vision", final_display)
            cv.waitKey(1)

            latency_ms = (time.perf_counter() - start_time) * 1000
            fps = 1000 / latency_ms if latency_ms > 0 else 0.0

            print(f"total={latency_ms:.1f}ms, fps={fps:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        node.frame_event.set()
        node.worker_thread.join(timeout=2.0)
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
