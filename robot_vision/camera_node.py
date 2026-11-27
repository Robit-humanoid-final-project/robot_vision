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

from humanoid_interfaces.msg import Vision2LocalObserv
from insta360_usb_cam.msg import PanTiltStatusMsgs
from robot_vision.detector import Detector
from robot_vision.distance_estimator import DistanceEstimator
from robot_vision.hsv import HSV
from robot_vision.vision_message_builder import VisionMessageBuilder
from robot_vision.vision_pipeline import VisionPipeline


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # 설치된 패키지 경로에서 모델 가져오기
        package_share_dir = get_package_share_directory("robot_vision")
        default_model_path = os.path.join(package_share_dir, "model", "best.xml")

        self.declare_parameter("topic_name", "camera1/camera/image_raw")
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device_name", "AUTO")  # //
        topic_name = self.get_parameter("topic_name").value
        model_path = self.get_parameter("model_path").value
        device_name = self.get_parameter("device_name").value

        # 거리 계산
        self.declare_parameter("camera_matrix.data", [0.0] * 9)
        self.declare_parameter("distortion_coefficients.data", [0.0] * 5)
        self.declare_parameter("camera_height", 0.58)  # m
        self.declare_parameter("camera_tilt_deg", 0.0)
        self.declare_parameter("mount_tilt_deg", 0.0)
        self.declare_parameter("ball.width", 0.130)
        self.declare_parameter("ball.height", 0.130)
        self.declare_parameter("cone.width", 0.140)
        self.declare_parameter("cone.height", 0.225)  # //
        k = self.get_parameter("camera_matrix.data").value
        d = self.get_parameter("distortion_coefficients.data").value
        self.distance_estimator = DistanceEstimator(
            camera_matrix=k,
            distortion_coefficients=d,
            camera_height=self.get_parameter("camera_height").value,
            mount_tilt_deg=self.get_parameter("mount_tilt_deg").value,
            ball_width=self.get_parameter("ball.width").value,
            ball_height=self.get_parameter("ball.height").value,
            cone_width=self.get_parameter("cone.width").value,
            cone_height=self.get_parameter("cone.height").value,
        )
        self.tilt_lock = threading.Lock()
        self.tilt_deg = self.get_parameter("camera_tilt_deg").value

        self.bridge = CvBridge()
        # OpenVINO 추론 model
        self.pipeline = VisionPipeline(
            Detector(model_path=model_path, device_name=device_name),
            HSV(),
        )
        self.message_builder = VisionMessageBuilder(self.distance_estimator)
        # worker 구조 (연산 중일 때 프레임 드랍, 실시간성 유지)
        self.latest_frame = None
        self.latest_display = None
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
        self.pan_tilt_subscription = self.create_subscription(
            PanTiltStatusMsgs,
            "/camera1/pan_tilt_status",
            self.pan_tilt_callback,
            camera_qos,
        )
        self.vision_pub = self.create_publisher(
            Vision2LocalObserv, "/vision_packages", 10
        )
        self.timer = self.create_timer(1 / 30, self.display_timer_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            self.publish_status(2)
            return

        with self.frame_lock:
            self.latest_frame = cv_image
        self.frame_event.set()

    def pan_tilt_callback(self, msg: PanTiltStatusMsgs):
        tilt_deg = msg.tilt / 3600.0
        with self.tilt_lock:
            self.tilt_deg = tilt_deg

    def display_timer_callback(self):
        with self.frame_lock:
            display_img = self.latest_display
            self.latest_display = None

        if display_img is not None:
            cv.imshow("Robot Vision", display_img)
            cv.waitKey(1)

    def publish_status(self, status):
        message = Vision2LocalObserv()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera"
        message.status = status
        self.vision_pub.publish(message)

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

            try:
                with self.tilt_lock:
                    tilt_deg = self.tilt_deg
                result = self.pipeline.process(cv_image)
                distance_context = self.distance_estimator.create_context(
                    tilt_deg=tilt_deg,
                    letterbox_scale=result.letterbox_scale,
                    letterbox_pad=result.letterbox_pad,
                )
                message = self.message_builder.build(result, distance_context)
                message.header.stamp = self.get_clock().now().to_msg()
                self.vision_pub.publish(message)
            except (cv.error, RuntimeError, ValueError, TypeError, IndexError) as error:
                self.get_logger().error(f"비전 처리 실패: {error}")
                self.publish_status(3)
                continue

            final_display = cv.cvtColor(result.display_image, cv.COLOR_GRAY2BGR)
            cv.copyTo(result.processed_image, result.mask, final_display)

            for x, y, w, h, obs_id in result.obstacle_bboxes:
                cv.rectangle(final_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(
                    final_display,
                    f"Obstacle:{obs_id}",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            for x, y, w, h in result.goalpost_bboxes:
                cv.rectangle(final_display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv.putText(
                    final_display,
                    "Goalpost(L)",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            with self.frame_lock:
                self.latest_display = final_display

            latency_ms = (time.perf_counter() - start_time) * 1000
            fps = 1000 / latency_ms if latency_ms > 0 else 0.0

            # print(f"total={latency_ms:.1f}ms, fps={fps:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
