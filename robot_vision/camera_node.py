import math
import os
import threading
import time

import cv2 as cv
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from humanoid_interfaces.msg import Vision2LocalObserv, VisionDetection, VisionObstacle
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from sensor_msgs.msg import Image

from insta360_usb_cam.msg import PanTiltStatusMsgs
from robot_vision.detector import Detector
from robot_vision.hsv import HSV


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # 설치된 패키지 경로에서 모델 가져오기
        package_share_dir = get_package_share_directory("robot_vision")
        default_model_path = os.path.join(package_share_dir, "model", "best.xml")

        self.declare_parameter("topic_name", "camera1/camera/image_raw")
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device_name", "AUTO")

        topic_name = self.get_parameter("topic_name").value
        model_path = self.get_parameter("model_path").value
        device_name = self.get_parameter("device_name").value

        # 거리 계산
        self.declare_parameter("camera_matrix.data", [0.0] * 9)
        self.declare_parameter("camera_height", 0.7)  # m
        self.declare_parameter("camera_tilt_deg", 0.0)
        self.declare_parameter("mount_tilt_deg", 0.0)

        k = self.get_parameter("camera_matrix.data").value
        self.fx = k[0]
        self.cx = k[2]
        self.fy = k[4]
        self.cy = k[5]
        self.h = self.get_parameter("camera_height").value
        self.mount_tilt_deg = self.get_parameter("mount_tilt_deg").value
        self.theta = math.radians(
            self.get_parameter("camera_tilt_deg").value + self.mount_tilt_deg
        )

        self.bridge = CvBridge()
        # OpenVINO 추론 model
        self.detector = Detector(model_path=model_path, device_name=device_name)
        # HSV
        self.hsv = HSV()
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
            Vision2LocalObserv, "/vision2local_observ", 10
        )
        self.timer = self.create_timer(1 / 30, self.display_timer_callback)

    def pan_tilt_callback(self, msg: PanTiltStatusMsgs):
        tilt_deg = msg.tilt / 3600.0
        self.theta = math.radians(-tilt_deg + self.mount_tilt_deg)

    def compute_distance(self, bbox=None, point=None, roi_offset=(0, 0)):
        if (bbox is None) == (point is None):
            raise ValueError()

        offset_x, offset_y = roi_offset

        if point is not None:
            u = float(point[0]) + offset_x
            v = float(point[1]) + offset_y
        else:
            x1, y1, x2, y2 = bbox
            # 장애물, 공, 골대는 bbox 최하단 중앙 픽셀 사용
            u = ((x1 + x2) / 2.0) + offset_x
            v = float(y2) + offset_y

        # 정규화 좌표(y_n) 및 수직 각도 계산
        y_n = (v - self.cy) / self.fy
        alpha = math.atan(y_n)
        total_angle = self.theta + alpha

        # 분모 0 방지
        if total_angle <= 1e-6:
            return None, None

        distance_x = self.h / math.tan(total_angle)
        distance_y = distance_x * ((self.cx - u) / self.fx)

        return distance_x, distance_y

        # 사용 예시:
        # self.compute_distance(bbox=(260, 180, 340, 310))
        # self.compute_distance(point=(320, 280))

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
            mask, bboxes = self.hsv.process(processed_bgr)
            self.publish_vision_observation(bboxes)

            final_display = cv.cvtColor(yolo_gray_img, cv.COLOR_GRAY2BGR)
            cv.copyTo(processed_bgr, mask, final_display)

            for x, y, w, h, obs_id in bboxes:
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

            with self.frame_lock:
                self.latest_display = final_display

            latency_ms = (time.perf_counter() - start_time) * 1000
            fps = 1000 / latency_ms if latency_ms > 0 else 0.0

            print(f"total={latency_ms:.1f}ms, fps={fps:.1f}")

    def publish_vision_observation(self, obstacle_bboxes):
        message = Vision2LocalObserv()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera"
        message.status = 0

        detection_id = 0
        for detection in self.detector.detections:
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            detection_id += 1

            if class_id == 0:
                distance = self.compute_distance(bbox=detection["bbox"])
                if distance[0] is not None:
                    message.ball = [float(distance[0]), float(distance[1]), confidence]
                    print(
                        f"Ball: x={distance[0]:.3f} m, "
                        f"y={distance[1]:.3f} m, confidence={confidence:.3f}"
                    )
            elif class_id == 5:
                distance = self.compute_distance(bbox=detection["bbox"])
                if distance[0] is None:
                    continue
                detection_message = VisionDetection()
                detection_message.type = 3
                detection_message.detection_id = detection_id
                detection_message.x = float(distance[0])
                detection_message.y = float(distance[1])
                detection_message.confidence = confidence
                message.detections.append(detection_message)
                print(
                    f"Goalpost: x={distance[0]:.3f} m, "
                    f"y={distance[1]:.3f} m, confidence={confidence:.3f}"
                )

        for intersection in self.detector.intersections:
            distance = self.compute_distance(point=intersection["point"])
            if distance[0] is None:
                continue
            detection_id += 1
            detection_message = VisionDetection()
            detection_message.type = intersection["class_id"] - 2
            detection_message.detection_id = detection_id
            detection_message.x = float(distance[0])
            detection_message.y = float(distance[1])
            detection_message.confidence = intersection["confidence"]
            message.detections.append(detection_message)
            print(
                f"Line type={detection_message.type}: "
                f"x={distance[0]:.3f} m, y={distance[1]:.3f} m, "
                f"confidence={intersection['confidence']:.3f}"
            )

        for x, y, width, height, obstacle_id in obstacle_bboxes:
            distance = self.compute_distance(bbox=(x, y, x + width, y + height))
            if distance[0] is None:
                continue
            obstacle = VisionObstacle()
            obstacle.detection_id = obstacle_id
            obstacle.x = float(distance[0])
            obstacle.y = float(distance[1])
            obstacle.confidence = 1.0
            message.obstacles.append(obstacle)
            print(
                f"Obstacle id={obstacle_id}: x={distance[0]:.3f} m, "
                f"y={distance[1]:.3f} m"
            )

        if not message.detections and not any(message.ball) and not message.obstacles:
            message.status = 1
        self.vision_pub.publish(message)

    def display_timer_callback(self):
        with self.frame_lock:
            display_img = self.latest_display
            self.latest_display = None

        if display_img is not None:
            cv.imshow("Robot Vision", display_img)
            cv.waitKey(1)


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
