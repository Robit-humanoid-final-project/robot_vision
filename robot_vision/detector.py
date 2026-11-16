import cv2 as cv
import numpy as np
from openvino import Core

from robot_vision.line_intersection import FindIntersection


class Detector:
    def __init__(self, model_path: str, device_name: str = "AUTO"):
        self.core = Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(
            model=self.model, device_name=device_name
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        _, _, self.h, self.w = self.input_layer.shape

        self.model = self.core.read_model(model=model_path)

        self.class_names = {
            0: "Ball",
            1: "Robot",
            2: "L-Line",
            3: "T-Line",
            4: "X-Line",
            5: "Goalpost",
        }

        self.conf_thresholds = {0: 0.57, 1: 0.65, 2: 0.35, 3: 0.50, 4: 0.45, 5: 0.99}
        self.detections = []
        self.intersections = []

    def preprocess(self, padded_bgr: np.ndarray):
        # 그레이스케일
        gray = cv.cvtColor(padded_bgr, cv.COLOR_BGR2GRAY)

        # NCHW 변환 (1, 1, 640, 640)
        expanded_img = np.expand_dims(gray, axis=-1)
        tensor = expanded_img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return np.expand_dims(tensor, axis=0), gray

    def postprocess(
        self, results: np.ndarray, display_img: np.ndarray, conf_threshold=0.3
    ):
        """
        Shape가 (1, 300, 6)인 모델의 출력을 시각화합니다.
        6개의 요소: [x1, y1, x2, y2, confidence, class_id]
        """
        # Batch 차원을 제거하여 (300, 6) 배열로 만듦
        predictions = results[0]

        self.detections = []
        self.intersections = []
        analysis_img = cv.cvtColor(display_img, cv.COLOR_BGR2GRAY)

        for pred in predictions:
            x1, y1, x2, y2, conf, class_id = pred

            # 신뢰도(Confidence) 필터링
            threshold = self.conf_thresholds.get(class_id, conf_threshold)

            # 개별 신뢰도 조건 검사
            if conf > threshold:
                if x2 <= 1.0 and y2 <= 1.0:
                    x1 *= self.w
                    x2 *= self.w
                    y1 *= self.h
                    y2 *= self.h

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(class_id)
                x1 = max(0, min(x1, display_img.shape[1] - 1))
                y1 = max(0, min(y1, display_img.shape[0] - 1))
                x2 = max(0, min(x2, display_img.shape[1]))
                y2 = max(0, min(y2, display_img.shape[0]))
                if x2 <= x1 or y2 <= y1:  # 박스 유효성 검사
                    continue

                if class_id in (2, 3, 4):  # L, T, X 인식한 경우 해당 부분 교차점 분석
                    intersection = FindIntersection(analysis_img, (x1, y1, x2, y2))
                    if intersection is not None:
                        local_x, local_y = intersection
                        point = (x1 + local_x, y1 + local_y)
                        self.intersections.append(
                            {
                                "class_id": class_id,
                                "point": point,
                                "confidence": float(conf),
                            }
                        )
                        cv.circle(display_img, point, 6, (0, 0, 0), -1)
                        cv.circle(display_img, point, 3, (255, 255, 255), -1)
                else:
                    self.detections.append(
                        {
                            "class_id": class_id,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(conf),
                        }
                    )

                # 바운딩 박스
                cv.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

                label = f"ID:{class_id} ({conf:.2f})"
                cv.putText(
                    display_img,
                    label,
                    (x1, max(15, y1 - 10)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return display_img

    def predict(self, img: np.ndarray):
        # 1. 전처리
        input_tensor, display_img = self.preprocess(img)
        # 2. 추론
        results = self.compiled_model([input_tensor])[self.output_layer]
        # 3. 후처리 및 시각화
        display_img = self.postprocess(results, display_img)

        return results, display_img
