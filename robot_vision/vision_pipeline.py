from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass
class VisionResult:
    processed_image: np.ndarray
    display_image: np.ndarray
    mask: np.ndarray
    detections: list
    intersections: list
    obstacle_bboxes: list
    goalpost_bboxes: list
    letterbox_scale: float
    letterbox_pad: tuple


class VisionPipeline:
    def __init__(self, detector, hsv):
        self.detector = detector
        self.hsv = hsv

    def letterbox_resize(self, image, target_w=640, target_h=640):
        orig_h, orig_w = image.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        # padding
        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, scale, (left, top)

    def process(self, image):
        processed_image, scale, pad = self.letterbox_resize(
            image, self.detector.w, self.detector.h
        )
        _, display_image = self.detector.predict(processed_image)
        ball_bboxes = [
            detection["bbox"]
            for detection in self.detector.detections
            if detection["class_id"] == 0
        ]
        mask, obstacle_bboxes, goalpost_bboxes = self.hsv.process(
            processed_image,
            excluded_bboxes=ball_bboxes,
        )

        return VisionResult(
            processed_image=processed_image,
            display_image=display_image,
            mask=mask,
            detections=self.detector.detections,
            intersections=self.detector.intersections,
            obstacle_bboxes=obstacle_bboxes,
            goalpost_bboxes=goalpost_bboxes,
            letterbox_scale=scale,
            letterbox_pad=pad,
        )
