import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class DistanceContext:
    theta: float
    letterbox_scale: float
    letterbox_pad: tuple


class DistanceEstimator:
    def __init__(
        self,
        camera_matrix,
        distortion_coefficients,
        camera_height,
        mount_tilt_deg,
        ball_width,
        ball_height,
        cone_width,
        cone_height,
    ):
        self.fx = camera_matrix[0]
        self.cx = camera_matrix[2]
        self.fy = camera_matrix[4]
        self.cy = camera_matrix[5]
        self.K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.D = np.array(distortion_coefficients, dtype=np.float64)
        self.camera_height = camera_height
        self.mount_tilt_deg = mount_tilt_deg
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.cone_width = cone_width
        self.cone_height = cone_height

    def create_context(self, tilt_deg, letterbox_scale, letterbox_pad):
        return DistanceContext(
            theta=math.radians(-tilt_deg + self.mount_tilt_deg),
            letterbox_scale=letterbox_scale,
            letterbox_pad=letterbox_pad,
        )

    def _undistort_points(self, points):
        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        undist = cv.undistortPoints(pts, self.K, self.D, P=self.K)
        return undist.reshape(-1, 2)

    def compute_distance_A(
        self, bbox, width, context, height=None, roi_offset=(0, 0)
    ):
        x1, y1, x2, y2 = bbox
        offset_x, offset_y = roi_offset
        pad_x, pad_y = context.letterbox_pad

        x1_orig = (x1 + offset_x - pad_x) / context.letterbox_scale
        y1_orig = (y1 + offset_y - pad_y) / context.letterbox_scale
        x2_orig = (x2 + offset_x - pad_x) / context.letterbox_scale
        y2_orig = (y2 + offset_y - pad_y) / context.letterbox_scale

        (ux1, uy1), (ux2, uy2) = self._undistort_points(
            [(x1_orig, y1_orig), (x2_orig, y2_orig)]
        )
        u = (ux1 + ux2) / 2.0
        v = (uy1 + uy2) / 2.0
        w_orig = ux2 - ux1
        h_orig = uy2 - uy1

        if w_orig <= 1e-6 or (height is not None and h_orig <= 1e-6):
            return None, None

        z_x = (self.fx * width) / w_orig
        if height is not None:
            z_y = (self.fy * height) / h_orig
            z_cam = 0.5 * (z_x + z_y)
        else:
            z_cam = z_x

        y_n = (v - self.cy) / self.fy
        alpha = math.atan(y_n)
        total_angle = context.theta + alpha

        distance_x = z_cam * (math.cos(total_angle) / math.cos(alpha))
        distance_y = z_cam * ((self.cx - u) / self.fx)

        return distance_x, distance_y

    # B 오차 이유: total angle이 0에 가까우면 오차 커짐 -> 기본 theta 설정으로 오차 최소화
    def compute_distance_B(
        self, context, bbox=None, point=None, roi_offset=(0, 0)
    ):
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

        # 640x480 좌표
        pad_x, pad_y = context.letterbox_pad
        u = (u - pad_x) / context.letterbox_scale
        v = (v - pad_y) / context.letterbox_scale

        (u, v) = self._undistort_points([(u, v)])[0]

        # 정규화 좌표(y_n) 및 수직 각도 계산
        y_n = (v - self.cy) / self.fy
        alpha = math.atan(y_n)
        total_angle = context.theta + alpha

        if total_angle <= 1e-6:
            return None, None

        distance_x = self.camera_height / math.tan(total_angle)
        distance_y = distance_x * ((self.cx - u) / self.fx)

        return distance_x, distance_y
