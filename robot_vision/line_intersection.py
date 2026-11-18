# test: 사용 제외해보기

import cv2 as cv
import numpy as np


def LineIntersection(line_a, line_b):
    ax1, ay1, ax2, ay2 = line_a
    bx1, by1, bx2, by2 = line_b

    a = np.array([ax1, ay1], dtype=np.float32)
    b = np.array([ax2, ay2], dtype=np.float32)
    c = np.array([bx1, by1], dtype=np.float32)
    d = np.array([bx2, by2], dtype=np.float32)
    direction_a = b - a  # 벡터
    direction_b = d - c
    # 외적, 평행 판정
    cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(cross) < 1e-6:  # 평행한 경우
        return None

    offset = c - a  # 오프셋 벡터
    # t = line A 교차점의 위치, u = line B 교차점의 위치 // 0(start) ~ 1(end)
    t = (offset[0] * direction_b[1] - offset[1] * direction_b[0]) / cross
    u = (offset[0] * direction_a[1] - offset[1] * direction_a[0]) / cross
    # 선분을 살짝 벗어난 경우까지 인정
    if not (-0.15 <= t <= 1.15 and -0.15 <= u <= 1.15):
        return None
    return a + t * direction_a  # == (c + u * dir_b)


def FindIntersection(gray_img, bbox):
    x1, y1, x2, y2 = bbox  # bounding box -> roi 설정
    roi = gray_img[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None
    # canny: edge 검출, weak edge의 경우 strong edge 주변부인지 판단
    edges = cv.Canny(roi, 50, 150, apertureSize=3)
    min_line_length = max(8, min(roi.shape) // 5)
    # Hough 직선 검출
    # HoughLinesP = HoughLines + Probabilistic(확률적)
    # rho = x*cos(theta) + y*sin(theta) (극좌표계 직선의방정식 파생)
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(10, min(roi.shape) // 8),
        minLineLength=min_line_length,
        maxLineGap=max(4, min(roi.shape) // 20),
    )
    if lines is None:
        return None

    best_candidate = None
    best_score = float("-inf")
    roi_center = np.array([roi.shape[1] / 2, roi.shape[0] / 2])
    detected_lines = lines.reshape(-1, 4)

    for index, line_a in enumerate(detected_lines):
        for line_b in detected_lines[index + 1 :]:
            point = LineIntersection(line_a, line_b)  # 교차점
            if point is None:
                continue
            if not (0 <= point[0] < roi.shape[1] and 0 <= point[1] < roi.shape[0]):
                continue  # 교차점이 roi 밖으로 나간 경우
            # 두 점 사이 거리(유클리드 거리) np.linalg.norm()
            length_a = np.linalg.norm(line_a[2:4] - line_a[0:2])
            length_b = np.linalg.norm(line_b[2:4] - line_b[0:2])
            # center로부터의 거리
            distance_from_center = np.linalg.norm(point - roi_center)

            score = length_a + length_b - 0.5 * distance_from_center
            if score > best_score:
                best_score = score
                best_candidate = point

    if best_candidate is None:
        return None
    return tuple(np.rint(best_candidate).astype(int))
