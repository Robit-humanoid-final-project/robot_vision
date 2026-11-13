import cv2 as cv
import numpy as np


def nothing(x):
    pass


class HSV:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.lower_bound = np.array([2, 110, 150])
        self.upper_bound = np.array([30, 255, 255])

        if self.debug_mode:
            cv.namedWindow("HSV Tuner", cv.WINDOW_NORMAL)
            cv.resizeWindow("HSV Tuner", 500, 300)
            cv.createTrackbar("H min", "HSV Tuner", self.lower_bound[0], 179, nothing)
            cv.createTrackbar("S min", "HSV Tuner", self.lower_bound[1], 255, nothing)
            cv.createTrackbar("V min", "HSV Tuner", self.lower_bound[2], 255, nothing)
            cv.createTrackbar("H max", "HSV Tuner", self.upper_bound[0], 179, nothing)
            cv.createTrackbar("S max", "HSV Tuner", self.upper_bound[1], 255, nothing)
            cv.createTrackbar("V max", "HSV Tuner", self.upper_bound[2], 255, nothing)

    def process(self, padded_bgr):
        if self.debug_mode:
            self.lower_bound[0] = cv.getTrackbarPos("H min", "HSV Tuner")
            self.lower_bound[1] = cv.getTrackbarPos("S min", "HSV Tuner")
            self.lower_bound[2] = cv.getTrackbarPos("V min", "HSV Tuner")
            self.upper_bound[0] = cv.getTrackbarPos("H max", "HSV Tuner")
            self.upper_bound[1] = cv.getTrackbarPos("S max", "HSV Tuner")
            self.upper_bound[2] = cv.getTrackbarPos("V max", "HSV Tuner")

        hsv = cv.cvtColor(padded_bgr, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_bound, self.upper_bound)

        # 노이즈 제거
        kernel_close = np.ones((5, 5), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open, iterations=1)

        split_mask = mask.copy()

        contours, _ = cv.findContours(
            split_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            x, y, w, h = cv.boundingRect(c)

            # 매우 작은 노이즈는 skip
            if w < 15 or h < 15 or cv.contourArea(c) < 50:
                continue

            # 2w > h 일 때 겹침 판단
            if w > h * 0.5:
                epsilon = 0.01 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)  # 다각형 근사

                if len(approx) > 3:  # 꼭짓점 3개 이상
                    hull = cv.convexHull(
                        approx, returnPoints=False
                    )  # Convex Hull 모양의 점 index를 반환(ex. 손 모양이면 손가락 끝 점 index)
                    if hull is not None and len(hull) > 3:
                        try:
                            # approx와 hull을 비교해서 파인 부분을 계산
                            defects = cv.convexityDefects(approx, hull)
                            if defects is not None:
                                valid_defects = []
                                for defect_row in defects:
                                    _, _, f, d = defect_row.flatten()
                                    # f(Far point): 깊게 파인(hull과 거리가 먼) 꼭짓점의 인덱스, d: depth
                                    depth = d / 256.0  # Normalization
                                    if depth > h * 0.03 and depth > 3:
                                        valid_defects.append(
                                            tuple(approx[f][0])
                                        )  # (x, y)
                                # 해당 파인 부분이 상단인지, 하단인지
                                top_defects = [
                                    pt for pt in valid_defects if pt[1] < y + h // 2
                                ]
                                bottom_defects = [
                                    pt for pt in valid_defects if pt[1] >= y + h // 2
                                ]

                                if top_defects:
                                    # 너비 기준 중앙에서 가까운 f를 찾음
                                    center_x = x + w // 2
                                    # value = min(iterable, key=...)
                                    # value: key 결과가 가장 작은 값을 반환
                                    best_top = min(
                                        top_defects,
                                        key=lambda pt: abs(pt[0] - center_x),
                                    )
                                    line_thickness = 3 if h > 100 else 2  # 분리선 굵기
                                    # 하단에 파인 부분 인지 시 두 점을 잇는 분리선
                                    if bottom_defects:
                                        best_bottom = min(
                                            bottom_defects,
                                            key=lambda pt: abs(pt[0] - best_top[0]),
                                        )
                                        cv.line(
                                            split_mask,
                                            best_top,
                                            best_bottom,
                                            0,
                                            thickness=line_thickness,
                                        )
                                    # 인지 X, 그냥 수직으로 내림
                                    else:
                                        cv.line(
                                            split_mask,
                                            best_top,
                                            (best_top[0], y + h),
                                            0,
                                            thickness=line_thickness,
                                        )
                        except cv.error:
                            pass

        # 바운딩 박스 추출
        final_contours, _ = cv.findContours(
            split_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        obs_id = 0

        for c in final_contours:
            x, y, w, h = cv.boundingRect(c)

            # 겹쳐있다가 쪼개진 꼬깔 판별
            if w < 7 or h < 10 or cv.contourArea(c) < 50:
                continue

            aspect_ratio = w / float(h)

            # 가로세로 꼬깔 비율인지
            if 0.3 <= aspect_ratio <= 0.7:
                area = cv.contourArea(c)

                # 꼭짓점 개수 검사, 노이즈 잡힌 경우 꼭짓점 다수
                if area >= 150:
                    epsilon = 0.05 * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    if len(approx) > 7:
                        continue

                bboxes.append((x, y, w, h, obs_id))
                obs_id += 1

        return split_mask, bboxes
