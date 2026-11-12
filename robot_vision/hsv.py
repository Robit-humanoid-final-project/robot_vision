import cv2 as cv
import numpy as np


def nothing(x):
    pass


class HSV:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.lower_bound = np.array([3, 110, 110])
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

        # HSV 마스킹
        hsv = cv.cvtColor(padded_bgr, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_bound, self.upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        color_extracted = cv.bitwise_and(padded_bgr, padded_bgr, mask=mask)
        return mask, color_extracted
