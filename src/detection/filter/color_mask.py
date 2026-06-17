import cv2
import numpy as np

def bgr_to_hsv(color):
    color = np.uint8([[color]])
    return cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0]

def color_mask(image, bgr_color, h_tol=30, s_tol=250, v_tol=250):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    target = bgr_to_hsv(bgr_color)

    lower = np.array([
        max(0, target[0] - h_tol),
        max(0, target[1] - s_tol),
        max(170, target[2] - v_tol),
    ], dtype=np.uint8)

    upper = np.array([
        min(179, target[0] + h_tol),
        min(20, target[1] + s_tol),
        min(255, target[2] + v_tol),
    ], dtype=np.uint8)

    return cv2.inRange(hsv, lower, upper)

def process_color(image, bgr_color, h_tol=10, s_tol=50, v_tol=50):

    mask = color_mask(image, bgr_color, h_tol, s_tol, v_tol)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
