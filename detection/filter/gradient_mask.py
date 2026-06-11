import cv2
import numpy as np


def gradient_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    grad = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    profile = np.abs(grad).sum(axis=0)

    left = np.argmax(profile[:len(profile)//2])
    right = np.argmax(profile[len(profile)//2:]) + len(profile)//2

    mask = np.zeros(
        gray.shape,
        dtype=np.uint8,
    )

    mask[:, left:right] = 255

    return mask


def process_gradient(image):

    mask = gradient_mask(image)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
