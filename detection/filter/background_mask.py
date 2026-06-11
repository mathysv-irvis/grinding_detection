import cv2
import numpy as np


def background_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = cv2.inRange(
        gray,
        140,
        220,
    )

    kernel = np.ones(
        (15, 15),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.bitwise_not(mask)

    return mask


def process_background(image):

    mask = background_mask(image)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
