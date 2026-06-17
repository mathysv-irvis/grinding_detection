import cv2
import numpy as np


def contour_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, mask = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = np.ones(
        (7, 7),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result = np.zeros_like(mask)

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        cv2.drawContours(
            result,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED,
        )

    return result


def process_contour(image):

    mask = contour_mask(image)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
