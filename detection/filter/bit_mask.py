import cv2
import numpy as np


def bit_mask(
    image,
    sigma=0.5
):
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    high = _
    low = int(high * 0.5)

    mask = cv2.Canny(
        gray,
        low,
        high,
    )

    return mask


def process_contrast(image):

    mask = bit_mask(image)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )
    return display, mask
