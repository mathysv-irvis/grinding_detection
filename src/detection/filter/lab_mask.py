import cv2
import numpy as np


def lab_mask(
    image,
    b_low=120,
    b_high=170,
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    _, _, B = cv2.split(lab)

    mask = cv2.inRange(
        B,
        b_low,
        b_high,
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def process_lab(
    image,
    b_low=150,
    b_high=160,
):

    mask = lab_mask(
        image,
        b_low,
        b_high,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
