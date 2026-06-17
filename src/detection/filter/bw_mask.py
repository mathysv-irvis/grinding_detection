import cv2
import numpy as np


def bw_mask(
    image,
    threshold=128,
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return mask


def process_bw(
    image,
    threshold=128,
):

    mask = bw_mask(
        image,
        threshold,
    )

    display = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    return display, mask
