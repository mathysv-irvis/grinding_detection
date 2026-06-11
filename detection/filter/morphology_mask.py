import cv2
import numpy as np


def morphology_mask(
    image,
    threshold=130,
    kernel_size=10,
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

    kernel = np.ones(
        (kernel_size, kernel_size),
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


def process_morphology(
    image,
    threshold=130,
    kernel_size=7,
):

    mask = morphology_mask(
        image,
        threshold,
        kernel_size,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
