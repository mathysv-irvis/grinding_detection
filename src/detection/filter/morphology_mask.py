import cv2
import numpy as np


def morphology_mask(
    image,
    threshold=100,
    kernel_size=1,
    inverse=False,
):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    if inverse:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        threshold_type,
    )

    if kernel_size > 1:
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
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
    threshold=100,
    kernel_size=1,
    inverse=False,
):
    mask = morphology_mask(
        image=image,
        threshold=threshold,
        kernel_size=kernel_size,
        inverse=inverse,
    )

    if len(image.shape) == 3:
        display = cv2.bitwise_and(
            image,
            image,
            mask=mask,
        )
    else:
        display = mask.copy()

    return display, mask
