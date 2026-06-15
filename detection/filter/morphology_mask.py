import cv2
import numpy as np


def morphology_mask(
    image,
    threshold=130,
    kernel_size=7,
    inverse=False,
):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    thresh_type = (
        cv2.THRESH_BINARY_INV
        if inverse
        else cv2.THRESH_BINARY
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        thresh_type,
    )

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

    return 255 - mask


def process_morphology(
    image,
    threshold=130,
    kernel_size=7,
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
