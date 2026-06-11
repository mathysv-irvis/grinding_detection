import cv2
import numpy as np


def largest_component_mask(
    image,
    threshold=130,
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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    result = np.zeros_like(mask)

    if num_labels > 1:

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        result[labels == largest] = 255

    return result


def process_largest_component(
    image,
    threshold=130,
):

    mask = largest_component_mask(
        image,
        threshold,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
