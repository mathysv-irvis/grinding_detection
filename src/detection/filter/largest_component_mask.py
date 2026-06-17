import cv2
import numpy as np


def largest_component_mask(
    image,
    threshold=130,
):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image,
        connectivity=8,
    )

    result = np.zeros_like(image)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= threshold:

            result[labels == i] = 255

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
