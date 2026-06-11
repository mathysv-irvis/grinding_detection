import cv2
import numpy as np


def kmeans_mask(
    image,
    cluster=1,
    K=5
):

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(
        image.shape[:2]
    )

    mask = (
        labels == cluster
    ).astype(np.uint8) * 255

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


def process_kmeans(
    image,
    cluster=1,
    K=3,
):

    mask = kmeans_mask(
        image,
        cluster,
        K,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
