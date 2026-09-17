import cv2
import numpy as np


def kmeans_mask(
    image,
    roi_mask,
    intensity=40,
    K=5,
):
    pixels = image[roi_mask > 0]

    if pixels.size == 0:
        return np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
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

    labels = labels.flatten()

    cluster_intensities = np.mean(
        centers,
        axis=1,
    )

    selected_cluster = int(np.argmin(np.abs(cluster_intensities - intensity)))

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    mask[roi_mask > 0] = (labels == selected_cluster).astype(np.uint8) * 255

    return mask


def process_kmeans(
    image,
    roi_mask,
    intensity=40,
    K=5,
):
    mask = kmeans_mask(
        image=image,
        roi_mask=roi_mask,
        intensity=intensity,
        K=K,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
