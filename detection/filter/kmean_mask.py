import cv2
import numpy as np


def kmeans_mask(
    image,
    roi_mask,
    cluster=0,
    K=3,
    kernel_size=5,
):

    # Extract only ROI pixels
    pixels = image[roi_mask == 255]

    if pixels.size == 0:
        return np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

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

    labels = labels.flatten()

    # -------------------------------
    # Sort clusters by brightness
    # so cluster indices stay stable
    # -------------------------------

    order = np.argsort(
        centers.mean(axis=1)
    )

    new_labels = np.zeros_like(labels)

    for new_id, old_id in enumerate(order):
        new_labels[labels == old_id] = new_id

    labels = new_labels

    # -------------------------------
    # Reconstruct full-size mask
    # -------------------------------

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    mask[roi_mask == 255] = (
        labels == cluster
    ).astype(np.uint8) * 255

    # -------------------------------
    # Clean mask
    # -------------------------------

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


def process_kmeans(
    image,
    roi_mask,
    cluster=0,
    K=3,
    kernel_size=5,
):

    mask = kmeans_mask(
        image=image,
        roi_mask=roi_mask,
        cluster=cluster,
        K=K,
        kernel_size=kernel_size,
    )

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return display, mask
