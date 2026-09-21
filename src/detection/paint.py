import cv2
import numpy as np


def create_intensity_mask(
    image,
    intensity_min,
    intensity_max,
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    intensity_min = int(
        np.clip(
            intensity_min,
            0,
            255,
        )
    )

    intensity_max = int(
        np.clip(
            intensity_max,
            0,
            255,
        )
    )

    if intensity_min > intensity_max:
        intensity_min, intensity_max = (
            intensity_max,
            intensity_min,
        )

    mask = cv2.inRange(
        gray,
        intensity_min,
        intensity_max,
    )

    return mask


def create_contrast_mask(
    image,
    contrast_threshold,
    blur_size=21,
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blur_size = int(blur_size)

    if blur_size < 3:
        blur_size = 3

    if blur_size % 2 == 0:
        blur_size += 1

    background = cv2.GaussianBlur(
        gray,
        (
            blur_size,
            blur_size,
        ),
        0,
    )

    contrast = cv2.absdiff(
        gray,
        background,
    )

    contrast_threshold = int(
        np.clip(
            contrast_threshold,
            0,
            255,
        )
    )

    mask = cv2.threshold(
        contrast,
        contrast_threshold,
        255,
        cv2.THRESH_BINARY,
    )[1]

    return mask


def merge_regions(
    mask,
    merge_distance,
):
    merge_distance = max(
        0,
        int(merge_distance),
    )

    if merge_distance == 0:
        return mask.copy()

    kernel_size = (2 * merge_distance) + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    merged = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return merged


def extract_paint_regions(
    mask,
):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    regions = []

    for label_id in range(
        1,
        num_labels,
    ):
        area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        if area <= 0:
            continue

        component_mask = (labels == label_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area <= 0:
                continue

            epsilon = 0.01 * cv2.arcLength(
                contour,
                True,
            )

            polygon = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            regions.append(
                {
                    "label": int(label_id),
                    "area": float(contour_area),
                    "polygon": polygon.reshape(
                        -1,
                        2,
                    ),
                }
            )

    return regions


def polygons_mask(
    regions,
    image_shape,
):
    height, width = image_shape[:2]

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint16,
    )

    for region in regions:
        polygon = region["polygon"].astype(np.int32)

        label = int(region["label"])

        cv2.fillPoly(
            mask,
            [polygon],
            label,
        )

    return mask


def detect_paint(
    image,
    intensity_min,
    intensity_max,
    merge_distance,
    detection_mode="intensity",
    contrast_threshold=25,
):
    if detection_mode == "contrast":
        raw_mask = create_contrast_mask(
            image,
            contrast_threshold,
            blur_size=21,
        )

    else:
        raw_mask = create_intensity_mask(
            image,
            intensity_min,
            intensity_max,
        )

    merged_mask = merge_regions(
        raw_mask,
        merge_distance,
    )

    regions = extract_paint_regions(merged_mask)

    polygon_mask = polygons_mask(
        regions,
        image.shape,
    )

    return {
        "raw_mask": raw_mask,
        "merged_mask": merged_mask,
        "regions": regions,
        "polygon_mask": polygon_mask,
        "detection_mode": detection_mode,
    }
