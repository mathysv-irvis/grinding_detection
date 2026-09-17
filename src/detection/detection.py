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

    intensity_min = int(np.clip(intensity_min, 0, 255))

    intensity_max = int(np.clip(intensity_max, 0, 255))

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


def merge_regions(
    mask,
    merge_distance,
):
    merge_distance = max(
        0,
        int(merge_distance),
    )

    if merge_distance == 0:
        return mask

    kernel_size = (merge_distance * 2) + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )


def extract_label_polygons(
    labels,
    num_labels,
):
    polygons = []

    for label_id in range(
        1,
        num_labels,
    ):
        mask = (labels == label_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 1:
                continue

            epsilon = 0.01 * cv2.arcLength(
                cnt,
                True,
            )

            approx = cv2.approxPolyDP(
                cnt,
                epsilon,
                True,
            )

            polygons.append(
                {
                    "label": label_id,
                    "area": area,
                    "polygon": approx.reshape(
                        -1,
                        2,
                    ),
                }
            )

    return polygons


def polygons_mask(
    polygons,
    image_shape,
):
    h, w = image_shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint16,
    )

    for obj in polygons:
        poly = obj["polygon"].astype(np.int32)

        label = int(obj["label"])

        cv2.fillPoly(
            mask,
            [poly],
            label,
        )

    return mask


def polygon_frame(
    image,
    polygons,
    thickness=2,
):
    output = image.copy()

    for i, obj in enumerate(polygons):
        poly = obj["polygon"].astype(np.int32)

        rng = np.random.default_rng(i)

        color = tuple(
            int(c)
            for c in rng.integers(
                50,
                255,
                size=3,
            )
        )

        cv2.polylines(
            output,
            [poly],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return output


def filter_intensity_range(
    image,
    intensity_min,
    intensity_max,
    merge_distance,
):
    intensity_mask = create_intensity_mask(
        image,
        intensity_min,
        intensity_max,
    )

    merged_mask = merge_regions(
        intensity_mask,
        merge_distance,
    )

    num_labels, labels = cv2.connectedComponents(
        merged_mask,
    )

    polygons = extract_label_polygons(
        labels,
        num_labels,
    )

    poly_mask = polygons_mask(
        polygons,
        image.shape,
    )

    overlay = polygon_frame(
        image,
        polygons,
    )

    result = cv2.addWeighted(
        image,
        0.3,
        overlay,
        0.7,
        0,
    )

    return (
        result,
        poly_mask,
        intensity_mask,
    )
