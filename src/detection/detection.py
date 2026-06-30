from .filter import process_kmeans, process_lab, process_contrast, process_bw
from .filter import process_morphology, process_largest_component
import numpy as np
import cv2

def extract_label_polygons(labels, num_labels):
    polygons = []

    for label_id in range(1, num_labels):
        mask = (labels == label_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            polygons.append({
                "label": label_id,
                "polygon": approx.reshape(-1, 2)
            })

    return polygons

def polygons_mask(polygons, image_shape):
    h, w = image_shape[:2]

    mask = np.zeros((h, w), dtype=np.uint16)

    for obj in polygons:
        poly = obj["polygon"].astype(np.int32)
        label = int(obj["label"])

        cv2.fillPoly(mask, [poly], label)

    return mask

def polygon_frame(image, polygons, thickness=2):
    output = image.copy()

    for i, obj in enumerate(polygons):
        poly = obj["polygon"].astype(np.int32)

        # Deterministic color based on polygon index
        rng = np.random.default_rng(i)
        color = tuple(int(c) for c in rng.integers(50, 255, size=3))

        cv2.polylines(
            output,
            [poly],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return output

def filter_kmeans_augmented(image, cluster, K, max_component, threshold=130, kernel_size=7):

    _, morph_mask = process_morphology(
        image,
        threshold=threshold,
        kernel_size=kernel_size,
    )

    kmean, mask = process_kmeans(
        image=image,
        roi_mask=morph_mask,
        cluster=cluster,
        K=K,
    )

    disp_postproc, mask_postproc = process_largest_component(mask, threshold=max_component)


    display = cv2.bitwise_and(
        image,
        image,
        mask=mask_postproc,
    )

    num_labels, labels = cv2.connectedComponents(mask_postproc)

    polygons = extract_label_polygons(labels, num_labels)

    poly_mask = polygons_mask(polygons, image.shape)

    overlay = polygon_frame(display, polygons)
    result = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)
    return result, poly_mask
