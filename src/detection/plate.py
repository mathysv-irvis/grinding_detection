import cv2
import numpy as np


DEFAULT_HSV_MIN = (
    0,
    40,
    40,
)

DEFAULT_HSV_MAX = (
    30,
    255,
    255,
)


def create_plate_mask(
    image,
    hsv_min=DEFAULT_HSV_MIN,
    hsv_max=DEFAULT_HSV_MAX,
):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lower = np.array(
        hsv_min,
        dtype=np.uint8,
    )

    upper = np.array(
        hsv_max,
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper,
    )

    return mask


def find_plate_contour(
    mask,
    min_area=1000,
):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area
    ]

    if not valid_contours:
        return None

    return max(
        valid_contours,
        key=cv2.contourArea,
    )


def order_corners(
    points,
):
    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.shape != (4, 2):
        raise ValueError("Expected four points")

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points[:, 0] + points[:, 1]

    differences = points[:, 0] - points[:, 1]

    ordered[0] = points[np.argmin(sums)]

    ordered[2] = points[np.argmax(sums)]

    ordered[1] = points[np.argmax(differences)]

    ordered[3] = points[np.argmin(differences)]

    return ordered


def contour_to_corners(
    contour,
):
    perimeter = cv2.arcLength(
        contour,
        True,
    )

    for epsilon_ratio in (
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ):
        epsilon = epsilon_ratio * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        if len(polygon) == 4:
            points = polygon.reshape(
                4,
                2,
            )

            return order_corners(points)

    rectangle = cv2.minAreaRect(contour)

    points = cv2.boxPoints(rectangle)

    return order_corners(points)


def detect_plate(
    image,
    hsv_min=DEFAULT_HSV_MIN,
    hsv_max=DEFAULT_HSV_MAX,
    min_area=1000,
):
    mask = create_plate_mask(
        image,
        hsv_min=hsv_min,
        hsv_max=hsv_max,
    )

    contour = find_plate_contour(
        mask,
        min_area=min_area,
    )

    if contour is None:
        return {
            "found": False,
            "mask": mask,
            "corners": None,
            "contour": None,
        }

    corners = contour_to_corners(contour)

    return {
        "found": True,
        "mask": mask,
        "corners": corners,
        "contour": contour,
    }


def crop_plate(
    image,
    corners,
    margin=0,
):
    corners = order_corners(corners)

    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    width_top = np.linalg.norm(top_right - top_left)

    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    height_left = np.linalg.norm(bottom_left - top_left)

    height_right = np.linalg.norm(bottom_right - top_right)

    width = max(
        int(width_top),
        int(width_bottom),
        1,
    )

    height = max(
        int(height_left),
        int(height_right),
        1,
    )

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination,
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
    )

    margin = max(
        0,
        int(margin),
    )

    if margin == 0:
        return warped

    if warped.shape[0] <= 2 * margin or warped.shape[1] <= 2 * margin:
        return warped

    return warped[
        margin:-margin,
        margin:-margin,
    ]
