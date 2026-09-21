import cv2
import numpy as np


def draw_plate(
    image,
    plate_result,
):
    output = image.copy()

    if not plate_result["found"]:
        cv2.putText(
            output,
            "PLATE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    corners = plate_result["corners"]

    if corners is None:
        return output

    corners = np.asarray(
        corners,
        dtype=np.int32,
    )

    cv2.polylines(
        output,
        [corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    for index, point in enumerate(corners):
        x, y = point

        cv2.circle(
            output,
            (int(x), int(y)),
            5,
            (255, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            str(index + 1),
            (
                int(x) + 8,
                int(y) - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_plate_roi(
    image,
    plate_result,
    margin,
):
    output = draw_plate(
        image,
        plate_result,
    )

    if not plate_result["found"]:
        return output

    corners = plate_result["corners"]

    if corners is None:
        return output

    corners = np.asarray(
        corners,
        dtype=np.float32,
    )

    center = corners.mean(axis=0)

    width = max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    )

    height = max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    )

    if width <= 0 or height <= 0:
        return output

    margin_x = min(
        float(margin),
        width / 2.0,
    )

    margin_y = min(
        float(margin),
        height / 2.0,
    )

    scale_x = (width - 2.0 * margin_x) / width

    scale_y = (height - 2.0 * margin_y) / height

    roi_corners = corners - center

    roi_corners[:, 0] *= scale_x
    roi_corners[:, 1] *= scale_y

    roi_corners += center

    roi_corners = roi_corners.astype(np.int32)

    cv2.polylines(
        output,
        [roi_corners],
        isClosed=True,
        color=(255, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return output


def draw_paint_regions(
    image,
    regions,
):
    output = image.copy()

    for index, region in enumerate(regions):
        polygon = np.asarray(
            region["polygon"],
            dtype=np.int32,
        )

        if len(polygon) < 3:
            continue

        cv2.polylines(
            output,
            [polygon],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        moments = cv2.moments(polygon)

        if moments["m00"] == 0:
            continue

        center_x = int(moments["m10"] / moments["m00"])

        center_y = int(moments["m01"] / moments["m00"])

        area = region.get(
            "area",
            0,
        )

        text = f"{index + 1}: {area:.0f}"

        cv2.putText(
            output,
            text,
            (
                center_x,
                center_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_paint_mask(
    image,
    mask,
):
    if mask is None:
        return image.copy()

    if len(mask.shape) != 2:
        raise ValueError("Paint mask must be a single-channel image.")

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    output = image.copy()

    overlay = np.zeros_like(image)

    overlay[:, :, 2] = mask

    output = cv2.addWeighted(
        output,
        0.7,
        overlay,
        0.3,
        0,
    )

    return output


def create_debug_view(
    image,
    plate_result,
    paint_result,
    margin,
):
    output = draw_plate_roi(
        image,
        plate_result,
        margin,
    )

    if paint_result is not None:
        output = draw_paint_regions(
            output,
            paint_result.get(
                "regions",
                [],
            ),
        )

    return output
