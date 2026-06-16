from detection.detection import filter_kmeans_augmented
from detection import CameraSim
import numpy as np
import time
import cv2

def triple_layout(mask, disp, frame):

    min_width  = 800
    max_width  = 1800
    min_height = 400
    max_height = 900

    images = [
        ("mask", mask),
        ("disp", disp),
        ("frame", frame),
    ]

    h = max(img.shape[0] for _, img in images)

    rendered = []

    for name, img in images:

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)

        img = cv2.resize(img, (w, h))

        cv2.putText(
            img,
            name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        rendered.append(img)

    canvas = np.hstack(rendered)

    H, W = canvas.shape[:2]

    scale = 1.0

    scale = min(
        scale,
        max_width / W,
        max_height / H,
    )

    if W * scale < min_width:
        scale = max(scale, min_width / W)

    if H * scale < min_height:
        scale = max(scale, min_height / H)

    new_w = int(W * scale)
    new_h = int(H * scale)

    canvas = cv2.resize(
        canvas,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    return canvas

def process(fname):

    cam = CameraSim(fname)
    cam.start(filter_kmeans_augmented, cluster=1, K=4, max_component=1400)
    time.sleep(1)

    while True:
        mask  = cam.get_mask()
        disp  = cam.get_display()
        frame = cam.get_frame()

        canvas = triple_layout(mask, disp, frame)

        cv2.imshow("Detection", canvas)
        if cv2.waitKey(1) == 27:
            break

    cam.stop()

if __name__ == "__main__":
    img = "./ressources/top_view_depth_camera_camera_image_raw_20260610_174122.png"
    vid = "./ressources/top_view_depth_camera_camera_image_raw_20260610_174122.mp4"
    cam = 0

    path = vid
    process(path)
