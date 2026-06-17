
# Grinding Detection

Simple computer vision pipeline for grinding detection supporting **live cameras**, **images**, and **videos**.

The library provides:

* A threaded camera interface (`CameraSim`)
* A configurable image processing pipeline
* Access to the original image, processed visualization, and binary detection mask
* A ROS2 publisher example

---

# Project Structure

```text
.
├── example_integration.py
├── README.md
├── ressources/
│
└── src/
    ├── detection/
    │   ├── camera.py
    │   ├── detection.py
    │   └── filter/
    │
    └── ros_publisher.py
```

---

# Camera Usage

## Image

```python
from detection import CameraSim
from detection.detection import filter_kmeans_augmented

cam = CameraSim("ressources/image.png")

cam.start(
    filter_kmeans_augmented,
    cluster=1,
    K=4,
    max_component=1400,
)
```

---

## Video

```python
cam = CameraSim("ressources/video.mp4")

cam.start(
    filter_kmeans_augmented,
    cluster=1,
    K=4,
    max_component=1400,
)
```

---

# Retrieving Results

The processing runs in a background thread.

At any time:

```python
frame = cam.get_frame()      # Original image
display = cam.get_display()  # Colored visualization
mask = cam.get_mask()        # Binary mask
```

The mask contains the detected grinding area.

---

# Display Example

```python
while True:

    frame = cam.get_frame()
    display = cam.get_display()
    mask = cam.get_mask()

    if display is not None:
        cv2.imshow("Detection", display)

    if cv2.waitKey(1) == 27:
        break

cam.stop()
```

---

# Filters

The `src/detection/filter/` folder contains reusable image processing modules:

* Background removal
* Color filtering
* KMeans segmentation
* Morphological operations
* Largest connected component extraction
* Contour extraction
* Gradient filtering
* LAB filtering

These filters are combined inside `filter_kmeans_augmented()` to generate the final detection mask.

---

# ROS2 Publisher

`src/ros_publisher.py` publishes the detection results as ROS2 topics at approximately **30 Hz**.

Published topics:

| Topic                      | Type                       | Description                 |
| -------------------------- | -------------------------- | --------------------------- |
| `/detection/cam`           | `sensor_msgs/Image`        | Original camera frame       |
| `/detection/plate_contour` | `sensor_msgs/Image`        | Detection visualization     |
| `/detection/mask`          | `std_msgs/UInt8MultiArray` | Binary detection mask (0/1) |

The binary mask is published as a flattened matrix with its dimensions stored in the `layout` field.

---

# Example

The complete usage example is available in:

```text
example_integration.py
```

The ROS2 integration example is available in:

```text
src/ros_publisher.py
```
