
# Grinding Detection

Simple computer vision pipeline for grinding detection supporting both **live cameras** and **simulated inputs** (images/videos).

## Project Structure

```text
.
├── detection/
│   ├── camera.py        # Camera interface
│   ├── detection.py     # Detection pipeline
│   └── filter/          # Image processing filters
│
├── ressources/
└── example_integration.py
```

## Camera Usage

### Live camera

```python
from detection.camera import CameraRunner

camera = CameraRunner()
camera.start(process_frame=my_filter)
```

### Image or video

```python
from detection.camera import CameraSim

camera = CameraSim("ressources/video.mp4")
camera.start(process_frame=my_filter)
```

## Getting the latest frame

The camera runs in a background thread. The latest processed frame can be retrieved at any time:

```python
filtered_frame = camera.get_display()
```

## Display

```python
while True:

    frame = camera.display()

    if frame is not None:
        cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == 27:
        break

camera.stop()
```

## Filters

The `detection/filter/` folder contains reusable image processing modules (background removal, color filtering, morphology, contours, gradients, etc.) that can be combined inside the detection pipeline.

## Example

See `example_integration.py` for a complete integration example.
