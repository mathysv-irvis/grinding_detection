import cv2
from abc import ABC, abstractmethod
from pathlib import Path
import threading

class BaseCamera(ABC):

    def __init__(
        self,
        window_name="Camera",
        exit_keys=(27, ord("q")),
    ):

        self.window_name = window_name
        self.exit_keys = exit_keys

        self.running = False
        self.thread = None

        self.frame = None
        self.display = None
        self.mask = None

        self.lock = threading.Lock()

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self):
        pass

    def _worker(
        self,
        process_frame=None,
        *args,
        **kwargs,
    ):

        self.open()

        try:

            while self.running:

                ok, frame = self.read()

                if not ok:
                    break

                if process_frame is not None:
                    display, mask = process_frame(
                        frame,
                        *args,
                        **kwargs,
                    )
                else:
                    display = frame.copy()
                    mask = None

                if display is None:
                    display = frame

                with self.lock:

                    self.frame   = frame
                    self.display = display
                    self.mask    = mask

        finally:

            self.release()
            self.running = False

    def start(
        self,
        process_frame=None,
        *args,
        **kwargs,
    ):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            args=(process_frame, *args),
            kwargs=kwargs,
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def get_display(self):

        with self.lock:

            if self.display is None:
                return None

            return self.display.copy()

    def get_mask(self):

        with self.lock:

            if self.mask is None:
                return None

            return self.mask.copy()


        if window_name is None:
            window_name = self.window_name

    def run(
        self,
        process_frame=None,
        *args,
        **kwargs,
    ):

        self.start(
            process_frame=process_frame,
            *args,
            **kwargs,
        )

        try:

            while self.running:

                display = self.get_display()

        finally:

            self.stop()
            cv2.destroyAllWindows()

class CameraRunner(BaseCamera):

    def __init__(
        self,
        camera_index=0,
        window_name="Camera",
    ):

        super().__init__(
            window_name=window_name,
        )

        self.camera_index = camera_index
        self.cap = None

    def open(self):

        self.cap = cv2.VideoCapture(
            self.camera_index
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_index}"
            )

    def read(self):

        return self.cap.read()

    def release(self):

        if self.cap is not None:
            self.cap.release()


class CameraSim(BaseCamera):

    def __init__(
        self,
        source,
        window_name="CameraSim",
        loop_video=True,
    ):

        super().__init__(
            window_name=window_name,
        )

        if isinstance(source, str):
            self.source = Path(source)
            if not self.source.exists():
                raise FileNotFoundError(self.source)

        else:
            self.source = int(source)

        self.loop_video = loop_video
        self.cap = None
        self.image = None
        self.is_image = False
        self.is_video = False

    def open(self):

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        video_extensions = {
            ".mp4",
        }

        if not isinstance(self.source, int):
            self.is_image = (
                self.source.suffix.lower() in image_extensions
            )

            self.is_video = (
                self.source.suffix.lower() in video_extensions
            )

        if self.is_image:

            self.image = cv2.imread(
                str(self.source)
            )

            if self.image is None:
                raise RuntimeError(
                    f"Cannot read image {self.source}"
                )

        elif self.is_video:

            self.cap = cv2.VideoCapture(
                str(self.source)
            )

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open video {self.source}"
                )

        else:

            self.cap = cv2.VideoCapture(
                int(self.source)
            )

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.source}"
                )

    def read(self):

        if self.is_image:
            return True, self.image.copy()

        ok, frame = self.cap.read()

        if ok:
            return True, frame

        if self.loop_video:

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            return self.cap.read()

        return False, None

    def release(self):

        if self.cap is not None:
            self.cap.release()
