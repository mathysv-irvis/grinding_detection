import cv2
from abc import ABC, abstractmethod
from pathlib import Path


class BaseCamera(ABC):

    def __init__(
        self,
        window_name="Camera",
        exit_keys=(27, ord("q")),
    ):
        self.window_name = window_name
        self.exit_keys = exit_keys

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def read(self):
        """
        Returns
        -------
        ok : bool
        frame : np.ndarray | None
        """
        pass

    @abstractmethod
    def release(self):
        pass

    def run(
        self,
        process_frame=None,
        *args,
        **kwargs
    ):

        self.open()

        try:

            while True:

                ok, frame = self.read()

                if not ok:
                    break

                if process_frame is not None:
                    display, detection_mask = process_frame(
                        frame,
                        *args,
                        **kwargs
                    )
                else:
                    display, detection_mask = frame.copy(), None



                if display is None:
                    display = frame

                cv2.imshow(
                    self.window_name,
                    display
                )

                key = cv2.waitKey(1)

                if key in self.exit_keys:
                    break

        finally:

            self.release()
            cv2.destroyAllWindows()


class CameraRunner(BaseCamera):

    def __init__(
        self,
        camera_index=0,
        window_name="Camera",
        exit_keys=(27, ord("q")),
    ):

        super().__init__(
            window_name=window_name,
            exit_keys=exit_keys,
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
        exit_keys=(27, ord("q")),
        loop_video=True,
    ):

        super().__init__(
            window_name=window_name,
            exit_keys=exit_keys,
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

        if not isinstance(self.is_image, int):
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
