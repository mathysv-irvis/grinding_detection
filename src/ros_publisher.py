import numpy as np
import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension

from cv_bridge import CvBridge

from detection import (
    CameraSim,
    filter_intensity_range,
)

from calibration_config import load_values


class DetectionPublisher(Node):
    def __init__(self, source):
        super().__init__("plate_contour_detection")

        self.bridge = CvBridge()

        self.publisher_plate_contour = self.create_publisher(
            Image,
            "/detection/plate_contour",
            10,
        )

        self.publisher_cam = self.create_publisher(
            Image,
            "/detection/cam",
            10,
        )

        self.publisher_mask = self.create_publisher(
            UInt8MultiArray,
            "/detection/mask",
            10,
        )

        self.publisher_mask_image = self.create_publisher(
            Image,
            "/detection/mask_image",
            10,
        )

        self.cam = CameraSim(source)

        self.cam.start(
            self.filter_live,
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.timer_callback,
        )

    def filter_live(
        self,
        image,
        **kwargs,
    ):
        params = load_values()

        result, poly_mask, _ = filter_intensity_range(
            image=image,
            intensity_min=params["intensity_min"],
            intensity_max=params["intensity_max"],
            merge_distance=params["merge_distance"],
        )

        return result, poly_mask

    def publish_cam(
        self,
        stamp,
    ):
        frame = self.cam.get_frame()

        if frame is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8",
        )

        msg.header.stamp = stamp

        self.publisher_cam.publish(msg)

    def publish_plate_contour(
        self,
        stamp,
    ):
        display = self.cam.get_display()

        if display is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            display,
            encoding="bgr8",
        )

        msg.header.stamp = stamp

        self.publisher_plate_contour.publish(msg)

    def publish_mask_image(
        self,
        stamp,
    ):
        mask = self.cam.get_mask()

        if mask is None:
            return

        mask = mask.astype(np.uint16)

        max_label = mask.max()

        if max_label > 0:
            display = (mask.astype(np.float32) * (255.0 / max_label)).astype(np.uint8)
        else:
            display = np.zeros(
                mask.shape,
                dtype=np.uint8,
            )

        msg = self.bridge.cv2_to_imgmsg(
            display,
            encoding="mono8",
        )

        msg.header.stamp = stamp

        self.publisher_mask_image.publish(msg)

    def publish_mask(
        self,
        stamp,
    ):
        mask = self.cam.get_mask()

        if mask is None:
            return

        binary = (mask > 0).astype(np.uint8)

        msg = UInt8MultiArray()

        msg.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=binary.shape[0],
                stride=(binary.shape[0] * binary.shape[1]),
            ),
            MultiArrayDimension(
                label="width",
                size=binary.shape[1],
                stride=binary.shape[1],
            ),
        ]

        msg.data = binary.flatten().tolist()

        self.publisher_mask.publish(msg)

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()

        self.publish_plate_contour(stamp)

        self.publish_cam(stamp)

        self.publish_mask(stamp)

        self.publish_mask_image(stamp)

    def destroy_node(self):
        self.cam.stop()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    source = "./plate_dataset/image_00024.jpg"

    node = DetectionPublisher(source)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
