from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "grinding_detection" / "src")
)

from detection.detection import filter_kmeans_augmented
from detection import CameraSim

from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.node import Node
import rclpy

import numpy as np
import cv2
import time


class DetectionPublisher(Node):

    def __init__(self, source):
        super().__init__("plate_contour_detection")

        self.bridge    = CvBridge()

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
            "detection/mask",
            10,
        )

        self.cam = CameraSim(source)
        self.cam.start(
            filter_kmeans_augmented,
            cluster = 1,
            K       = 4,
            max_component = 1400,
        )

        time.sleep(1)

        self.timer = self.create_timer(
            1 / 30,
            self.timer_callback,
        )

    def publish_cam(self, stamp):
        frame = self.cam.get_frame()
        if frame is None:
            return

        msg_frame = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8",
        )
        msg_frame.header.stamp = stamp
        self.publisher_cam.publish(msg_frame)

        return

    def publish_plate_contour(self, stamp):
        disp = self.cam.get_display()
        if disp is None:
            return

        msg_disp = self.bridge.cv2_to_imgmsg(
            disp,
            encoding="bgr8",
        )
        msg_disp.header.stamp = stamp
        self.publisher_plate_contour.publish(msg_disp)

        return

    def publish_mask(self, stamp):
        mask = self.cam.get_mask()
        if mask is None:
            return

        binary = (mask > 0).astype(np.uint8)

        msg_mask = UInt8MultiArray()
        msg_mask.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=binary.shape[0],
                stride=binary.shape[0] * binary.shape[1],
            ),
            MultiArrayDimension(
                label="width",
                size=binary.shape[1],
                stride=binary.shape[1],
            ),
        ]
        msg_mask.data = binary.flatten().tolist()

        self.publisher_mask.publish(msg_mask)

        return

    def timer_callback(self):


        stamp = self.get_clock().now().to_msg()

        self.publish_plate_contour(stamp)
        self.publish_cam(stamp)
        self.publish_mask(stamp)

    def destroy_node(self):
        self.cam.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init()

    source = 0

    node = DetectionPublisher(source)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
