from detection.detection import filter_kmeans_augmented
from detection import CameraSim

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

        self.publisher = self.create_publisher(
            Image,
            "/detection/plate_contour",
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

    def timer_callback(self):
        mask  = self.cam.get_mask()
        disp  = self.cam.get_display()
        frame = self.cam.get_frame()

        if mask is None or disp is None or frame is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            disp,
            encoding="bgr8",
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

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
