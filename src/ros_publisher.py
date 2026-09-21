import cv2
import numpy as np
import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import (
    UInt8MultiArray,
    MultiArrayDimension,
)

from cv_bridge import CvBridge

from detection import (
    CameraSim,
    detect_plate,
    crop_plate,
    detect_paint,
    draw_plate_roi,
)

from calibration_config import load_values


class DetectionPublisher(Node):
    def __init__(
        self,
        source,
    ):
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
            self.process_frame,
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.timer_callback,
        )

    def get_plate_transform(
        self,
        corners,
    ):
        corners = np.asarray(
            corners,
            dtype=np.float32,
        )

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

        inverse_matrix = np.linalg.inv(matrix)

        return (
            matrix,
            inverse_matrix,
            width,
            height,
        )

    def map_regions_to_camera(
        self,
        regions,
        corners,
        margin,
    ):
        (
            _,
            inverse_matrix,
            _,
            _,
        ) = self.get_plate_transform(corners)

        margin = max(
            0,
            int(margin),
        )

        mapped_regions = []

        for region in regions:
            polygon = np.asarray(
                region["polygon"],
                dtype=np.float32,
            )

            if len(polygon) < 3:
                continue

            polygon_warped = polygon.copy()

            polygon_warped[:, 0] += margin
            polygon_warped[:, 1] += margin

            polygon_warped = polygon_warped.reshape(
                -1,
                1,
                2,
            )

            polygon_camera = cv2.perspectiveTransform(
                polygon_warped,
                inverse_matrix,
            )

            polygon_camera = polygon_camera.reshape(
                -1,
                2,
            )

            mapped_region = region.copy()

            mapped_region["polygon"] = polygon_camera

            mapped_regions.append(mapped_region)

        return mapped_regions

    def create_camera_mask(
        self,
        regions,
        image_shape,
    ):
        height, width = image_shape[:2]

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        for region in regions:
            polygon = np.asarray(
                region["polygon"],
                dtype=np.float32,
            )

            if len(polygon) < 3:
                continue

            polygon = np.round(polygon).astype(np.int32)

            polygon[:, 0] = np.clip(
                polygon[:, 0],
                0,
                width - 1,
            )

            polygon[:, 1] = np.clip(
                polygon[:, 1],
                0,
                height - 1,
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255,
            )

        return mask

    def draw_paint_regions(
        self,
        image,
        regions,
    ):
        output = image.copy()

        for index, region in enumerate(regions):
            polygon = np.asarray(
                region["polygon"],
                dtype=np.float32,
            )

            if len(polygon) < 3:
                continue

            polygon = np.round(polygon).astype(np.int32)

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

            cv2.putText(
                output,
                f"{index + 1}: {area:.0f}",
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

    def process_frame(
        self,
        image,
    ):
        params = load_values()

        plate_result = detect_plate(image)

        if not plate_result["found"]:
            output = image.copy()

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

            empty_mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8,
            )

            return (
                output,
                empty_mask,
            )

        corners = plate_result["corners"]

        margin = int(params["plate_margin"])

        plate_roi = crop_plate(
            image,
            corners,
            margin=margin,
        )

        paint_result = detect_paint(
            plate_roi,
            intensity_min=params["intensity_min"],
            intensity_max=params["intensity_max"],
            merge_distance=params["merge_distance"],
            detection_mode=params.get(
                "detection_mode",
                "intensity",
            ),
            contrast_threshold=params.get(
                "contrast_threshold",
                25,
            ),
        )

        camera_regions = self.map_regions_to_camera(
            paint_result["regions"],
            corners,
            margin,
        )

        camera_mask = self.create_camera_mask(
            camera_regions,
            image.shape,
        )

        output = draw_plate_roi(
            image,
            plate_result,
            margin,
        )

        output = self.draw_paint_regions(
            output,
            camera_regions,
        )

        return (
            output,
            camera_mask,
        )

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

        binary = (mask > 0).astype(np.uint8) * 255

        msg = self.bridge.cv2_to_imgmsg(
            binary,
            encoding="mono8",
        )

        msg.header.stamp = stamp

        self.publisher_mask_image.publish(msg)

    def publish_mask(
        self,
    ):
        mask = self.cam.get_mask()

        if mask is None:
            return

        binary = (mask > 0).astype(np.uint8)

        height, width = binary.shape

        msg = UInt8MultiArray()

        msg.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=height,
                stride=(height * width),
            ),
            MultiArrayDimension(
                label="width",
                size=width,
                stride=width,
            ),
        ]

        msg.data = binary.flatten().tolist()

        self.publisher_mask.publish(msg)

    def timer_callback(
        self,
    ):
        stamp = self.get_clock().now().to_msg()

        self.publish_plate_contour(stamp)

        self.publish_cam(stamp)

        self.publish_mask(stamp)

        self.publish_mask_image(stamp)

    def destroy_node(
        self,
    ):
        self.cam.stop()

        super().destroy_node()


def main(
    args=None,
):
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
