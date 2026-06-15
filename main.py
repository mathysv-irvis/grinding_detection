from detection import CameraSim, process_contrast, process_color, process_gradient, process_background, process_contour, process_lab, process_kmeans


MEDIA_PATH = "./ressources/top_view_depth_camera_camera_image_raw_20260610_174122"
# MEDIA_PATH = "./ressources/Video_20260108_165858_967"
# MEDIA_PATH = "./ressources/Image_20260108_165858_954"
# MEDIA_PATH = "./ressources/Image_20260108_165859_364"
# MEDIA_PATH = "./ressources/Video_20260108_165859_488"
CAMERA_VID = CameraSim(MEDIA_PATH+".mp4")
# CAMERA_PIC = CameraSim(MEDIA_PATH+".jpeg")

def filter_contrast():

    CAMERA_VID.run(process_contrast)

def filter_color():

    paint_color = (175, 105, 63)
    CAMERA_VID.run(process_color, bgr_color=paint_color)

def filter_gradient():

    CAMERA_VID.run(
        process_gradient,
    )


def filter_background():

    CAMERA_VID.run(
        process_background,
    )


def filter_contour():

    CAMERA_VID.run(
        process_contour,
    )


def filter_lab():

    CAMERA_VID.run(
        process_lab,
    )

def filter_kmeans():

    CAMERA_VID.run(
        process_kmeans,
        cluster=5
    )

def augmented_kmean():
    from detection.detection import filter_kmeans_augmented
    
    CAMERA_VID.run(
        filter_kmeans_augmented,
        cluster=1
    )

if __name__ == "__main__":

    # CAMERA_PIC.run()
    # filter_contrast()
    # filter_color()
    # filter_gradient()
    # filter_background()
    # filter_contour()
    # filter_lab()
    # filter_kmeans()

    augmented_kmean()
