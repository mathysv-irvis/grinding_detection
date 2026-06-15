from .filter import process_kmeans, process_lab, process_contrast, process_bw
from .filter import process_morphology, process_largest_component
import cv2

def filter_kmeans_augmented(image, cluster):

    _, morph_mask = process_morphology(
        image,
        threshold=130,
        kernel_size=7,
    )

    kmean, mask = process_kmeans(
        image=image,
        roi_mask=morph_mask,
        cluster=cluster,
        K=4,
    )

    disp_postproc, mask_postproc = process_largest_component(mask, threshold=1400)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask_postproc,
    )
    return display, mask
