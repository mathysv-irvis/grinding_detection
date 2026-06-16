from .filter import process_kmeans, process_lab, process_contrast, process_bw
from .filter import process_morphology, process_largest_component
import cv2

def filter_kmeans_augmented(image, cluster, K, max_component, threshold=130, kernel_size=7):

    _, morph_mask = process_morphology(
        image,
        threshold=threshold,
        kernel_size=kernel_size,
    )

    kmean, mask = process_kmeans(
        image=image,
        roi_mask=morph_mask,
        cluster=cluster,
        K=K,
    )

    disp_postproc, mask_postproc = process_largest_component(mask, threshold=max_component)

    display = cv2.bitwise_and(
        image,
        image,
        mask=mask_postproc,
    )
    return display, mask
