from .filter import process_kmeans, process_lab, process_contrast, process_bw
from .filter import process_morphology, process_largest_component

def filter_kmeans_augmented(image, cluster):

    _, morph_mask = process_morphology(
        image,
        threshold=130,
        kernel_size=7,
    )

    display, mask = process_kmeans(
        image=image,          # ORIGINAL IMAGE
        roi_mask=morph_mask,  # Morphology ROI
        cluster=cluster,
        K=4,
    )

    return display, mask
