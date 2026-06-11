from .filter import process_kmeans, process_lab, process_contrast, process_bw
from .filter import process_morphology, process_largest_component

def filter_kmeans_augmented(image, cluster):

    lab, mask = process_bw(image.copy(), 130)
    # kmeans, mask = process_kmeans(lab, cluster)
    morph, _ = process_morphology(lab)
    # larg, _ = process_largest_component(lab)


    return morph, None
