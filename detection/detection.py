from .filter import process_kmeans, process_lab, process_contrast, process_bw

def filter_kmeans_augmented(image, cluster):

    lab, mask = process_bw(image.copy(), 130)
    kmeans, mask = process_kmeans(lab, cluster)

    return kmeans, None
