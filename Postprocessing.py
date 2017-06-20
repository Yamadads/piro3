import numpy as np
import cv2
import scipy
import DataLoader


def process_image(image):
    image *= 255
    im = np.array(image, np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 150
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    kernel = np.ones((3, 3), np.uint8)
    img3 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    img4 = scipy.signal.medfilt(img3, kernel_size=3)
    final_image = DataLoader.get_compressed_image(img4, 1500)
    return final_image
