import os
import random

import scipy
import numpy as np
from PIL import Image
from skimage import data


def get_images_names_list(images_path, labels_path):
    names_list = os.listdir(images_path)
    full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
                       names_list]
    return full_path_names


def get_image(path, compress_to_size):
    needed_channels = 3
    image = data.imread(path, False)

    # image2 = cv.resize(image, (compress_to_size, compress_to_size))
    image2 = get_compressed_image(image, compress_to_size)

    if len(image2.shape) != needed_channels:
        return_image = np.resize(image2, (image.shape[0], image.shape[1], 1))
        image2 = return_image

    # show_image(image2)

    return image2

def get_compressed_image(image, final_size):
    compressed_image = scipy.misc.imresize(image, (final_size, final_size))
    return compressed_image

def split_image(picture_window_size, input_image, output_image, set_size):
    half_window_size = int(picture_window_size/2)
    patches = []
    labels = []

    for x in range(set_size):
        i = random.randint(half_window_size, len(input_image) - 1 - half_window_size)
        j = random.randint(half_window_size, len(input_image[0]) - 1 - half_window_size)

        patch = input_image[
                i - half_window_size:i + half_window_size,
                j - half_window_size:j + half_window_size
        ]

        label_patch = output_image[
                i - half_window_size:i + half_window_size,
                j - half_window_size:j + half_window_size
        ]

        label = np.resize([1 if np.sum(label_patch) > 0 else 0], (1, 1, 1))
        patches.append(patch)
        labels.append(label)

    return np.array(patches), np.array(labels)


def show_image(image):
    im = Image.fromarray(image)
    im.show()
