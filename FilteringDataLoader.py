import os

import numpy as np
import scipy
from PIL import Image
from keras.utils.np_utils import to_categorical
from skimage import data
from sklearn.utils import shuffle


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

    return image2


def get_compressed_image(image, final_size):
    compressed_image = scipy.misc.imresize(image, (final_size, final_size))
    return compressed_image


def split_image(picture_window_size, input_image, output_image, patch_step):
    half_window_size = int(picture_window_size / 2)

    patches = []
    labels = []

    for i in range(half_window_size, len(input_image) - 1 - half_window_size, patch_step):
        for j in range(half_window_size, len(input_image[0]) - 1 - half_window_size, patch_step):

            patch = input_image[
                i - half_window_size:i + half_window_size,
                j - half_window_size:j + half_window_size
            ]

            patches.append(patch)

            label_patch = output_image[
                i - half_window_size:i + half_window_size,
                j - half_window_size:j + half_window_size
            ]

            class_id = 1 if np.sum(label_patch) > 0 else 0

            labels.append(to_categorical(class_id, 2)[0])

    s_patches, s_labels = shuffle(patches, labels)
    return np.array(s_patches), np.array(s_labels)


def show_image(image, title='PIRO_image'):
    im = Image.fromarray(image)
    im.show(title)
