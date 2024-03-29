import os
from skimage import data
from PIL import Image
import numpy as np
import random
import scipy
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical


def get_images_names_list(images_path, labels_path):
    names_list = os.listdir(images_path)
    full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
                       names_list]
    return full_path_names


def get_image(path):
    image = data.imread(path, False)
    return image


def split_image_exact(picture_window_size, center_size, input_image, output_image, set_size):
    half_window_size = int(picture_window_size / 2)
    half_center_size = int(center_size / 2)
    center_start_pos = half_window_size - half_center_size
    patches = []
    labels = []
    for i in range(picture_window_size, len(output_image) - picture_window_size, center_size):
        for j in range(picture_window_size, len(output_image) - picture_window_size, center_size):
            if np.average(output_image[i:i + center_size, j + center_size]) > 127.5:
                patch = input_image[i - half_window_size:i - half_window_size + picture_window_size,
                        j - half_window_size:j - half_window_size + picture_window_size]
                patches.append(patch)
                label = 1
                labels.append(to_categorical(label, 2)[0])
            if len(labels) >= int(set_size / 2):
                break

    for x in range(len(labels)):
        patch_good = False
        while not patch_good:
            i = random.randint(0, len(input_image) - picture_window_size)
            j = random.randint(0, len(input_image) - picture_window_size)
            if np.average(output_image[i:i + center_size, j + center_size]) < 127.5:
                patch_good = True
                patch = input_image[i:i + picture_window_size, j:j + picture_window_size]
                patches.append(patch)
                label = 0
                labels.append(to_categorical(label, 2)[0])
    s_patches, s_labels = shuffle(patches, labels)
    return np.array(s_patches), np.array(s_labels)


def split_image_filter(picture_window_size, center_size, input_image, output_image, set_size):
    half_window_size = int(picture_window_size / 2)
    half_center_size = int(center_size / 2)
    half_set_size = int(set_size / 2)
    center_start_pos = half_window_size - half_center_size
    true_patches = []
    false_patches = []
    true_labels = []
    false_labels = []

    while (len(true_patches) < half_set_size or len(false_patches) < half_set_size):
        i = random.randint(0, len(input_image) - picture_window_size)
        j = random.randint(0, len(input_image) - picture_window_size)
        patch = input_image[i:i + picture_window_size, j:j + picture_window_size]
        label_patch = output_image[i:i + picture_window_size, j:j + picture_window_size]
        if np.sum(label_patch) > 1:
            true_patches.append(patch)
            label = 1
            true_labels.append(to_categorical(label, 2)[0])
        else:
            false_patches.append(patch)
            label = 0
            false_labels.append(to_categorical(label, 2)[0])
    result_patches, result_labels = shuffle(
        true_patches[:half_set_size] + false_patches[:half_set_size]
        , true_labels[:half_set_size] + false_labels[:half_set_size]
    )

    return np.array(result_patches), np.array(result_labels)


def get_compressed_image(image, final_size):
    compressed_image = scipy.misc.imresize(image, (final_size, final_size))
    return compressed_image


def show_image(image):
    im = Image.fromarray(image)
    im.show()


def save_image(image, path):
    im = Image.fromarray(image)
    im.save(path)
