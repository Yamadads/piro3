import os, sys
from skimage import data, io
from PIL import Image
import numpy as np
import random
from sklearn.feature_extraction import image
import scipy


def get_images_names_list(images_path, labels_path):
    names_list = os.listdir(images_path)
    full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
                       names_list]
    return full_path_names


def get_image(path):
    needed_channels = 3
    image = data.imread(path, False)
    # if len(image.shape) != needed_channels:
    #     return_image = np.resize(image, (image.shape[0], image.shape[1], 1))
    #     return return_image
    return image


def split_image(picture_window_size, center_size, input_image, output_image, set_size):
    half_window_size = int(picture_window_size / 2)
    half_center_size = int(center_size / 2)
    center_start_pos = half_window_size - half_center_size
    patches = []
    labels = []
    for x in range(set_size):
        i = random.randint(0, len(input_image) - picture_window_size)
        j = random.randint(0, len(input_image) - picture_window_size)
        patch = input_image[i:i + picture_window_size, j:j + picture_window_size]
        center = output_image[
                 i + center_start_pos:i + center_start_pos + center_size,
                 j + center_start_pos:j + center_start_pos + center_size]
        label_value = 1.0 if np.average(center) > 0.5 else 0.0
        label = np.resize(label_value, (1, 1, 1))
        patches.append(patch)
        labels.append(label)
    return np.array(patches), np.array(labels)


def get_compressed_image(image, final_size):
    zoom = final_size/len(image)
    compressed_image = scipy.misc.imresize(image, (final_size, final_size))
    #compressed_image = scipy.ndimage.interpolation.zoom(image, zoom, order=3, mode='constant', cval=0.0, prefilter=True)
    return compressed_image


def show_image(image):
    im = Image.fromarray(image)
    im.show()
