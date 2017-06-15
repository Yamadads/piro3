import os, sys
from skimage import data, io
from PIL import Image
import numpy as np


def get_images_names_list(images_path, labels_path):
    names_list = os.listdir(images_path)
    full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
                       names_list]
    return full_path_names


def get_image(path):
    needed_channels = 3
    image = data.imread(path, False)
    if len(image.shape) != needed_channels:
        return_image = np.resize(image, (image.shape[0], image.shape[1], 1))
        return return_image
    return image


def show_image(image):
    im = Image.fromarray(image)
    im.show()
