import os, sys
import glob
from skimage import data, io
from PIL import Image


def get_images_names_list(images_path, labels_path):
    names_list = os.listdir(images_path)
    full_path_names = [[os.path.splitext(x)[0], images_path + x, labels_path + os.path.splitext(x)[0] + '.tif'] for x in
                       names_list]
    return full_path_names


def get_image(path):
    image = data.imread(path)
    return image


def show_image(image):
    im = Image.fromarray(image)
    im.show()
