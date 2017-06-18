import os

import numpy as np
import scipy
import scipy.ndimage
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


def split_image(picture_window_size, input_images, output_images, patch_step):
    half_window_size = int(picture_window_size / 2)

    positive_patches = []
    negative_patches = []

    positive_labels = []
    negative_labels = []

    for input_image, output_image in zip(input_images, output_images):
        for i in range(half_window_size, len(input_image) - 1 - half_window_size, patch_step):
            for j in range(half_window_size, len(input_image[0]) - 1 - half_window_size, patch_step):

                patch = input_image[
                    i - half_window_size:i + half_window_size,
                    j - half_window_size:j + half_window_size
                ]

                label_patch = output_image[
                    i - half_window_size:i + half_window_size,
                    j - half_window_size:j + half_window_size
                ]

                class_id = 1 if np.sum(label_patch) > 0 else 0

                if class_id == 1:
                    positive_patches.append(patch)
                    positive_labels.append(to_categorical(class_id, 2)[0])
                else:
                    negative_patches.append(patch)
                    negative_labels.append(to_categorical(class_id, 2)[0])

    s_p_patches, s_p_labels = shuffle(positive_patches, positive_labels)
    s_n_patches, s_n_labels = shuffle(negative_patches, negative_labels)

    min_size = min([len(s_p_patches), len(s_n_patches)])

    target_size = min(4 * min_size, len(s_n_patches) if min_size == len(s_p_patches) else len(s_p_patches))

    if min_size == len(s_p_patches):
        generate_new_examples(s_p_patches, s_p_labels, target_size)
    else:
        generate_new_examples(s_n_patches, s_n_labels, target_size)

    print('removing {0} examples from {1} examples, rest count = {2}'.format(
        len(s_n_patches) - target_size if min_size == len(s_p_patches) else len(s_p_patches) - target_size
        , 'non-roads' if min_size == len(s_p_patches) else 'roads'
        , target_size))

    result_patches, result_labels = shuffle(
        s_p_patches[:target_size] + s_n_patches[:target_size]
        , s_p_labels[:target_size] + s_n_labels[:target_size]
    )

    return np.array(result_patches), np.array(result_labels)


def generate_new_examples(images, labels, target_size):
    i = 0
    rotate_step = len(images)

    while len(images) != target_size:
        current_angle = 90 * ((i / rotate_step) + 1)
        images.append(scipy.ndimage.rotate(images[i % rotate_step], current_angle, reshape=False))
        labels.append(labels[i % rotate_step])

        # show_image(images[i % rotate_step])
        # show_image(images[-1], 'ee')

        i += 1


def show_image(image, title='PIRO_image'):
    im = Image.fromarray(image)
    im.show(title)
