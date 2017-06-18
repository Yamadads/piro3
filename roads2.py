import numpy as np
import DataLoader
import FilteringModel
import ExactModel
import cv2
from  skimage.filters import median
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, opening


def get_exact_patch(input_image, i, j, window_size, decision_size, ):
    window_size_half = int(window_size / 2)
    decision_size_half = int(decision_size / 2)
    patch = np.ones((window_size, window_size, 3))
    p_x = -1
    for x in range(i - (window_size_half - decision_size_half),
                   i - (window_size_half - decision_size_half) + window_size):
        p_y = -1
        p_x += 1
        for y in range(j - (window_size_half - decision_size_half),
                       j - (window_size_half - decision_size_half) + window_size):
            p_y += 1
            if (x >= 0 and x < len(input_image)) and (y >= 0 and y < len(input_image)):
                patch[p_x][p_y][0] = input_image[x][y][0]
                patch[p_x][p_y][1] = input_image[x][y][1]
                patch[p_x][p_y][2] = input_image[x][y][2]
    arr_patch = patch
    return arr_patch


def get_filter_patches(image, filter_window_size):
    patches = []
    for i in range(0, len(image) - filter_window_size +1, filter_window_size):
        for j in range(0, len(image) - filter_window_size +1, filter_window_size):
            patch = image[i:i + filter_window_size, j:j + filter_window_size]
            patches.append(patch)
    return np.array(patches)


def get_exact_patches(image, exact_window_size, exact_window_decision_kernel_size):
    patches = []
    for i in range(0, len(image) - exact_window_decision_kernel_size + 1, 2):
        for j in range(0, len(image) - exact_window_decision_kernel_size + 1, 2):
            patch = get_exact_patch(image, i, j, exact_window_size, exact_window_decision_kernel_size)
            patches.append(patch)
    patches = np.array(patches)
    return patches

def get_filter_result(compressed_image):
    filtering_model_weights_name = "filter_net_1_weights_1"
    filtering_window_size = 60

    filter_model = FilteringModel.FilteringModel()
    filter_model.init_model()
    filter_model.load_weights(filtering_model_weights_name)
    filter_result = np.zeros((len(compressed_image), len(compressed_image[0])))

    filter_patches = get_filter_patches(compressed_image, filtering_window_size)
    results = filter_model.model.predict(filter_patches)

    idx = -1
    for i in range(0, len(compressed_image) - filtering_window_size +1, filtering_window_size):
        for j in range(0, len(compressed_image) - filtering_window_size +1, filtering_window_size):
            idx += 1
            if results[idx][0] == 0 and results[idx][1] == 1:
                for x in range(60):
                    for y in range(60):
                        filter_result[x+i][y+j] = 1.0

    return filter_result

def roads(image):
    # model
    exact_model_name = "exact_net_1"
    exact_model_weights_name = "exact_net_1_weights_22"
    exact_window_size = 20
    exact_window_decision_kernel_size = 2

    exact_image = DataLoader.get_compressed_image(image, 600)
    exact_image = exact_image / 255


    filtering_image = DataLoader.get_compressed_image(image, 600)
    filtering_image = filtering_image / 255

    filtered_image = get_filter_result(filtering_image)
    to_show_filtered_image = filtered_image / 255
    DataLoader.show_image(to_show_filtered_image)

    exact_model = ExactModel.Model()
    exact_model.init_model()
    exact_model.load_weights(exact_model_weights_name)
    exact_result = np.zeros((len(exact_image), len(exact_image)), dtype=int)

    print('predict')
    patches = get_exact_patches(exact_image, exact_window_size, exact_window_decision_kernel_size)
    res = exact_model.model.predict(patches)
    idx = -1
    for i in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
        for j in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
            idx += 1
            single_res = res[idx]
            if single_res[1] > 0.5 or single_res[0] > 0.5:
                dec = 1 if single_res[1] > single_res[0] else 0
            else:
                dec = 0
            for x in range(exact_window_decision_kernel_size):
                for y in range(exact_window_decision_kernel_size):
                    exact_result[x + i][y + j] = dec
    im_gray = exact_result * 255
    return im_gray
