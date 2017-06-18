import numpy as np
import DataLoader
import FilteringModel
import ExactModel
import cv2
from  skimage.filters import median
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, opening


def get_patch(input_image, i, j, window_size, decision_size, ):
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


def roads(image):
    # model
    exact_model_name = "exact_net_1"
    exact_model_weights_name = "exact_net_1_weights_22"
    exact_window_size = 20
    exact_window_decision_kernel_size = 2

    filtering_model_name = "filter_net_1"
    filtering_model_weights_name = "filter_net_1_weights_1"
    filtering_window_size = 20

    filtering_image = DataLoader.get_compressed_image(image, 200)
    filtering_image = filtering_image / 255

    exact_image = DataLoader.get_compressed_image(image, 600)
    exact_image = exact_image / 255

    # filter_model = FilteringModel.FilteringModel()
    # filter_model.load_model(filtering_model_name, filtering_model_weights_name)
    # filter_result = np.zeros((len(filtering_image), len(filtering_image[0])))

    exact_model = ExactModel.Model()
    exact_model.init_model()
    exact_model.load_weights(exact_model_weights_name)
    exact_result = np.zeros((len(exact_image), len(exact_image)), dtype=int)

    print('predict')
    patches = []
    for i in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
        for j in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
            patch = get_patch(exact_image, i, j, exact_window_size, exact_window_decision_kernel_size)
            patches.append(patch)
    patches = np.array(patches)
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
