import numpy as np
import DataLoader
import FilteringModel
import ExactModel
import cv2
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
    for i in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
        patches = []
        for j in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
            patch = get_patch(exact_image, i, j, exact_window_size, exact_window_decision_kernel_size)
            patches.append(patch)
        patches = np.array(patches)
        res = exact_model.model.predict(patches)
        for j in range(0, len(exact_image) - exact_window_decision_kernel_size + 1, 2):
            single_res = res[j]
            if single_res[1] > 0.5 or single_res[0] > 0.5:
                dec = 1 if single_res[1] > res[0] else 0
            else:
                dec = 0
            for x in range(exact_window_decision_kernel_size):
                for y in range(exact_window_decision_kernel_size):
                    exact_result[x + i][y + j] = dec

    im_gray = exact_result * 255

    #
    # ret, thresh = cv2.threshold(im_gray.astype(np.uint8), 127, 255, 0)
    # nb_edges, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # size = stats[1:, -1]
    # result = thresh
    # for e in range(0, nb_edges - 1):
    #     if size[e] >= 20:
    #         th_up = e + 2
    #         th_do = th_up
    #
    #         mask = cv2.inRange(output, th_do, th_up)
    #         result = cv2.bitwise_xor(result, mask)

    # im_gray = dilation(im_gray)
    # ret, thresh = cv2.threshold(im_gray.astype(np.uint8), 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours {0}".format(len(contours)))
    # cv2.drawContours(im_gray, contours, -1, (0, 255, 0), 3)
    return im_gray
