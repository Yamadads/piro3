import numpy as np
import DataLoader
import Model
import Postprocessing

def get_exact_patch(input_image, i, j, window_size, decision_size):
    window_size_half = int(window_size / 2)
    decision_size_half = int(decision_size / 2)
    win_h_min_dec_h = window_size_half - decision_size_half
    if (i > win_h_min_dec_h) and (j > win_h_min_dec_h) and (i < len(input_image) - window_size_half) and (
                j < len(input_image) - window_size_half):
        return input_image[
               i - win_h_min_dec_h: i - win_h_min_dec_h + window_size,
               j - win_h_min_dec_h: j - win_h_min_dec_h + window_size]

    patch = np.ones((window_size, window_size, 3))
    p_x = -1
    for x in range(i - (win_h_min_dec_h),
                   i - (win_h_min_dec_h) + window_size):
        p_y = -1
        p_x += 1
        for y in range(j - win_h_min_dec_h,
                       j - win_h_min_dec_h + window_size):
            p_y += 1
            if (x >= 0 and x < len(input_image)) and (y >= 0 and y < len(input_image)):
                patch[p_x][p_y][0] = input_image[x][y][0]
                patch[p_x][p_y][1] = input_image[x][y][1]
                patch[p_x][p_y][2] = input_image[x][y][2]
    arr_patch = patch
    return arr_patch


def get_filter_patches(image, filter_window_size):
    patches = []
    for i in range(0, len(image) - filter_window_size + 1, filter_window_size):
        for j in range(0, len(image) - filter_window_size + 1, filter_window_size):
            patch = image[i:i + filter_window_size, j:j + filter_window_size]
            patches.append(patch)
    return np.array(patches)


def get_exact_patches(image, exact_window_size, exact_window_decision_kernel_size, filter):
    patches = []
    for i in range(0, len(image) - exact_window_decision_kernel_size + 1, 2):
        for j in range(0, len(image) - exact_window_decision_kernel_size + 1, 2):
            if filter[i][j] == 0.0:
                continue
            patch = get_exact_patch(image, i, j, exact_window_size, exact_window_decision_kernel_size)
            patches.append(patch)
    patches = np.array(patches)
    return patches


def get_filter_result(compressed_image):
    filtering_model_weights_name = "small_filter_net_1_weights_1"
    filtering_window_size = 20

    filter_model = Model.Model()
    filter_model.init_model()
    filter_model.load_weights(filtering_model_weights_name)
    filter_result = np.zeros((len(compressed_image), len(compressed_image[0])))

    filter_patches = get_filter_patches(compressed_image, filtering_window_size)
    results = filter_model.model.predict(filter_patches)

    idx = -1
    for i in range(0, len(compressed_image) - filtering_window_size + 1, filtering_window_size):
        for j in range(0, len(compressed_image) - filtering_window_size + 1, filtering_window_size):
            idx += 1
            single_res = results[idx]
            if single_res[0] > 0.90 and single_res[1] < 0.2:
                dec = 0
            else:
                dec = 1
            if dec == 1:
                for x in range(filtering_window_size):
                    for y in range(filtering_window_size):
                        filter_result[x + i][y + j] = 1.0

    return filter_result


def get_exact_result(compressed_image, filtered_image):
    exact_model_weights_name = "exact_net_1_weights_22"
    exact_window_size = 20
    exact_window_decision_kernel_size = 2
    exact_model = Model.Model()
    exact_model.init_model()
    exact_model.load_weights(exact_model_weights_name)
    exact_result = np.zeros((len(compressed_image), len(compressed_image)), dtype=int)

    patches = get_exact_patches(compressed_image, exact_window_size, exact_window_decision_kernel_size, filtered_image)
    res = exact_model.model.predict(patches)

    idx = -1
    for i in range(0, len(compressed_image) - exact_window_decision_kernel_size + 1, 2):
        for j in range(0, len(compressed_image) - exact_window_decision_kernel_size + 1, 2):
            if filtered_image[i][j] == 0:
                continue
            idx += 1
            single_res = res[idx]
            if single_res[1] > 0.5 or single_res[0] > 0.5:
                dec = 1 if single_res[1] > single_res[0] else 0
            else:
                dec = 0
            for x in range(exact_window_decision_kernel_size):
                for y in range(exact_window_decision_kernel_size):
                    exact_result[x + i][y + j] = dec
    return exact_result


def roads(image):
    compressed_image = DataLoader.get_compressed_image(image, 600)
    compressed_image = compressed_image / 255

    filtered_image = get_filter_result(compressed_image)
    exact_result = get_exact_result(compressed_image, filtered_image)
    processed_image = Postprocessing.process_image(exact_result)
    return processed_image
