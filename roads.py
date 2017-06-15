import Model
import DataLoader
import numpy as np
import scipy.misc
import progressbar


def get_patch_by_center(input_image, x, y, half_window_size):
    return np.array([input_image[x - half_window_size:x + half_window_size + 1,
            y - half_window_size:y + half_window_size + 1]])


def generate_road_image(image):
    model_name = "net_1"
    model_weights_name = "net_1_weights_1"
    window_size = 31
    half_window_size = int(window_size / 2)
    model = Model.Model()
    model.load_model(model_name, model_weights_name)
    result = np.zeros((len(image), len(image[0])))

    i_start = half_window_size
    i_end = len(image) - 1 - half_window_size
    j_start = half_window_size
    j_end = len(image[0]) - 1 - half_window_size
    with progressbar.ProgressBar(max_value=(i_end - i_start) * (j_end - j_start)) as bar:
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                patch = get_patch_by_center(image, i, j, half_window_size)

                # print('patch {1}\n{0}\n{1}'.format(patch, ''.join(['=' for i in range(30)])))
                predicted_value = model.model.predict(patch)
                result[i][j] = predicted_value[0][0][0]

                bar.update(i * j)

    scipy.misc.imsave('test_result', result)

    output = model.model.predict(image)
    DataLoader.show_image(output)

    return output
