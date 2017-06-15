import Model
import DataLoader
import numpy as np


def get_patch_by_center(input_image, x, y, half_window_size):
    return np.array([input_image[x - half_window_size:x + half_window_size + 1,
            y - half_window_size:y + half_window_size + 1]])


def roads(image):
    model_name = "net_1"
    model_weights_name = "net_1_weights_1"
    window_size = 31
    half_window_size = int(window_size / 2)
    model = Model.Model()
    print("model before init")
    model.init_model()
    print("model init")
    model.load_weights(model_weights_name)
    print("model initiated")
    result = np.zeros((len(image), len(image[0])))

    for i in range(half_window_size, len(image) - 1 - half_window_size):
        for j in range(half_window_size, len(image[0]) - 1 - half_window_size):
            patch = get_patch_by_center(image, i, j, half_window_size)

            print('patch {1}\n{0}\n{1}'.format(patch, ''.join(['=' for i in range(30)])))
            predicted_value = model.model.predict(patch)
            result[i][j] = predicted_value[0][0][0]
    return result
