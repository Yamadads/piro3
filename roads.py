import Model
import numpy as np
import scipy.misc

import time

progressbar_enabled = True

if progressbar_enabled:
    from progressbar import ProgressBar
else:
    class ProgressBar(object):
        def __init__(self, max_value=0):
            pass

        def start(self, **kwargs):
            pass

        def update(self, value=None):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def __enter__(self):
            return self


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

    part = 1

    i_start = half_window_size
    i_end = int((len(image) - 1 - half_window_size) * part)
    j_start = half_window_size
    j_end = int((len(image[0]) - 1 - half_window_size) * part)

    predict_time = 0
    patch_time = 0

    with ProgressBar(max_value=(i_end - i_start + 1) * (j_end - j_start + 1)) as bar:
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                patch_time_start = time.clock()
                patch = get_patch_by_center(image, i, j, half_window_size)
                patch_time += time.clock() - patch_time_start

                predict_time_start = time.clock()
                predicted_value = model.model.predict(patch)
                predict_time += time.clock() - predict_time_start

                result[i][j] = predicted_value[0][0][0]

                bar.update((i - i_start) * (j_end - j_start) + j)

    print('predict {0}\npatch   {1}'.format(predict_time, patch_time))

    # scipy.misc.imsave('test_result', result)

    return result