import Model
import DataLoader
import numpy as np


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
            patch = image[i - half_window_size:i + half_window_size + 1,
                    j - half_window_size:j + half_window_size + 1]
            print(patch)
            #predicted_value = model.model.predict(patch)
            #result[i][j] = predicted_value[0][0][0]
    output = model.model.predict(image)
    DataLoader.show_image(output)
    return output
