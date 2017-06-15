import Model
import DataLoader


def roads(image):
    model_name = "net_1"
    model_weights_name = "net_1_weights_1"
    model = Model.Model()
    model.load_model(model_name, model_weights_name)
    output = model.model.predict(image)
    DataLoader.show_image(output)
    return output
