import DataLoader
import numpy as np
import Model

seed = 7
np.random.seed(seed)


def learn(images_dir, labels_dir, window_size, patch_size, start_learning_example_number, stop_learning_example_number,
          batch_size, epochs, model_name, model_weights_name):
    # load data
    data = DataLoader.get_images_names_list(images_dir, labels_dir)

    # load current model
    model = Model.Model()
    model.init_model()
    model.load_weights(model_weights_name)

    # learn
    for i in range(start_learning_example_number, stop_learning_example_number):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        image = DataLoader.get_image(data[i][1])
        label = DataLoader.get_image(data[i][2])
        train_pictures, labels = DataLoader.split_image(window_size, image, label, patch_size)
        model.train_model(train_pictures, labels, batch_size, epochs)
        model.save_model(model_name, model_weights_name)


def main():
    learning_set_count = 1108
    learning_files_path = 'D:/piro/piro3/pictures/'
    learning_labels_path = 'D:/piro/piro3/labels/'
    model_name = "net_1"
    model_weights_name = "net_1_weights_1"
    window_size = 31  # must be odd
    patch_size = 150
    start = 0
    stop = 1
    batch_size = 30
    epochs = 1

    learn(learning_files_path, learning_labels_path, window_size, patch_size, start, stop, batch_size, epochs,
          model_name,
          model_weights_name)


if __name__ == '__main__':
    print("start")
    main()
