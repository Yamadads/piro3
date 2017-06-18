import FilteringDataLoader
import numpy as np
import FilteringModel

seed = 7
np.random.seed(seed)


def learn(images_dir, labels_dir, window_size, start_learning_example_number, stop_learning_example_number,
          batch_size, epochs, model_name, model_weights_name):
    # load data
    data = FilteringDataLoader.get_images_names_list(images_dir, labels_dir)

    # load current model
    model = FilteringModel.FilteringModel()
    model.init_model()
    model.load_weights(model_weights_name)

    # learn
    for i in range(start_learning_example_number, stop_learning_example_number):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        image = FilteringDataLoader.get_image(data[i][1], 600)
        label = FilteringDataLoader.get_image(data[i][2], 600)
        train_pictures, labels = FilteringDataLoader.split_image(window_size, image, label, 15)
        model.train_model(train_pictures, labels, batch_size, epochs)
        model.save_model(model_name, model_weights_name)


def main():
    learning_files_path = './pictures/'
    learning_labels_path = './labels/'
    model_name = "filter_net_1"
    model_weights_name = "filter_net_1_weights_1"
    window_size = 60  # must be odd
    start = 0
    stop = 1
    batch_size = 32
    epochs = 4

    learn(learning_files_path, learning_labels_path, window_size, start, stop, batch_size, epochs,
          model_name,
          model_weights_name)


if __name__ == '__main__':
    print("FilterLearning started")
    main()
