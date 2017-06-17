import FilteringDataLoader
import numpy as np
import FilteringModel

seed = 7
np.random.seed(seed)


def learn(images_dir, labels_dir, window_size, patches_count, start_learning_example_number, stop_learning_example_number,
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
        image = FilteringDataLoader.get_image(data[i][1], 200)
        label = FilteringDataLoader.get_image(data[i][2], 200)
        train_pictures, labels = FilteringDataLoader.split_image(window_size, image, label, patches_count)
        model.train_model(train_pictures, labels, batch_size, epochs)
        model.save_model(model_name, model_weights_name)


def main():
    learning_set_count = 1108
    learning_files_path = './pictures/'
    learning_labels_path = './labels/'
    model_name = "filter_net_1"
    model_weights_name = "filter_net_1_weights_1"
    window_size = 20  # must be odd
    patches_count = 300
    start = 0
    stop = 300
    batch_size = 32
    epochs = 4

    learn(learning_files_path, learning_labels_path, window_size, patches_count, start, stop, batch_size, epochs,
          model_name,
          model_weights_name)


if __name__ == '__main__':
    print("FilterLearning started")
    main()
