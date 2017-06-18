import FilteringDataLoader
import numpy as np
import FilteringModel

seed = 46
np.random.seed(seed)


def learn(images_dir, labels_dir, window_size, start_learning_example_number, stop_learning_example_number,
          batch_size, epochs, model_name, model_weights_name):
    # load data
    data = FilteringDataLoader.get_images_names_list(images_dir, labels_dir)

    # load current model
    model = FilteringModel.FilteringModel()
    model.init_model()
    model.load_weights(model_weights_name)

    step = 10

    # learn
    for i in range(start_learning_example_number + step, stop_learning_example_number + 1, step):

        images_f = []
        labels_f = []

        print("set id = {0}".format(int(i / step)))
        for j in range(i - step, i):
            print("pic num: {0}, name: {1}".format(j, data[j][0]))
            images_f.append(FilteringDataLoader.get_image(data[j][1], 600))
            labels_f.append(FilteringDataLoader.get_image(data[j][2], 600))

        train_pictures, labels = FilteringDataLoader.split_image(window_size, images_f, labels_f, 15)
        model.train_model(train_pictures, labels, batch_size, epochs)
        model.save_model(model_name, model_weights_name)


def main():
    learning_files_path = './pictures/'
    learning_labels_path = './labels/'
    model_name = "filter_net_1"
    model_weights_name = "filter_net_1_weights_1"
    window_size = 60  # must be odd
    start = 0
    stop = 100
    batch_size = 32
    epochs = 4

    learn(learning_files_path, learning_labels_path, window_size, start, stop, batch_size, epochs,
          model_name,
          model_weights_name)


if __name__ == '__main__':
    print("FilterLearning started")
    main()
