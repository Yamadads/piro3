import FilteringDataLoader
import numpy as np
import FilteringModel

seed = 7
np.random.seed(seed)


def filter_testing(images_dir, labels_dir, window_size, start_testing_example_number, stop_testing_example_number,
                   model_weights_name):
    # load data
    data = FilteringDataLoader.get_images_names_list(images_dir, labels_dir)

    # load current model
    model = FilteringModel.FilteringModel()
    model.init_model()
    model.load_weights(model_weights_name)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    image_batch = []
    label_batch = []

    for i in range(start_testing_example_number, stop_testing_example_number):
        image_batch.append(FilteringDataLoader.get_image(data[i][1], 600))
        label_batch.append(FilteringDataLoader.get_image(data[i][2], 600))

    test_pictures, labels = FilteringDataLoader.split_image(window_size, image_batch, label_batch, 15)
    results = model.model.predict(test_pictures)
    print(results)
    for j in range(len(results)):
        if results[j][0] == 1 and results[j][1] == 0:  # negative
            if labels[j][0] == 1 and labels[j][1] == 0:
                tn += 1
            else:
                fn += 1
        else:  # positive
            if labels[j][0] == 0 and labels[j][1] == 1:
                tp += 1
            else:
                fp += 1

    print('=============================================')
    print('|  p/e      |    positive   |   negative    |')
    print('|-----------|---------------|---------------|')
    print('|  positive |      {0}      |      {1}      |'.format(tp, fp))
    print('|-----------|---------------|---------------|')
    print('| negative  |      {0}      |      {1}      |'.format(fn, tn))
    print('=============================================')


def main():
    learning_files_path = './pictures/'
    learning_labels_path = './labels/'
    model_weights_name = "filter_net_1_weights_1"
    window_size = 60  # must be odd
    start = 0
    stop = 30

    filter_testing(learning_files_path, learning_labels_path, window_size, start, stop,
                   model_weights_name)


if __name__ == '__main__':
    print("FilterLearning started")
    main()
