import DataLoader
import numpy as np
import ExactModel

seed = 7
np.random.seed(seed)


def learn(images_dir, labels_dir, compressed_image_size, decision_kernel_size, window_size, patch_size, start_learning_example_number, stop_learning_example_number,
          batch_size, epochs, model_name, model_weights_name):
    # load paths
    data = DataLoader.get_images_names_list(images_dir, labels_dir)

    # load current model
    model = ExactModel.Model()
    model.init_model()
    model.load_weights(model_weights_name)

    # learn
    for i in range(start_learning_example_number, stop_learning_example_number):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        image = DataLoader.get_image(data[i][1])
        compressed_image = DataLoader.get_compressed_image(image, compressed_image_size)
        label = DataLoader.get_compressed_image(DataLoader.get_image(data[i][2]), compressed_image_size)
        train_pictures, labels = DataLoader.split_image(window_size, decision_kernel_size, compressed_image, label, patch_size)
        model.train_model(train_pictures, labels, batch_size, epochs)
        model.save_model(model_name, model_weights_name)


def main():
    learning_set_count = 1108
    learning_files_path = './pictures/'
    learning_labels_path = './labels/'
    model_name = "exact_net_1"
    model_weights_name = "exact_net_1_weights_22"
    window_size = 20
    decision_kernel_size = 2
    patch_size = 10000
    start = 0
    stop = 1000
    batch_size = 128
    epochs = 12
    compressed_image_size = 600

    learn(learning_files_path, learning_labels_path, compressed_image_size, decision_kernel_size, window_size, patch_size, start, stop, batch_size, epochs,
          model_name,
          model_weights_name)


if __name__ == '__main__':
    print("start")
    main()
