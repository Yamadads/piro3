import DataLoader
import numpy as np
import Model

seed = 7
np.random.seed(seed)


# DataLoader.show_image(labels[i])
# DataLoader.show_image(train_pictures[i])

def learn(images_dir, labels_dir, trainig_set_size, start_learning_example_number, stop_learning_example_number,
          batch_size, epochs, model_name, model_weights_name):
    #load data
    data = DataLoader.get_images_names_list(images_dir, labels_dir)
    train_pictures = []
    labels = []

    #load current model
    model = Model.Model()
    model.init_model()
    model.load_weights(model_weights_name)

    #learn
    for i in range(start_learning_example_number, stop_learning_example_number):
        if i % trainig_set_size == 0 and i > 0:
            print("learning")
            model.train_model(train_pictures, labels, batch_size, epochs)
            train_pictures = []
            labels = []
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        train_pictures.append(DataLoader.get_image(data[i][1]))
        labels.append(DataLoader.get_image(data[i][2]))

    if len(train_pictures) > 0:
        model.train_model((train_pictures, labels, batch_size, epochs))

    #save model
    model.save_model(model_name, model_weights_name)


def main():
    learning_set_count = 1108
    learning_files_path = 'D:/piro/piro3/pictures/'
    learning_labels_path = 'D:/piro/piro3/labels/'
    model_name = "net_1"
    model_weights_name = "net_1_weights_1"
    trainig_set_size = 6
    start = 0
    stop = 12
    batch_size = 3
    epochs = 1

    learn(learning_files_path, learning_labels_path, trainig_set_size, start, stop, batch_size, epochs, model_name, model_weights_name)


if __name__ == '__main__':
    print("start")
    main()
