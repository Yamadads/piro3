import DataLoader
import numpy as np
import Model

seed = 7
np.random.seed(seed)

#DataLoader.show_image(labels[i])
#DataLoader.show_image(train_pictures[i])

def learn(images_dir, labels_dir, batch_size):
    data = DataLoader.get_images_names_list(images_dir, labels_dir)
    train_pictures = []
    labels = []
    model = Model.Model()
    for i in range(len(data)):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        train_pictures.append(DataLoader.get_image(data[i][1]))
        labels.append(DataLoader.get_image(data[i][2]))
        print(train_pictures[0].shape)
        print(labels[0].shape)
        if i % batch_size == 0 and i > 0:
            print("learning")
            model.train_model(train_pictures, labels)
            train_pictures = []
            labels = []
    model.save_model("network_1_architecture", "network_1_weights")


def main():
    learning_files_path = 'D:/piro/piro3/pictures/'
    learning_labels_path = 'D:/piro/piro3/labels/'
    batch_size = 10

    learn(learning_files_path, learning_labels_path, batch_size)


if __name__ == '__main__':
    print("start")
    main()
