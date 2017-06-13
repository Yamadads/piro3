import DataLoader
import numpy as np

seed = 7
np.random.seed(seed)


def learn(images_dir, labels_dir, batch_size):
    data = DataLoader.get_images_names_list(images_dir, labels_dir)
    train_pictures = []
    labels = []
    for i in range(len(data)):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))
        train_pictures.append(DataLoader.get_image(data[i][1]))
        labels.append(DataLoader.get_image(data[i][2]))
        if i % batch_size == 0 and i>0:
            print("learning")



def main():
    learning_files_path = 'D:/Users/sebastian.pawlak/Downloads/pliki'
    learning_labels_path = 'D:/Users/sebastian.pawlak/Downloads/pliki'
    batch_size = 10
    learn(learning_files_path, learning_labels_path)


if __name__ == '__main__':
    print("start")
    main()
