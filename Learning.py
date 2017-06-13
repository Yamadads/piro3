import DataLoader
import numpy as np

seed = 7
np.random.seed(seed)


def learn(images_dir, labes_dir):
    data = DataLoader.get_images_names_list(images_dir, labes_dir)
    for i in range(len(data)):
        print("pic num: {0}, name: {1}".format(i, data[i][0]))


def main():
    learning_files_path = 'D:/Users/sebastian.pawlak/Downloads/pliki'
    learning_labels_path = 'D:/Users/sebastian.pawlak/Downloads/pliki'

    learn(learning_files_path, learning_labels_path)


if __name__ == '__main__':
    print("start")
    main()
