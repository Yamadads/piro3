import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import DataLoader
import roads2

learning_files_path = './pictures/'
learning_labels_path = './labels/'


def display_first_test():
    data = DataLoader.get_images_names_list(learning_files_path, learning_labels_path)
    testing_image = DataLoader.get_image(data[0][1])
    reference_solution = DataLoader.get_image(data[0][2])
    network_solution = roads2.roads(testing_image)

    # DataLoader.show_image(testing_image)
    DataLoader.show_image(network_solution)
    DataLoader.show_image(reference_solution)


def main():
    display_first_test()


if __name__ == '__main__':
    main()
