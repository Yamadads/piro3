import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import DataLoader
import roads

learning_files_path = './pictures/'
learning_labels_path = './labels/'


def display_first_test():
    test_image_no = 2
    data = DataLoader.get_images_names_list(learning_files_path, learning_labels_path)
    testing_image = DataLoader.get_image(data[test_image_no][1])
    reference_solution = DataLoader.get_image(data[test_image_no][2])
    network_solution = roads.roads(testing_image)

    # DataLoader.show_image(testing_image)
    DataLoader.show_image(network_solution.astype(int))
    DataLoader.show_image(reference_solution)
    DataLoader.save_image(network_solution, './results/result1.tif')


def main():
    display_first_test()


if __name__ == '__main__':
    main()
