from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2DTranspose, Conv2D, Dropout
from sklearn.utils import shuffle
import os
import numpy as np


class Model:
    def __init__(self):
        self.model = Sequential()

    def init_model(self):
        pool_size = (2, 2)
        input_shape = (1500, 1500, 3)
        output_shape = (1500, 1500)  # not used
        kernel_size = (8, 8)
        padding = 'same'
        activation = 'relu'

        self.model = Sequential()
        self.model.add(
            Conv2D(60, kernel_size=kernel_size, input_shape=input_shape, padding=padding, strides=(1, 1),
                   activation=activation))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(50, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        # self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Conv2D(40, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        self.model.add(Dropout(0.2))
        self.model.add(
            Conv2DTranspose(40, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        # self.model.add(UpSampling2D(size=self.pool_size))
        self.model.add(
            Conv2DTranspose(50, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        self.model.add(
            Conv2DTranspose(60, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        self.model.add(
            Conv2DTranspose(1, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))

        self.model.compile(optimizer='Adam', loss='mean_squared_error')

    def train_model(self, learning_pictures, learning_labels, batch_size, epochs):
        pictures = np.array(learning_pictures)
        labels = np.array(learning_labels)
        labels = labels / np.max(labels)
        pictures = pictures / np.max(pictures)
        pictures, labels = shuffle(pictures, labels)
        self.model.fit(pictures, labels, batch_size, epochs, 1, validation_split=0.1, shuffle=True)

    def save_model(self, architecture_filename, weights_filename):
        model_json = self.model.to_json()
        with open(architecture_filename, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights_filename)

    def load_model(self, architecture_filename, weights_filename):
        if os.path.exists(architecture_filename) and os.path.exists(weights_filename):
            json_file = open(architecture_filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(weights_filename)
            print("Loaded model from disk")
        else:
            print("Model cannot be loaded")

    def load_weights(self, weights_filename):
        if os.path.exists(weights_filename):
            self.model.load_weights(weights_filename)
        else:
            print('Model weights cannot be loaded')
