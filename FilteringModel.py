import os

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense
from sklearn.utils import shuffle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FilteringModel:
    def __init__(self):
        self.model = Sequential()

    def init_model(self):
        pool_size = (2, 2)
        input_shape = (11, 11, 3)
        kernel_size = (3, 3)
        padding = 'valid'
        activation = 'relu'

        self.model = Sequential()
        self.model.add(
            Conv2D(50, kernel_size=kernel_size, input_shape=input_shape, padding=padding, strides=(1, 1),
                   activation=activation))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(40, kernel_size=kernel_size, padding=padding, strides=(1, 1), activation=activation))
        self.model.add(MaxPooling2D(pool_size=pool_size, dim_ordering="tf"))
        self.model.add(Conv2D(40, kernel_size=(2, 2), padding=padding, strides=(1, 1), activation=activation))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(10, kernel_size=(2, 2), padding=padding, strides=(1, 1), activation=activation))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='Adam', loss='mean_squared_error')

    def train_model(self, learning_pictures, learning_labels, batch_size, epochs):
        labels = learning_labels / 255
        pictures = learning_pictures / 255
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