import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras import optimizers
from keras import losses
import numpy as np


class Model:
    def __init__(self):
        self.model = Sequential()

    def init_model(self):
        input_shape = (20, 20, 3)
        padding = 'valid'
        activation = 'relu'

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding=padding, activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(50, kernel_size=(4, 4), padding=padding, activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation=activation))
        self.model.add(Dense(60, activation=activation))
        self.model.add(Dense(2, activation='softmax'))

        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5)
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adadelta(), metrics=["accuracy"])
        # self.model.compile(optimizer='Adam', loss='mean_squared_error')
        # self.set_random_weights()

    def set_random_weights(self):
        weights = self.model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self.model.set_weights(weights)
        print(weights)

    def train_model(self, learning_pictures, learning_labels, batch_size, epochs):
        pictures = learning_pictures / 255
        # pictures, labels = shuffle(pictures, learning_labels)
        self.model.fit(pictures, learning_labels, batch_size, epochs, 1, validation_split=0.1, shuffle=True)

    def save_model(self, architecture_filename, weights_filename):
        model_json = self.model.to_json()
        with open(architecture_filename, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights_filename)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

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
        # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])
