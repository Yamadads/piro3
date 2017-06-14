from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Deconv2D, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np


class Model:
    def __init__(self):
        self.batch_size = 150
        self.epochs = 20
        self.pool_size = (2, 2)
        self.input_shape = (1500, 1500, 3)
        self.output_shape = (1500, 1500)

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        self.model.add(Conv2D(60, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(Conv2D(50, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Conv2D(40, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Deconv2D(40, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[4].output_shape))
        self.model.add(Dropout(0.2))
        self.model.add(UpSampling2D(size=self.pool_size))
        self.model.add(Deconv2D(50, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[3].output_shape))
        self.model.add(Deconv2D(60, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[1].output_shape))
        self.model.add(Deconv2D(1, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.output_shape))

        self.model.compile(optimizer='Adam', loss='mean_squared_error')

    def train_model(self, learning_pictures, learning_labels):
        pictures = np.array(learning_pictures)
        labels = np.array(learning_labels)
        labels = labels / 255
        pictures, labels = shuffle(pictures, labels)
        self.model.fit(pictures, labels, self.batch_size, self.epochs, 1, validation_split=0.1, shuffle=True)

    def save_model(self, architecture_filename, weights_filename):
        model_json = self.model.to_json()
        with open(architecture_filename, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights_filename)
