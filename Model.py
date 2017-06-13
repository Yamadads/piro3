from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

import numpy as np


class Model:
    def __init__(self, gamma, pictures):
        self.gamma = gamma

        self.batch_size = 150
        self.epochs = 20
        self.pool_size = (2, 2)
        self.input_shape = pictures.shape[1:]

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        self.model.add(Convolution2D(60, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(Convolution2D(50, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Convolution2D(40, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Deconvolution2D(40, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[4].output_shape))
        self.model.add(Dropout(0.2))
        self.model.add(UpSampling2D(size=self.pool_size))
        self.model.add(Deconvolution2D(50, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[3].output_shape))
        self.model.add(Deconvolution2D(60, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[1].output_shape))
        self.model.add(Deconvolution2D(1, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu',
                                       output_shape=self.model.layers[0].output_shape))


        self.model.compile(optimizer='Adam', loss='mean_squared_error')

    def train_model(self, pictures, labels):
        labels = labels / 255
        train_images, labels = shuffle(pictures, labels)
        datagen = ImageDataGenerator()
        datagen.fit(pictures)
        self.model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch=len(X_train),
                                 nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

    def save_model(self, architecture_filename, weights_filename):
        model_json = self.model.to_json()
        with open(architecture_filename, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights_filename)