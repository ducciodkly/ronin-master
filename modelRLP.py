import keras
import numpy as np

import config
from data_provider import Datasets
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

# ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
#               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
#               25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32:'6', 33:'7', 34:'8', 35:'9'}

class CNN_Model(object):
    def __init__(self, trainable=True):
        self.batch_size = config.BATCH_SIZE
        self.trainable = trainable
        self.num_epochs = config.EPOCHS

        # Building model
        self._build_model()

        # Input data
        if trainable:
            self.model.summary()
            self.data = Datasets()

        # self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])
        self.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['acc'])

    def _build_model(self):
        # CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='softmax'))
        self.model.summary()

    def train(self):
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, )
        # Model Checkpoint
        cpt_save = ModelCheckpoint('weights/model_weight.h5', save_best_only=True, monitor='val_acc', mode='max')

        print("Training......")
        trainX, trainY = self.data.gen()
        trainX = np.array(trainX)

        self.model.fit(trainX, trainY, validation_split=0.15, callbacks=[cpt_save, reduce_lr], verbose=1,
                       epochs=self.num_epochs, shuffle=True, batch_size=self.batch_size)
