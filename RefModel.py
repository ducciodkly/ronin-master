from tensorflow.keras import optimizers, layers, losses, datasets, utils, callbacks, initializers, regularizers
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import time, pickle

path = './drive/MyDrive/DeepLearningDatasets/SavedData_Chinese/'
identity = 'CNV'; maxMC_inner = 20; epochs = 15

def scheduler(epoch, lr):
  if epoch % 3 == 0 and epoch > 0:    return lr * 0.5
  else:    return lr

# region Build model
def buildModel():
  kernel_regularizer = regularizers.L2(l2=0.0005)
  optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
  model = keras.Sequential([keras.Input(shape=input_shape),
    # not effective: kernel_initializer=initializers.RandomNormal(mean=0.0,stddev=0.01,seed=None)
    layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="relu", bias_initializer='zeros',kernel_regularizer=kernel_regularizer),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="relu", bias_initializer='zeros',kernel_regularizer=kernel_regularizer),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation="relu", bias_initializer='zeros',kernel_regularizer=kernel_regularizer),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"), ])
  model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  return model

# optimizer = optimizers.Adam(learning_rate=0.01)
# kernel_initializer = initializers.RandomNormal(mean=0.0,stddev=0.01,seed=None)
# kernel_regularizer = regularizers.L2(l2=0.0005)
# optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
# reduce_lr = callbacks.LearningRateScheduler(scheduler,verbose=0)
# endregion

# region Dataset preprocessing
train = np.load(path + 'Datasets/' + 'data10_train.npy')
test = np.load(path + 'Datasets/' + 'data10_test.npy')
x_train = train[:, 1:]; y_train = train[:, 0]
x_test = test[:, 1:]; y_test = test[:, 0]

num_classes = np.max(np.concatenate((y_test, y_train))) + 1
image_width = 64; input_shape = (image_width, image_width, 1)
x_train = np.reshape(x_train, (x_train.shape[0], image_width, -1))
x_test = np.reshape(x_test, (x_test.shape[0], image_width, -1))
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
# y_train = utils.to_categorical(y_train, num_classes)
# y_test = utils.to_categorical(y_test, num_classes)
print("train shape:", x_train.shape, "- test shape:", x_test.shape)
# plt.imshow(x_train[0],cmap='gray'); plt.show(block=False)
# chineseDatasets = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
# endregion

batch_size = 100
historyTrain = np.zeros((maxMC_inner, 2, epochs))
historyTest = np.zeros((maxMC_inner, 2))
historyTime = np.zeros(maxMC_inner)
for MC in range(maxMC_inner):
  print("Monte Carlo:", MC + 1)
  reduce_lr = callbacks.LearningRateScheduler(scheduler, verbose=0)
  model = 1
  model = buildModel()

  s_time = time.time()
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0., callbacks=reduce_lr, verbose=0)
  score = model.evaluate(x_test, y_test, verbose=1)
  e_time = time.time();  elapsedTime = e_time - s_time;  print(elapsedTime, 's')
  # print("Test loss:", score[0]); print("Test accuracy:", score[1])

  historyTrain[MC, 0, :] = 100 * np.array(history.history['accuracy'])
  historyTrain[MC, 1, :] = np.array(history.history['loss'])
  historyTest[MC, 0] = 100 * score[1]  # acc in %
  historyTest[MC, 1] = score[0]  # loss
  historyTime[MC] = elapsedTime
  # model.save(path + identity + '_' + str(MC) + '_KerasModel.h5')

refData = [historyTrain, historyTest, historyTime]
with open(path + identity + '_refData', "wb") as pickling: pickle.dump(refData, pickling)
print('aver_acc_train:', np.mean(historyTrain[:, 0, :], axis=0))
print('aver_los_train:', np.mean(historyTrain[:, 1, :], axis=0))
print('aver_accu_test:', np.mean(historyTest[:, 0], axis=0))
print('aver_loss_test:', np.mean(historyTest[:, 1], axis=0))
print('aver_time:', np.mean(historyTime))
