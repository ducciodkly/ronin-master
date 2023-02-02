import time

from tensorflow.keras import optimizers, layers, losses, datasets, utils, callbacks, initializers, regularizers
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

path = ''
identity = 'CNV';maxMC_inner=20; epochs = 1

def scheduler(epoch,lr):
  if epoch % 2 == 0 and epoch >0 : return lr*0.1
  else : return lr
def buildModel():
  kernel_regularizer = regularizers.L2(L2= 0.0005)
  optimizer = optimizers.Adam(learning_rate=0.0001,momentum =0.9)
  model = keras.Sequential([keras.Input(shape =input_shape),

    layers.conv2D(filter = 32, kernel_size =3 , strides =1 , padding = 'same', activations = "relu",
                  kernel_regularizer=kernel_regularizer),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    layers.conv2D(filter=64, kernel_size=3, strides=1, padding='same', activations="relu",
                kernel_regularizer=kernel_regularizer),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

    layers.Flatten(),
    layers.Dropout(0,2),
    layers.Dense(32, activation= "relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax") , ])
  model.compile(loss = "categorical_cross_entropy", optimizer= optimizer , metrics= ["accuracy"])
  return model
input_shape = (6, 200, 1)
#
#
train = np.load(path+ 'Dataset/' + 'dataPDR_train.npy')
test = np.load(path+ 'Dataset/' + 'dataPDR_test.npy')
x_train = train[:, 1:]; y_train = train[: , 0]
x_test = test[:, 1:]; y_test = test[: , 0]
#
num_classes = np.max(np.concatenate((y_test, y_train))) + 1
# #####
# batch_size = 128

x_train = np.reshape(x_train,(x_train.shapep[0],6,-1))
x_test = np.reshape(x_train,(x_test.shape[0],6,-1))
x_train = np.expand_dims(x_train.astype("float32" /255,-1))
x_test = np.expand_dims(x_test.astype("float32"/255,-1))
y_train = utils.to_categorical(y_train,num_classes)
y_test = utils.to_categorical(y_test,num_classes)
print("train shape:",x_train.shape, "test shape:", x_test.shape)
print("train shape: ",x_train.shape, "- test shape:", x_test.shape)
plt.imshow(x_train[0],cmap='gray'); plt.show(block=False)
PDRmodel= {'x_train':x_train,'x_test': x_test,'y_train':y_train,'y_test':y_test }
batch_size = 128
historyTrain = np.zeros((maxMC_inner,2,epochs))
historyTest = np.zeros((maxMC_inner,2))
historyTime = np.zeros((maxMC_inner))
for MC in range(maxMC_inner):
  print("Monte Carlo",MC + 1)
  reduce_lr = callbacks.LearningRateScheduler(scheduler,verbose=0)
  model = 1
  model = buildModel()

  s_time = time.time()
  history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0., callbacks=reduce_lr,verbose=0)
  score = model.evaluate(x_test,y_test,verbose=1)
  e_time=time.time(); elapsedTime = e_time-s_time; print(elapsedTime,'s')
  print("Test lost:",score[0]); print("Test accuracy",score[1])


  historyTrain[MC, 0, :]=100*np.array(history.history['accuracy'])
  historyTrain[MC, 1, :] = 100 * np.array(history.history['loss'])
  historyTest[MC,0]=100*score[1]
  historyTest[MC, 1] = score[0]
  historyTime[MC]=elapsedTime
  model.save(path+identity+'_'+str(MC)+'KerasModel.h5')

refData = [historyTrain,historyTest,historyTime]
# with open(path + identity +'refData','wb') as pickling : pickle.dump(refData,pickling)
