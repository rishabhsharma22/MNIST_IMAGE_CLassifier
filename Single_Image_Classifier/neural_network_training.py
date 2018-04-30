# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:51:51 2018

@author: Rishabh Sharma
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from IPython.display import display
params = {'legend.fontsize': 16,
          'legend.handlelength': 2,
          'figure.figsize': (14,12),
          'axes.titlesize': 16,
          'axes.labelsize': 16
         }
plt.rcParams.update(params)


import h5py
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 128
num_classes = 10
epochs = 5

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("X_train original shape", x_train.shape)
print("y_train original shape", y_train.shape)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape (after adding channels):', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), input_shape=input_shape, name='Input_Conv2D_1', activation='relu'))
model2.add(BatchNormalization(axis=-1, name='BatchNorm_1'))

model2.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu', name='Conv2D_2'))
model2.add(BatchNormalization(axis=-1, name='BatchNorm_2'))
model2.add(MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'))
model2.add(Dropout(0.25, name='Dropout_2'))

model2.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), name='Conv2D_3', activation='relu'))
model2.add(BatchNormalization(axis=-1, name='BatchNorm_3'))
model2.add(MaxPooling2D(pool_size=(2,2), name='MaxPool_3'))

model2.add(Flatten(name='Flatten')) # Fully connected layer
model2.add(BatchNormalization(name='BatchNorm_Flatten'))

model2.add(Dense(128, name='Dense_5', activation='relu'))
model2.add(BatchNormalization(name='BatchNorm_5'))
model2.add(Dropout(0.5, name='Dropout_5'))

model2.add(Dense(num_classes, activation='softmax', name='SoftMax_Output'))

model2.summary()


model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(x_train, y_train,batch_size=batch_size, epochs=epochs,
           validation_split=0.1, verbose=1)

score2 = model2.evaluate(x_test, y_test)
print()
print('Test accuracy: ', score2[1])

model2.save("mnist_cnn_batchnorm2.h5", overwrite=True)

