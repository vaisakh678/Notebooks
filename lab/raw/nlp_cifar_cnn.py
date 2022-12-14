# -*- coding: utf-8 -*-
"""NLP CIFAR CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ebStY-KT_GODIgngDumx9dWlsYUWPW-B
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

cifar =  keras.datasets.cifar10

(x_train, y_train),(x_test, y_test) = cifar.load_data()
print(x_train.shape)

plt.imshow(x_train[7])

print(y_test[0])

x_train[0]

x_train[0].shape

x_train_norm = x_train/255
x_test_norm = x_test/255

x_train_norm.shape

x_train_norm[0]

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

print(x_train.shape)
print(x_test.shape)

from tensorflow.keras.utils import to_categorical
y_cat_train  = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4,4), input_shape = (32, 32, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (4,4), input_shape = (32, 32, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(512, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))

model.add(Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.summary()

"""Alert: This Will Take very Looooooooooooooooooooong Time to Complete"""

model_history = model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test))

