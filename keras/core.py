# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
import utils
import random


TRAIN_SIZE=60
print('Building model...')
model = Sequential()
model.add(Dense(500, input_shape = (TRAIN_SIZE, )))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer='adam', 
              loss='mse')

#Generate some test data, here a sine wave of two cycles. Prediction at end is zero.
Fs = 100
items_per_set = TRAIN_SIZE
scale=10.0
training_sets = TRAIN_SIZE
num_cycles=7

X_train = []
Y_train = []
for i in range(training_sets):
    x = np.arange(items_per_set)
    y = np.sin(2 * np.pi *  x / TRAIN_SIZE * num_cycles )
    yr = [y + random.random()/scale for y in y]
    X_train.append(yr)
    result=0
    Y_train.append(result)


Y_test = Y_train
X_test = X_train

model.fit(X_train, 
          Y_train, 
          nb_epoch=20, 
          batch_size = 128, 
          verbose=1, 
          validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=128)
print("SCORE:", score)



predicted = model.predict(X_test)
#print("predicted:", predicted)

mse = mean_squared_error(predicted, Y_train)
print("MSE:", mse)
fig = plt.figure()
plt.title("Actual vs Predicted")
plt.plot(X_test,predicted, color='red')
plt.show()
