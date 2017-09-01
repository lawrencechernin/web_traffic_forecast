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


class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.predictions = []
        self.i = 0
        self.save_every = 5000

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = model.predict(X_train)
            self.predictions.append(pred)



TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

print('Data loading...')
timeseries, dates = utils.load_snp_close()
#print("timeseries:", timeseries)
#print("dates:", dates)
dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
plt.title("APPL stock close")
plt.plot(dates, timeseries)

TRAIN_SIZE = 20
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

X, Y = utils.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = utils.create_Xt_Yt(X, Y, percentage=0.9)

Xp, Yp = utils.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
Xp, Yp = np.array(Xp), np.array(Yp)
X_trainp, X_testp, Y_trainp, Y_testp = utils.create_Xt_Yt(Xp, Yp, percentage=0.9)


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

model.fit(X_train, 
          Y_train, 
          nb_epoch=1, 
          batch_size = 128, 
          verbose=1, 
          validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=128)
print("SCORE:", score)


params = []
for xt in X_testp:
    xt = np.array(xt)
    mean_ = xt.mean()
    scale_ = xt.std()
    params.append([mean_, scale_])

predicted = model.predict(X_test)
new_predicted = []

for pred, par in zip(predicted, params):
    a = pred*par[1]
    a += par[0]
    new_predicted.append(a)
    

mse = mean_squared_error(predicted, new_predicted)
print("MSE:", mse)

try:
    fig = plt.figure()
    max_range=int(len(Y_testp))
    plt.title("Actual vs Predicted")
    #plt.plot(Y_test[:max_range], color='black',label="trained") # BLUE - trained RESULT
    #plt.plot(predicted[:max_range], color='blue',label="predicted") # RED - trained PREDICTION
    plt.plot(Y_testp[:max_range], color='green',label="actual") # GREEN - actual RESULT
    plt.plot(new_predicted[:max_range], color='red',label="restored") # ORANGE - restored PREDICTION
    plt.show()
except Exception as e:
    print(str(e))
