import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
from datetime import datetime
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





#set up the data sequences for training and predicting
start_predict_date = '2017-01-01'
end_predict_date = '2017-03-01'
date_format = "%Y-%m-%d"
a = datetime.strptime(start_predict_date, date_format)
b = datetime.strptime(end_predict_date, date_format)
delta = b - a
days_to_predict = int( delta.days + 1)
print("Predicting visits starting on ", start_predict_date, "and ending on ", end_predict_date)
print("We need make ", days_to_predict, " days of predictions after the historical period")

#train_x_y_split(train)

import keras.backend as K

def smape(y_true, y_pred): #to run inside Keras
    print("y_t", y_true)
    print("y_p", y_pred)
    y_true = list(y_true)
    y_pred = list(y_pred)
    denominator = (y_true + y_pred) / 200.0
    diff = K.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return K.mean(diff)

def smape_fast0(y_true, y_pred):
       
       print("y_t", y_true)
       print("y_p", y_pred)
       y_true=pd.DataFrame(y_true)
       y_pred=pd.DataFrame(y_pred)
       assert y_true.shape[1]==1
       assert y_pred.shape[1]==1
       df=pd.concat([y_true, y_pred], axis=1)
       df.columns=['true', 'pred']
       df['sum']=df['true']+df['pred']
       df['diff']=df['true']-df['pred']
       df['diff']=pd.DataFrame.abs(df['diff'])
       df['smape_base']=df['diff']/df['sum']
       out=df['smape_base'].sum()
       out*= (200/y_true.shape[0])
       return out

def explore(): # try this to work on small subset of data
    #train = pd.read_csv("../input/train_1.csv")
    train = pd.read_csv("../input/train_1_10.csv")
    train = train.fillna(0.)
    print("Train head:", train.head())
    print("Full size of training:", train.shape)
    # now we take a subset...
    Page="2NE1_zh.wikipedia.org_all-access_spider"
    train_1 = train[train['Page']==Page]
    labels=train_1[train_1.columns[-days_to_predict:]]
    train_1=train_1[train_1.columns[:-days_to_predict]]
    print("labels:",labels.shape)
    print("train:", train_1.shape)
    print("type:", type(train_1))
    print("size:", train_1)
    del train_1["Page"]
    visits_train = train_1.values[0]
    dates_train = train_1.columns
    dates_train = [ datetime.strptime(x, '%Y-%m-%d')  for x in dates_train]
    visits_labels = labels.values[0]
    dates_labels=labels.columns
    dates_labels = [ datetime.strptime(x, '%Y-%m-%d')  for x in dates_labels]
    fig = plt.figure()
    plt.title(Page+'training in red and labels in blue')
    plt.plot(dates_train,visits_train, color='red')
    plt.plot(dates_labels,visits_labels, color='blue')
    plt.show()

#explore()


### ok, now for real

#train = pd.read_csv("../input/train_1_100.csv")
train = pd.read_csv("../input/train_1.csv")
train = train.fillna(0.)
train_1 = train
class_0='2016-11-02' # just model the first date and see what smape is, later we can make models for each date
labels=train_1[train_1.columns[-days_to_predict:]]
labels = labels[class_0]
train_1=train_1[train_1.columns[:-days_to_predict]]
print("labels:",labels.shape)
print("train:", train_1.shape)
print("type:", type(train_1))
print("train_1:", train_1)
del train_1["Page"]
visits_train = train_1.values
dates_train = train_1.columns
dates_train = [ datetime.strptime(x, '%Y-%m-%d')  for x in dates_train]
visits_labels = labels.values
#lets just predict the first label
print("visits_train:", visits_train)

X_train = visits_train
Y_train = labels
print("X_train.shape:", X_train.shape)
Pages=X_train.shape[0]
Dates=X_train.shape[1]
print("X_train:", X_train)

print('Building model...')
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(X_train, look_back)
testX, testY = create_dataset(X_train, look_back)

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')



#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',smape])
#model.compile(optimizer='rmsprop', loss='mape')
#model.compile(optimizer='adam', loss='mse')
#model.compile(optimizer='rmsprop',loss='mape')




epochs=10
model.fit(trainX,
          trainY,
          nb_epoch=epochs,
          batch_size = 128,
          verbose=1,
          validation_split=0.1)
score = model.evaluate(testX, testY, batch_size=128)
print("MODEL SCORE:", score)



predicted = model.predict(testX)
print("predicted:", predicted)

smape = smape(predicted, trainY)
print("SMAPE:", smape)
#fig = plt.figure()
f, ax = plt.subplots(2)

ax[0].set_title("Actual vs Predicted")
ax[0].set(adjustable='box-forced', aspect='equal')
ax[0].scatter(trainY,predicted, color='red',s=2,marker='+')
ax[0].set_xlabel('Y_train')
ax[0].set_ylabel('predicted')
xlim = [0,np.max(Y_train)]
ylim=xlim
diag_line, = ax[0].plot(xlim, ylim, ls="--", c=".3")
def on_change(axes):
    # When this function is called it checks the current
    # values of xlim and ylim and modifies diag_line
    # accordingly.
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    diag_line.set_data(x_lims, y_lims)

# Connect two callbacks to your axis instance.
# These will call the function "on_change" whenever
# xlim or ylim is changed.
ax[0].callbacks.connect('xlim_changed', on_change)
ax[0].callbacks.connect('ylim_changed', on_change)

# zoomed plot
ax[1].set_title("Actual vs Predicted zoomed")
ax[1].set(adjustable='box-forced', aspect='equal')
ax[1].scatter(trainY,predicted, color='blue',s=1,marker='+')
xrang=[-1,25]
yrang=[-1,25]
ax[1].set_xlim(xrang)
ax[1].set_ylim(yrang)
ax[1].set_xlabel('Y_train')
ax[1].set_ylabel('predicted')
xlim = [0,max(xrang)]
ylim=xlim
diag_line, = ax[1].plot(xlim,ylim, ls="--", c=".2")
ax[1].callbacks.connect('xlim_changed', on_change)
ax[1].callbacks.connect('ylim_changed', on_change)


#plt.show()






