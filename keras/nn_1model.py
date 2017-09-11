import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
#import matplotlib.pylab as plt
import datetime as dt
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
import random


def smape_fast(y_true, y_pred):
       epsilon = 0.01  # almost no visit ;-)
       y_true=pd.DataFrame(y_true)
       y_pred=pd.DataFrame(y_pred)
       assert y_true.shape[1]==1
       assert y_pred.shape[1]==1
       df=pd.concat([y_true, y_pred], axis=1)
       df.columns=['true', 'pred']
       df['sum']=df['true']+df['pred']
       df['diff']=df['true']-df['pred']
       df['diff']=pd.DataFrame.abs(df['diff'])
       if df['sum'].sum() == 0:
           df['sum'] = epsilon  # avoid nan

       df['smape_base']=df['diff']/df['sum']
       out=df['smape_base'].sum()
       out*= (200/y_true.shape[0])
       return out




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

def explore():
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





def run_nn(train_1):
    class_0='2017-08-31' # just model the first one
    #print("TTTT", train_1)
    labels=train_1[train_1.columns]
    labels = labels[class_0]
    visits_train = train_1.values
    visits_labels = labels.values
    
    X_train = visits_train
    Y_train = visits_labels
    #print("X_train.shape:", X_train.shape)
    Pages=X_train.shape[0]
    Dates=X_train.shape[1]
    
    print('Building model... pages:', Pages,", Dates:", Dates)
    model = Sequential()
    model.add(Dense(500, input_shape = (Dates,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mse')
    
    Y_test = Y_train
    X_test = X_train
    
    epochs=100
    model.fit(X_train,
              Y_train,
              nb_epoch=epochs,
              batch_size = 512,
              verbose=1, validation_split=0.1)
    #score = model.evaluate(X_test, Y_test, batch_size=128)
    #print("SCORE:", score)
    
    predicted = model.predict(X_test)
    if predicted[0] < 0:
       predicted[0] = 0
    #print("predicted:", predicted)
    
    #mse = mean_squared_error(predicted, Y_train)
    #print("MSE:", mse)
    smape = smape_fast(predicted, Y_train)
    #print("predicted:", predicted)
    #print("X_test:", X_test)
    #print("Y_train:", Y_train)
    print("SMAPE:", smape)
    #fig = plt.figure()
    #plt.title("Actual vs Predicted",Page, "SMAPE:", str(0.1* int(float(smape)*10.0) ) )
    #plt.title("Actual vs Predicted",Page)
    #plt.scatter(Y_train,predicted, color='red')
    #plt.show()
    #TBD predicted = page+visits
    print("predicted:", predicted)
    return predicted



train = pd.read_csv("../input/train_2.csv")
train = train.fillna(0.)
print("Train shape 0:", train.shape)
pages=train['Page'].values
print("pages:", len(pages))
max_days_back=790

train = train[train.columns[-max_days_back:]]
print("Train shape:", train.shape)
predicted_visits0 = run_nn(train)
print("predicted_visits0:", predicted_visits0)
predicted_visits = [i[0] for i in predicted_visits0]
print("predicted_visits", predicted_visits)
print("predicted_visits len", len(predicted_visits))

visits_df = pd.DataFrame( {'Page': pages, 'Visits': predicted_visits} )
visits_df.to_csv('visits_df.csv')

test1 = pd.read_csv("../input/key_2.csv")
test1['Page'] = test1.Page.apply(lambda x: x[:-11])
test1 = test1.merge(visits_df[['Page','Visits']], on='Page', how='left')
test1[['Id','Visits']].to_csv('sub_nn1.csv', index=False)
