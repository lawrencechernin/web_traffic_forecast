import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

def plotter(scaler,look_back,dataset,trainPredict,testPredict,Page):
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.title(Page)
    plt.plot(scaler.inverse_transform(dataset),c="black")
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    print("testPredictPlot", type(testPredictPlot))
    print("testPredictPlot", testPredictPlot)
    print("testPredict:", testPredict)
    print("trainPredict:", trainPredict)
    plt.show()

def smape_fast(y_true, y_pred):
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


def run_LTSM(dataset,Page):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    test_size = 60
    # split into train and test sets
    train_size = int(len(dataset) - test_size )
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1 #TEST smape = 47.4
    look_back = 2 #TEST smape = 40.9
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    #model.add(LSTM(8, input_shape=(1, look_back))) #TEST smape = 41.49
    model.add(LSTM(4, input_shape=(1, look_back))) #TEST smape = 40.9
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(optimizer='adam', loss='mape') #terrible
    
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) # TEST smape = 40.2
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    #trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    #smape = smape_fast(trainY[0], trainPredict[:,0])
    #print("SMAPE train:", smape)
    #testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    smape = smape_fast(testY[0],testPredict[:,0])
    print("SMAPE on test: [",Page,"]", smape)
    plotter(scaler,look_back,dataset,trainPredict,testPredict,Page)

######################## MAIN PROGRAM ##########################

train = pd.read_csv("../input/train_1_1000.csv")
train = train.fillna(0.)
pages=train['Page'].values
pages=['必娶女人_zh.wikipedia.org_all-access_spider']
for Page in pages:
    train_1 = train[train['Page']==Page]
    
    del train_1["Page"]
    x=train_1.columns
    x=list(x)
    y=list(train_1.values[0])
    #dataframe = read_csv('p1.csv', usecols=[1], engine='python')
    dataframe = pd.DataFrame(y, columns=['visits'])
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    #print("DATASET:", dataset)
    run_LTSM(dataset,Page)
    
        
