### EXERCISE 5
### TRAFFIC TIMESERIES

# import some useful things
import pandas as pd
import numpy as np     
import scipy as sp 
import math           
import matplotlib.pyplot as plt      
import seaborn as sns  
import warnings                        
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# bring the train/test sets into pandas dataframes - 
# these can be acquired from 
# https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

train=pd.read_csv("Train_SU63ISt.csv")
test=pd.read_csv("Test_0qrQsBZ.csv")

df = train

# looks sort of exponential with clear large variations.
# there are many ways we can try to predict timeseries - 
# here we will go for a neural network approach using LSTM.

# import some things we will want.

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# get the variable to be modeled, scale it into range 0-1 for the LSTM

data = df["Count"].values
data = data.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

# train/test - normally we would cross validate, but since this is a timeseries
# we want to preserve the order of the variables

train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
print('Train and test sizes are',len(train),'and', len(test))

#function to get data from t and t-(look back time) for the LSTM
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
	
#apply to our data
# reshape into X=t and Y=t+1

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# the LSTM needs the data to be formatted in a certain way that includes time step info

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# the LSTM network itself. this is tunable

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

# this takes forever to train with batch size 1, so we will train with a large batch size and then
# switch the learned model weights onto a new network with batch size 1 to predict.
# even still, training this isn't super fast on my laptop.

# network params

n_batch = 250
n_epoch = 100
n_neurons = 8
prop = optimizers.rmsprop(lr=0.001, decay=1e-7)

# network architecture

model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, trainX.shape[1], trainX.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=prop)

# fit

for i in range(n_epoch):
	model.fit(trainX, trainY, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	
# re-define the batch size

n_batch = 1

# re-define model

new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, trainX.shape[1], trainX.shape[2]), stateful=True))
new_model.add(Dense(1))

# copy weights

old_weights = model.get_weights()
new_model.set_weights(old_weights)
new_model.compile(loss='mean_squared_error', optimizer=prop)

# predictions

trainPredict = new_model.predict(trainX,batch_size=1)
testPredict = new_model.predict(testX,batch_size=1)

# invert them into original scale (traffic counts) instead of 0 to 1

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# compute and output RMSE

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# finally, let's plot predictions and data together
# shift train predictions for plotting

trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# when I run this, im getting a train/test set RMSE of about 28/100, respectively,
# and a plot that looks pretty reasonable.
# we do appear to be under-capturing the big peaks in the testing portion of the dataset -
# very thorough hyperparameter tuning might be able to help here.