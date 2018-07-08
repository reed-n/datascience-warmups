### EXERCISE 4
### BOSTON HOUSING

# import some useful things
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt      
import seaborn as sns  
import warnings                        
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# bring in the dataset from https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data

boston=pd.read_csv('housing.data',delim_whitespace=True, header=None)

#numpy arrays of X and y in the usual notation

X = boston.values[:,:13]
y = boston.values[:,13]

# we'll (somewhat arbitrarily) choose to approach this problem through a neural network
# the framework here is Keras, which prototypes networks easily and uses a tensorflow backend
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# construct a very simple model with one set of nodes equal to the number of input features
def network():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam')
	return model

# define the estimator 
estimator = KerasRegressor(build_fn=network, epochs=100, batch_size=5, verbose=0)

# kfold cross validation is good practice to check if our model is robust to unseen input data
# we will use RMSE as our evaluation metric in this case.
kf = KFold(n_splits=5)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Naive Model Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))

# RMSE of about 2 is pretty good for such a simple model! 
# one thing we didnt do here is normalize the feature data at all - does this help?
# (it is good practice to do this almost always)

X_backup = X
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Normalized Feature Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))

# looks like we got some modest improvement there. 
# what about different network topologies? can we get improvements from network depths?? a wide network?

def deep_network():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=deep_network, epochs=100, batch_size=5, verbose=0)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Deep Network Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))

def deeper_network():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))	
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam')
	return model	

estimator = KerasRegressor(build_fn=deeper_network, epochs=100, batch_size=5, verbose=0)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Deeper Network Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))
	
def wide_network():
	# create model
	model = Sequential()
	model.add(Dense(26, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=wide_network, epochs=100, batch_size=5, verbose=0)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Wide Network Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))

# it looks like we got the most improvement from adding another layer of hidden units, and not much beyond that.
# we could combine/tinker with this basically forever, and add things like dropouts to try and dodge overfitting, e.g.

from keras.layers import Dropout
def fancy_network():
	# create model
	model = Sequential()
	model.add(Dense(26, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(13, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=fancy_network, epochs=100, batch_size=5, verbose=0)
rmses = np.sqrt(-1*cross_val_score(estimator, X, y, cv=kf))
print("Wide/deep with dropout Network Results: %.2f mean (%.2f) stdev RMSE" % (rmses.mean(), rmses.std()))

# we've improved our error once again (noting that to be rigorous about this, the random seed should be fixed),
# but the improvements are starting to become more marginal. At this stage things like feature engineering may be 
# a better use of time than tuning network topology itself.

# as a final check, did we really need to do all this at all? could we have achieved just as good of results
# in a more straightforward way? Try a common contest winner, extreme gradient boost, which typically has excellent results

import xgboost as xgb
xg = xgb.XGBRegressor()
scores = cross_val_score(xg, X, y, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-scores)
print('Mean and stdev of XGB RMSE scores are:', rmse_scores.mean(),'and',rmse_scores.std())

# looks like our network was a little more work to set up, but significantly outperformed this
# XGB model. this is not a very strong statement, as the XGB hasn't been tuned or optimised in any way,
# but it is nice to know we couldnt have done better with one line of code :)