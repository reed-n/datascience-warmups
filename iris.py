### EXERCISE 1
### IRIS CLASSIFICATION

# import some useful things
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
import requests

# bring in the iris data itself
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')),names = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Class'])

# let's take a look at the data
print(df.head())

# Let's encode the classes
lb_make = LabelEncoder()
df['Class'] = lb_make.fit_transform(df['Class'])

# NP arrays of the dataset split into input and output
data = df.values
X = data[:,:4]
y = data[:,4:].ravel()

# Chop these into train test split - train on 75%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# We know there are 3 - let's try K-means as a simple first look
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)

# Plot colored by kmeans predicted
plt.scatter(X_test[:,0],X_test[:,2],c=kmeans.predict(X_test), s=200)
plt.ylabel('Petal Length')
plt.xlabel('Sepal Length')
plt.show()

# Can see that it clusters reasonably on this set of variables
# One clearly different set and two closer together

# Let's try ordinary support vector machine and see if we can get reasonable accuracy.
# Best practice to scale the data - 
# We will scale based on the training set (in principle we do not know the test set!)
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

#now lets make the actual support vector classifier and see how it does
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)

print('Accuracy of the SVM model is',clf.score(X_test,y_test))
# Not bad at all. Could likely be improved still by blending, a larger dataset, or cuter feature selection, or hyperparameter tweaking -
# In particular this is a little sensitive to randomness in selecting train/test data right now.

# Truthfully this dataset is simple enough to work pretty well with almost any classifier, 
# so optimizing it too much may not be a very valuable use of time:
# e.g, this popular competition winner performs comparably or worse

import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)
print('Accuracy of extreme gradient boosted model is',xgb_clf.score(X_test,y_test))