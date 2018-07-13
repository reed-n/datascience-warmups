### EXERCISE 6
### WINE
# import some useful things
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt      
import seaborn as sns  
import warnings                        
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import the data - your path here
red = pd.read_csv('winequality-red.csv',sep=';')
white = pd.read_csv('winequality-white.csv',sep=';')

# head one of them so we can see what they look like
print(red.head())

# ok, look for missing values first off
print(red.isnull().sum())
print(white.isnull().sum())

# looks like neither of these is missing any values.
# good news, everyone!
# I'll do 2 tasks with this dataset. first, I will  run regression
# to try and determine wine quality.
# secondly, I'll combine the data into one big set and see if 
# we can do predictions on whether a given wine is red or white!

#   =====================   #
### REGRESSION ON QUALITY ###
#   =====================   #

# backup old red and white for later use and so we don't lose the data
red_old = red
white_old = white

# split into some pretty standard X,y nomenclature
Xred = red.drop(['quality'],axis=1)
yred = red.quality
Xwh = white.drop(['quality'],axis=1)
ywh = white.quality

# let's do some exploratory data analysis. seeing distributions is nice.
# red first
fig = plt.figure(1)
count = 1
for i in list(Xred):
    fig.add_subplot(11,1,count)
    sns.distplot(Xred[i]);

#    plt.subplot(122)
#    red[i].plot.box(figsize=(16,5))
    count = count + 1
plt.suptitle('Red Wine Feature Distributions', fontsize=12)
plt.show()

# and white
fig = plt.figure(2)
count = 1
for i in list(Xwh):
    fig.add_subplot(11,1,count)
    sns.distplot(Xwh[i]);

#    plt.subplot(122)
#    red[i].plot.box(figsize=(16,5))
    count = count + 1
plt.suptitle('White Wine Feature Distributions', fontsize=12)
plt.show()


# features 6,7,10,11 skew right on red., slightly different on white.
#it may be to our benefit to do some feature engineering here
# to get them closer to normally distributed. we will use a log transform.
for i in ['free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']:
    Xred[i] = np.log(Xred[i])

for i in ['residual sugar', 'total sulfur dioxide', 'density','sulphates', 'alcohol']:
    Xwh[i] = np.log(Xwh[i])

# take a sneak peek
plt.figure(3)
sns.distplot(Xwh['density'])
plt.title('Log Density to help with right tail')
plt.show()

# ok. it is good practice for us to normalize our input features so they live
# in the same range. let's do that now.

scaler = StandardScaler()
Xred = scaler.fit_transform(Xred)
Xwh = scaler.fit_transform(Xwh)

# great. regression models! some classic ones
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# kfold cross validation is one principled way to help dodge overfitting and 
# improve robustness to unseen data
def crossval(clf,kfolds,X_Train,Y_Train):
    scores = cross_val_score(clf, X_Train, Y_Train, scoring="neg_mean_squared_error", cv=kfolds)
    rmse_scores = np.sqrt(-scores)
    print('Mean and stdev of rmse scores for %.6s are:%.5f %.5f' % (clf,rmse_scores.mean(),rmse_scores.std()))
    
# ok here we go - red first arbitrarily, then white
print('Red Wine Regression Scores')
for models in [LinearRegression(), Lasso(), Ridge(), ElasticNet(), RandomForestRegressor(), xgb.XGBRegressor()]:
    clf = models
    crossval(clf,10,Xred,yred)

print('White Wine Regression Scores')
for models in [LinearRegression(), Lasso(), Ridge(), ElasticNet(), RandomForestRegressor(), xgb.XGBRegressor()]:
    clf = models
    crossval(clf,10,Xwh,ywh)
    
# we see that in general, our algorithms had a harder time with the white wines than with the reds.
# additionally, it looks like naked xgboost does the best in both cases - 
# although linear regression, ridge models, and random forests are all also pretty good.
# we could do some hyperparameter tuning to try and improve this, but it is ok for now.
# a quick (NON RIGOROUS) scatterplot - i won't even train/test split this
clf = xgb.XGBRegressor()
clf.fit(Xwh,ywh)
plt.figure(4)
plt.scatter(clf.predict(Xwh),ywh)
plt.plot([2,10],[2,10],'g-')
plt.xlabel('Predicted quality')
plt.ylabel('Real quality')
plt.title('White Wine Predictions')
plt.show()

# it seems that one problem we are having is that the algorithm does not predict actually highly rated wines very well.

#   =====================   #
### REGRESSION ON QUALITY ###
#   =====================   #

# can i tell you if a wine is red or white just based on these attributes?
# reset the dataframes
red = red_old
white = white_old

#let's add some columns to red and white to indicate what type they are
red['color'] = 0;
white['color'] = 1;

# and combine into new frame.
wines = red.append(white)

# split into X,y. this time i will leave quality in as a predictor - 
# maybe knowing how good a wine is helps you know the color!
# this fits my hypothesis of personally thinking reds are much better, although
# according to this dataset they have slightly lower mean quality.
X = wines.drop(['color'],axis=1)
y = wines.color
print('Mean red quality and white quality are ',np.mean(red.quality),np.mean(white.quality))

#lets look at the feature histograms once more.
fig = plt.figure(5)
count = 1
for i in list(X):
    fig.add_subplot(12,1,count)
    sns.distplot(X[i]);

#    plt.subplot(122)
#    red[i].plot.box(figsize=(16,5))
    count = count + 1
plt.suptitle('All Wine Feature Distributions', fontsize=12)
plt.show()

# help take care of some obvious right skew, for one
for i in ['volatile acidity','chlorides','free sulfur dioxide','alcohol']:
    X[i] = np.log(X[i])

# we will do some stratified kfold here

from sklearn.model_selection import StratifiedKFold

def kfold(model, nsplits, X, y):
    accur = np.zeros(nsplits)
    i = 1
    X = X.values
    y = y.values
    kf = StratifiedKFold(n_splits=nsplits,shuffle=True)
    for train_index,test_index in kf.split(X,y):
        xtr,xvl = X[train_index,:],X[test_index,:]
        ytr,yvl = y[train_index],y[test_index]
        
        scaler = StandardScaler()
        scaler.fit(xtr)
        xtr = scaler.transform(xtr)
        xvl = scaler.transform(xvl)

        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = model.score(xvl,yvl)
# this in for debugging
#       print(np.nonzero(pred_test-yvl))
        accur[i-1] = score
        i+=1

    print('Accuracy/standard dev of %.2f -fold validated model are:' % nsplits, np.mean(accur),',', np.std(accur))
    
# import some pretty standard classifier models here
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# same shtick as above more or less
for models in [LogisticRegression(),SVC(kernel='rbf'),RandomForestClassifier(),xgb.XGBClassifier()]:
    clf = models
    print('%.3s Model' % models)
    kfold(models,10,X,y)
    
# everything performs shockingly well. high quality data minimizes the impact of any given classifier!
# as a final fun thing, let's extract feature importances from one of these classifiers.

#train test, normalize features
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_cv = scaler.transform(x_cv)

# classifier
clf = xgb.XGBClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_cv,y_cv))
importances = clf.feature_importances_
print('Feature importances vector to classify wine as red or white is',importances)

# looks like the most important features that distinguish between red and white are
# total sulfur dioxide and chlorides (for this classifier, in any case.)
# Random Forest Classifier turns out to tell the same story for importances!
