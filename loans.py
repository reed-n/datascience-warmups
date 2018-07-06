# -*- coding: utf-8 -*-
### EXERCISE 2
### LOAN CLASSIFICATION

# import some useful things
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt      
import seaborn as sns  
import warnings                        
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# bring the train/test sets into pandas dataframes - 
# these can be acquired from 
# https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

train=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

# take a quick peek at what kind of data we will be working with here
print(train.head())

# drop the loan ID from the model, we won't need that to train any model
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

# set up a map to fix the categorical data
mapping = {'Y':1, 'N':0, 'Urban':2,'Semiurban':1,'Rural':0,'Male':1,'Female':0,'Yes':1,'No':0,'Graduate':1,'Not Graduate':0,'3+':3}
train.replace(mapping,inplace=True)
test.replace(mapping,inplace=True)

# take a look at the correlation matrix between various parameters
correlations = train.corr()
sns.heatmap(correlations, vmax=1, square=True, cmap="jet");
plt.show()

# we can see from this that credit history is the strongest correlator by far.
# there are some missing values in the dataset - first we will see how we do by just dropping them
# later we can see if making some assumptions can improve the result
train_dropped = train.dropna()
test_dropped = test.dropna()

# loan amount skews right - one way to help this is to work with log(n) instead
train_dropped['LoanAmount_log'] = np.log(train_dropped['LoanAmount'])
test_dropped['LoanAmount_log'] = np.log(test_dropped['LoanAmount'])
train_dropped, test_dropped = train_dropped.drop('LoanAmount',axis=1), test_dropped.drop('LoanAmount',axis=1)


# get into the standard format that we usually use with sklearn
X, y = train_dropped.drop('Loan_Status',axis=1), train_dropped.Loan_Status
# clean up the format
X = X.astype(float)


x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.25)

# scale our the data so features live in the same rough magnitude
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_cv = scaler.transform(x_cv)

# alright, as a baseline let's train a couple of simple models and see how they do.
# We'll check with k-fold cross validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logi = LogisticRegression()
svm = SVC(kernel='rbf')
forest = RandomForestClassifier()
xg = xgb.XGBClassifier()


logi.fit(x_train,y_train)
svm.fit(x_train,y_train)
forest.fit(x_train,y_train)
xg.fit(x_train,y_train)

#accuracies
print('Logistic:',logi.score(x_cv,y_cv))
print('SVM:',svm.score(x_cv,y_cv))
print('Forest:',forest.score(x_cv,y_cv))
print('XGB:',xg.score(x_cv,y_cv))

#we can also get cute and try to achieve better accuracy with a Neural Network, e.g.
from keras import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# simple network with one hidden layer that has as many neurons as features
# create model
model = Sequential()
model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=100,
                    verbose=1,
                    validation_data=(x_cv, y_cv))
                    

                    
# we see that for even an extremely simple network, the accuracy is not too bad.
# we could (and should) do k-fold cross validation here to get a reliable and good accuracy measurement -
# this just demonstrates that very simple models can achieve pretty good performance.
# in fact, in this case, logistic regression and svm often perform better than a fancy network!

# earlier we mentioned that we would just drop missing values from the dataset and train only on complete
# input vectors. Now, let's see if we can improve our predictive accuracy by making some assumptions

train_backup = train;
test_backup = test;

# fill various NaN with modes and medians
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# fuss with loan amount skew again, there are other ways to do this and other variables
# we could do it to 
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])
train, test = train.drop('LoanAmount',axis=1), test.drop('LoanAmount',axis=1)

#split into x and y 
X = train.drop('Loan_Status',1)
y = train.Loan_Status
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#mentioned cross validation earlier - let's actually implement it here
#we will scale the data inside the function
from sklearn.model_selection import StratifiedKFold

def kfold(model, nsplits, X, y):
    accur = np.zeros(5)
    i = 1
    kf = StratifiedKFold(n_splits=nsplits,shuffle=True)
    for train_index,test_index in kf.split(X,y):
        xtr,xvl = X.loc[train_index],X.loc[test_index]
        ytr,yvl = y[train_index],y[test_index]
        
        scaler = StandardScaler()
        scaler.fit(xtr)
        xtr = scaler.transform(xtr)
        xvl = scaler.transform(xvl)

    
        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = model.score(xvl,yvl)
        accur[i-1] = score
        i+=1

    print('Accuracy/standard dev of k-fold validated model are:', np.mean(accur),',', np.std(accur))
 #   pred_test = model.predict(test)
 #   pred=model.predict_proba(xvl)[:,1]
    
kfold(logi,5,X,y)

#logistic regression seems to do the best here, but it is not totally clear that 
#filling the missing values actually gave us a good boost in performance after all.
#again, there is more feature engineering that could probably be done here and 
#hyperparameter tuning could offer some improvements.
