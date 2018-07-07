### EXERCISE 3
### BIGMART SALES

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
# https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/

train=pd.read_csv("Train_UWu5bXk.csv")
test=pd.read_csv("Test_u94Q5KV.csv")

# as usual, take a peek at the data
print(train.head(5))

# look for missing values in here
print(train.isnull().sum())

# seems as though everything is in place except for some item weights and outlet sizes.
# how are these distributed?

plt.figure(1)
plt.subplot(121)
sns.distplot(train['Item_Weight'].dropna());

plt.subplot(122)
train['Item_Weight'].plot.box(figsize=(16,5))

plt.show()

#item weight appears to basically have a flat distribution between 5 and 20

plt.figure(2)
train['Outlet_Size'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Outlet Size')

plt.show()

# and outlet size appears to be mostly medium/small and slightly less high.
# let's fill these in a simple way that seems to make some sense - otherwise we are dropping a lot 
# of training data. it turns out that item visibility also has many zeros that are effectively NaN

train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)
train=train.replace({'Item_Visibility': {0: train['Item_Visibility'].mean()}}) 

#also it may be interesting to see what the actual distribution of outlet sales looks like.
plt.figure(3)
plt.subplot(121)
sns.distplot(train['Item_Outlet_Sales'].dropna());

plt.subplot(122)
train['Item_Outlet_Sales'].plot.box(figsize=(16,5))

plt.show()

# looks like it skews right - there is some small number of items that have much higher sales.
# peek at the largest for fun, as an example
print(train.loc[train['Item_Outlet_Sales'].argmax()])

# this item turns out to be labeled NCE42
print(train.loc[train['Item_Identifier'] == 'NCE42'])
# the one huge sale appears to be pretty anomalous for this particular product,
# whose mean appears to resemble the overall sales mean

# split into the usual x,y format that is commonly used
X = train.drop('Item_Outlet_Sales',1)
y = train.Item_Outlet_Sales
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

# scaling is good - we will make a backup first, in any case
X_backup = X
y_backup = y
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# ok. in any case, we have a multivariate regression problem on our hands now.

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# add a column of 1's for x0 in the regression
a = (np.ones(len(X)).reshape(len(X),1))
X = np.concatenate((a,X),axis=1)

# linear regress first, straightforward
linreg = LinearRegression()
linreg.fit(X,y)

# lets create an RMSE function so we can see how our predictions are performing 
# on the test set
def rmse(model,x,y):
    p = model.predict(x)
    err = abs(p-y)
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))
    return(rmse_train)

# a quick note on why k-fold cross validation and train/test are so important:
p = linreg.predict(X)
plt.figure(4)
plt.subplot(111)
plt.scatter(y,p)
plt.plot([0,12000],[0,12000],'g-')
plt.show

print(rmse(linreg,X,y))

# this is ok. we can try some other linear classifiers
# Create linear regression object with a ridge coefficient 0.5
ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(X,y)
lasso = Lasso(fit_intercept=True,alpha=0.5)
lasso.fit(X,y)
net = ElasticNet(fit_intercept=True, alpha=0.5)
net.fit(X,y)
rf = RandomForestRegressor()
rf.fit(X,y)
xg = xgb.XGBRegressor()
xg.fit(X,y)

# now list out the RMSE
for model in linreg,ridge,lasso,net,rf,xg:
    print(model,'RMSE is:',rmse(model,X,y))
    
# random forest performs suspiciously well on the entire dataset.
# under half the root mean square error of any other model!
# intuitively, this is almost certainly a victim of heavy overfitting - 
# we can see this via, e.g.,

overfittest = RandomForestRegressor()
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
overfittest.fit(x_train,y_train)
print('RMSE of train/test split is on Random Forest is',rmse(overfittest,x_test,y_test))

# our random forest was trained to do extremely well on the whole set, but performs very badly 
# upon the introduction of unseen data. this would be ruinous for say, a leaderboard score,
# or worse, real application! let's do this a bit better (note this is pretty slow on my laptop)

from sklearn.model_selection import cross_val_score
def crossval(clf,kfolds,X_Train,Y_Train):
    scores = cross_val_score(clf, X_Train, Y_Train, scoring="neg_mean_squared_error", cv=kfolds)
    rmse_scores = np.sqrt(-scores)
    print('Mean and stdev of rmse scores are:', rmse_scores.mean(),rmse_scores.std())

for mods in ridge, lasso, rf, xg:
    crossval(mods,5,X,y)

# so our simple xgboost model performs consistently the best by a considerable margin
# (although it is slow to train.)
# what about further feature engineering? can we improve performance in this way?
# we backed up our initial set of features, we will just overwrite here

X = X_backup
y = y_backup
# hypothesis:
# maybe item weight isnt so important, and what we really care about is mrp per weight, for example.
X['Item_MRP_per'] = X['Item_MRP']/X['Item_Weight']

X = X.drop(['Item_MRP'],axis=1)
X = X.drop(['Item_Weight'],axis=1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#lets see if this improves our regression at all. use random forest, just because it trains faster.
crossval(rf,5,X,y)

# it looks like in this case, we actually hurt our performance a little bit!! our hypothesis 
# did not pan out for this model.
# we now would loop back and reevaluate assumptions and construct a different working hypothesis at this point,
# and iterate testing these hypotheses until we came across something good.