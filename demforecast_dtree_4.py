import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import copy
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression
import math
from xgboost import plot_tree



plt.close('all')

trainpercntg = .8
testpercntg = 1- trainpercntg

from tkinter.filedialog import askopenfilename
filename = askopenfilename()
df = pd.read_csv(filename)
h2 = df.shape

dff = copy.copy(df)

df.columns = ['date','value']
df.date = pd.to_datetime(df.date)
df = df.sort_values(by='date')

# removing NaT rows
n1 = df[df.date.isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2])


df["Year"] = df.date.dt.year
df["Month"] = df.date.dt.month
df["Day"] = df.date.dt.day


df = df.drop('date', 1)

X = df[['Year','Month','Day']]
Y = df[['value']]

X1 = X

from sklearn import preprocessing
def encode(x):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(x)

for column in X1.columns:
    X1[column] = encode(X1[column])
    

l1 = len(X1)
l2 = l1*trainpercntg
l2 = np.floor(l2)
l2 = int(l2)
l3 = l1-l2

Xtrain = X1[:l2]
Ytrain = Y[:l2]

Xtest = X1[l2:l1]
Ytest = Y[l2:l1]

Y1 = Y.values.tolist()
ymax = max(Y1)
    
plt.figure()
plt.plot(Y1,'-')
plt.title('Data')
plt.show()

dff.columns = ['date','value']
dff['date'] = pd.to_datetime(dff.date)
dff.set_index('date', inplace=True)


# Linear regression--------------------------


lr = LinearRegression()
lr.fit(Xtrain,Ytrain)


predlin = lr.predict(Xtest)
predlin = predlin.reshape((-1,))
predtrain = lr.predict(Xtrain)
predtrain = predtrain.reshape((-1,))

t = range(0,l3)

plt.figure()
plt.plot(t,predlin,'r', label='Predicted')
plt.plot(t,Ytest,'b', label='Observed')
plt.legend(loc='upper right')
plt.title('Linear regression')
plt.show()


dff2 = dff.iloc[l2:l1,0]
dff2.iloc[0:l3] = predlin

dff3 = dff.iloc[0:l2,0]
dff3.iloc[0:l2] = predtrain

dff1 = dff.iloc[:,0]
plt.figure()
ax = dff1.plot(label='Observed',color='#006699');
dff2.plot(ax=ax, label='Predicted',title='Linear regression', alpha=.7, color='r',linestyle='--');
dff3.plot(ax=ax, label='Train data', alpha=.7, color='m',linestyle='--');
ax.set_xlabel('Date');
ax.set_ylabel('Orders');
plt.legend(loc='upper left');
plt.show()

Ytest = Ytest.iloc[:,0]

res1 = predlin-Ytest

mse = ((res1) ** 2).mean()
rmse = math.sqrt(mse)
print('RMSE for Linear regression is', rmse)

# Decision tree---------------------------
from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(Xtrain, Ytrain)


predtest1 = clf.predict(Xtest)   
predtrain1 = clf.predict(Xtrain)    
 

t = range(0,l3)

fig=plt.figure()
plt.plot(t,predtest1,'r', label='Predicted')
plt.plot(t,Ytest,'b', label='Observed')
plt.legend(loc='upper right')
plt.title('Decision tree')
plt.show()


dff2 = dff.iloc[l2:l1,0]
dff2.iloc[0:l3] = predtest1

dff3 = dff.iloc[0:l2,0]
dff3.iloc[0:l2] = predtrain1

dff1 = dff.iloc[:,0]
plt.figure()
ax = dff1.plot(label='Observed',color='#006699');
dff2.plot(ax=ax, label='Predicted',title='Decision tree', alpha=.7, color='r',linestyle='--');
dff3.plot(ax=ax, label='Train data',title='Decision tree', alpha=.7, color='m',linestyle='--');
ax.set_xlabel('Date');
ax.set_ylabel('Orders');
plt.legend(loc='upper left');
plt.show()
    

res2 = predtest1-Ytest
mse = ((res2) ** 2).mean()
rmse2 = math.sqrt(mse)
print('RMSE for Decision tree is', rmse2)

# Xgboost-----------------------------

xgb = XGBRegressor()
xgb = xgb.fit(Xtrain, Ytrain)

predtest2 = xgb.predict(Xtest)   
predtrain2 = xgb.predict(Xtrain)    
 


fig=plt.figure()
plt.plot(t,predtest2,'r', label='Predicted')
plt.plot(t,Ytest,'b', label='Observed')
plt.legend(loc='upper right')
plt.title('Xgboost - Date as feature')
plt.show()


dff2 = dff.iloc[l2:l1,0]
dff2.iloc[0:l3] = predtest2

dff3 = dff.iloc[0:l2,0]
dff3.iloc[0:l2] = predtrain2

plt.figure()
ax = dff1.plot(label='Observed',color='#006699');
dff2.plot(ax=ax, label='Predicted',title='Xgboost - Date as feature', alpha=.7, color='r',linestyle='--');
dff3.plot(ax=ax, label='Train data', alpha=.7, color='m',linestyle='--');
ax.set_xlabel('Date');
ax.set_ylabel('Orders');
plt.legend(loc='upper left');
plt.show()


res3 = predtest2-Ytest
mse = ((res3) ** 2).mean()
rmse3 = math.sqrt(mse)
print('RMSE for Xgboost - Date as feature is', rmse3)

# adding lag as feature------------------------------------

data = pd.DataFrame(df.copy())

for i in range(1,24):
    data["lag_{}".format(i)] = df.value.shift(i)


def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


y = data.iloc[:,0]
X = data.drop('value',axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.2)


xgb1 = XGBRegressor()
xgb1.fit(X_train, y_train)

prediction = xgb1.predict(X_test)
predtrain3 = xgb1.predict(X_train)    


plt.figure()
plt.plot(prediction, "g", label="prediction", linewidth=2.0)
plt.plot(y_test.values, label="actual", linewidth=2.0)
plt.title('xgboost - date and lag as feature')


dff2 = dff.iloc[l2:l1,0]
dff2.iloc[0:l3] = prediction

dff3 = dff.iloc[0:l2,0]
dff3.iloc[0:l2] = predtrain3

plt.figure()
ax = dff1.plot(label='Observed',color='#006699');
dff2.plot(ax=ax, label='Predicted',title='Xgboost - Date and lag as feature', alpha=.7, color='r',linestyle='--');
dff3.plot(ax=ax, label='Train data', alpha=.7, color='m',linestyle='--');
ax.set_xlabel('Date');
ax.set_ylabel('Orders');
plt.legend(loc='upper left');
plt.show()

res4 = prediction-y_test
mse = ((res4) ** 2).mean()
rmse4 = math.sqrt(mse)
print('RMSE for Xgboost with date and lag is', rmse4)



# Take only lag as feature---------------------------------------

data = data.drop('Year', 1)
data = data.drop('Month', 1)
data = data.drop('Day', 1)



def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


y = data.iloc[:,0]
X = data.drop('value',axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.2)


xgb2 = XGBRegressor()
xgb2.fit(X_train, y_train)

prediction2 = xgb2.predict(X_test)
predtrain4 = xgb2.predict(X_train)    


plt.figure()
plt.plot(prediction2, "g", label="prediction", linewidth=2.0)
plt.plot(y_test.values, label="actual", linewidth=2.0)
plt.title('xgboost - lag as feature')


dff2 = dff.iloc[l2:l1,0]
dff2.iloc[0:l3] = prediction2

dff3 = dff.iloc[0:l2,0]
dff3.iloc[0:l2] = predtrain4

plt.figure()
ax = dff1.plot(label='Observed',color='#006699');
dff2.plot(ax=ax, label='Predicted',title='xgboost - lag as feature', alpha=.7, color='r',linestyle='--');
dff3.plot(ax=ax, label='Train data', alpha=.7, color='m',linestyle='--');
ax.set_xlabel('Date');
ax.set_ylabel('Orders');
plt.legend(loc='upper left');
plt.show()

res5 = prediction2-y_test
mse = ((res5) ** 2).mean()
rmse5 = math.sqrt(mse)
print('RMSE for Xgboost with lag feature only is', rmse5)

t = range(0,l3)
plt.figure()
plt.plot(t,abs(res1),'r', label='Linear regression')
plt.plot(t,abs(res2),'b', label='Decision tree')
plt.plot(t,abs(res3),'m', label='Xgboost - date as feature')
plt.plot(t,abs(res4),'c', label='Xgboost- date and lag as feature')
plt.plot(t,abs(res5),'g', label='Xgboost- lag only')
plt.xlabel('time')
plt.ylabel('Error value')
plt.legend(loc='upper right')
plt.title('Residual plot')
plt.show()


