import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import math 
from xgboost import plot_tree

plt.close('all')

trainpercntg = .8

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

from tkinter.filedialog import askopenfilename
filename = askopenfilename()
df = pd.read_csv(filename)
df = df[:5000]
h = df.shape
h1 = h[0]
h2 = h[1]

# removing NaT rows
n1 = df[df.Date.isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2])

df['Date'] = pd.to_datetime(df.Date)
df.drop_duplicates(inplace=True)
df = df.sort_values(by='Date')


X = df.iloc[:,0:h2-1]
Y = df.iloc[:,h2-1]


#X["Year"] = X.Date.dt.year
X["Month"] = X.Date.dt.month
X["Day"] = X.Date.dt.day


X = X.drop('Date', 1)

X1 = X.copy()

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
Y2 = Ytest.values.tolist()
    
plt.figure()
plt.plot(Y1,'-')
plt.title('Original data')
plt.xlabel('Time')
plt.ylabel('Order demand')


plt.figure()
plt.plot(Y2,'-')
plt.title('Test data')
plt.xlabel('Time')
plt.ylabel('Order demand')


# Decision tree---------------------------------------
from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(Xtrain, Ytrain)

predtest1 = clf.predict(Xtest)
predtrain1 = clf.predict(Xtrain)


t = range(0,l3)

plt.figure()
plt.plot(t,Ytest,'b',label = 'Original')
plt.plot(t, predtest1,'r',label = 'prediction') 
plt.legend(loc='upper left')
plt.title('Decision tree - Test data prediction')
plt.xlabel('Time')
plt.ylabel('Order demand')
plt.show()


t = range(0,l2)

plt.figure()
plt.plot(t,Ytrain,'b',label = 'Original')
plt.plot(t, predtrain1,'r',label = 'prediction') 
plt.legend(loc='upper left')
plt.title('Decision tree - Train data prediction')
plt.xlabel('Time')
plt.ylabel('Order demand')
plt.show()

Ytest = np.float64(Ytest)
Ytest = Ytest.reshape((-1,))
res1 = predtest1-Ytest

mse = ((res1) ** 2).mean()
rmse = math.sqrt(mse)
print('RMSE for Decision tree is', rmse)

## Calculate mean absolute percentage error (MAPE)
#res1 = abs(res1)
#mape = 100 * (res1/Ytest)
#
## Calculate and display accuracy
#accuracy1 = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy1, 2), '%.')


# Xgboost--------------------------------

xgb = XGBRegressor()
xgb.fit(Xtrain, Ytrain)


predtest2 = xgb.predict(Xtest)
predtrain2 = xgb.predict(Xtrain)


t = range(0,l3)

plt.figure()
plt.plot(t,Ytest,'b',label = 'Original')
plt.plot(t, predtest2,'r',label = 'prediction') 
plt.legend(loc='upper left')
plt.title('Xgboost - Test data prediction')
plt.xlabel('Time')
plt.ylabel('Order demand')
plt.show()


t = range(0,100)

plt.figure()
plt.plot(t,Ytest[800:900],'b',label = 'Original')
plt.plot(t, predtest2[800:900],'r',label = 'prediction') 
plt.legend(loc='upper left')
plt.title('Xgboost - Test data prediction')
plt.xlabel('Time')
plt.ylabel('Order demand')
plt.show()


t = range(0,l2)

plt.figure()
plt.plot(t,Ytrain,'b',label = 'Original')
plt.plot(t, predtrain2,'r',label = 'prediction') 
plt.legend(loc='upper left')
plt.title('Xgboost - Train data prediction')
plt.xlabel('Time')
plt.ylabel('Order demand')
plt.show()


res2 = predtest2-Ytest

mse = ((res2) ** 2).mean()
rmse1 = math.sqrt(mse)
print('RMSE for Xgboost is', rmse1)

# Calculate mean absolute percentage error (MAPE)
#res2 = abs(res2)
#mape = 100 * (res2/Ytest)
#
## Calculate and display accuracy
#accuracy2 = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy2, 2), '%.')


t = range(0,l3)
plt.figure()
plt.plot(t,abs(res1),'r', label='Decision tree')
plt.plot(t,abs(res2),'b', label='Xgboost')
plt.xlabel('time')
plt.ylabel('Error value')
plt.legend(loc='upper right')
plt.title('Residual plot')
plt.show()


plot_features(xgb, (10,14))