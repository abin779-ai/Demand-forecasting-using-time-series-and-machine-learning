import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import math 
from xgboost import plot_tree

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


plt.close('all')

trainpercntg = .8

df = pd.read_csv('ecom data.csv',encoding='cp1252')
df = df[:1000]
h = df.shape
h1 = h[0]
h2 = h[1]

# removing NaT rows
n1 = df[df.InvoiceDate.isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2])

# removing negative value rows
df = df.ix[df['Quantity'] >= 0]

df['Description'].astype(str)

title = list(df.columns.values)

df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
df = df.sort_values(by='InvoiceDate')

    
X = df.iloc[:,0:h2-1]
Y = df.iloc[:,h2-1]

X["Month"] = X.InvoiceDate.dt.month
X["Day"] = X.InvoiceDate.dt.day
X = X.drop('InvoiceDate', 1)
    
X = X.fillna(0)

X1 = X.copy()
    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(X1['StockCode'])
X1['StockCode'] = le.transform(X1['StockCode'])    

#X11 = X1['Description']
#
#for i in X11.index:
#    X12 = str(X11[i])
#    X11[i] = X12
#    
#X1['Description'] = X11

le.fit(X1['Description'])
X1['Description'] = le.transform(X1['Description'])

le.fit(X1['Country'])
X1['Country'] = le.transform(X1['Country'])
    
    
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

    
# Xgboost--------------------------------
    
xgb = XGBRegressor()
xgb.fit(Xtrain, Ytrain)

X2 = Xtest
Y2 = Ytest
Y2 = np.float64(Y2)

l4 = len(X2)
    
predtest1 = clf.predict(X2)
predtrain1 = clf.predict(Xtrain)

t = range(0,l4)

plt.figure()
plt.plot(t,Y2,'b',label = 'Original')
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

#    Ytest = np.float64(Ytest)
#    Ytest = Ytest.reshape((-1,))
#    res1 = predtest1-Ytest
#    
#    mse = ((res1) ** 2).mean()
#    rmse = math.sqrt(mse)
#    print('RMSE for Decision tree is', rmse)


# Xgboost--------------------------------

predtest2 = xgb.predict(X2)
predtrain2 = xgb.predict(Xtrain)

t = range(0,l4)

plt.figure()
plt.plot(t,Y2,'b',label = 'Original')
plt.plot(t, predtest2,'r',label = 'prediction') 
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
    
#    res2 = predtest2-Ytest
#    
#    mse = ((res2) ** 2).mean()
#    rmse1 = math.sqrt(mse)
#    print('RMSE for Xgboost is', rmse1)
    
    
plot_features(xgb,(8,12))    

