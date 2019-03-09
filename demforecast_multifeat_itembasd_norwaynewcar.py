import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import math 
from xgboost import plot_tree

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

plt.close('all')

trainpercntg = .8

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

df = pd.read_csv('norway_new_car_sales_by_model.csv',encoding='cp1252')
#df = df[:80000]
h = df.shape
h1 = h[0]
h2 = h[1]

title = list(df.columns.values)

X = df.iloc[:,0:h2-1]
Y = df.iloc[:,h2-1]
    
X = X.fillna(0)

X1 = X.copy() 
    
    
l1 = len(X1)
l2 = l1*trainpercntg
l2 = np.floor(l2)
l2 = int(l2)
l3 = l1-l2

Xtrain = X1[:l2]
Ytrain = Y[:l2]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Xtrain[title[0]])
Xtrain[title[0]] = le.transform(Xtrain[title[0]])    

le1 = preprocessing.LabelEncoder()
le1.fit(Xtrain[title[1]])
Xtrain[title[1]] = le1.transform(Xtrain[title[1]]) 

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
    
   
mod = input('Enter Required Category\t {}\t{}\n'.format(title[0],title[1]))


if mod == 'Make':
    
    cat = input('Enter Required {} \n'.format(title[0]))
    cat = cat+' '
    
    dcat = Xtest.loc[Xtest[title[0]]==cat]
    
elif mod == 'Model':
    
    cat = input('Enter Required {} \n'.format(title[1]))

    dcat = Xtest.loc[Xtest[title[1]]==cat]


ind = dcat.index
ind = list(ind)



X2 = dcat
X2[title[0]] = le.transform(X2[title[0]])    
X2[title[1]] = le1.transform(X2[title[1]])    


Y2 = Y.loc[ind]
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

Ytest = np.float64(Y2)
Ytest = Ytest.reshape((-1,))
res1 = predtest1-Ytest

mse = ((res1) ** 2).mean()
rmse = math.sqrt(mse)
print('RMSE for Decision tree is', rmse)


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
    
res2 = predtest2-Ytest

mse = ((res2) ** 2).mean()
rmse1 = math.sqrt(mse)
print('RMSE for Xgboost is', rmse1)
    
    
plot_features(xgb, (10,14))


