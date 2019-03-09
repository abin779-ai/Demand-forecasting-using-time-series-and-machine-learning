import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import math 
from xgboost import plot_tree

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

plt.close('all')

trainpercntg = .8

from tkinter.filedialog import askopenfilename
filename = askopenfilename()
df = pd.read_csv(filename)
df = df[:5000]
h = df.shape
h1 = h[0]
h2 = h[1]

lcat = df['Product_Category'].unique()
lcat = list(lcat)

# removing NaT rows
n1 = df[df.Date.isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2])

title = list(df.columns.values)

df['Date'] = pd.to_datetime(df.Date)
df.drop_duplicates(inplace=True)
df = df.sort_values(by='Date')

    
X = df.iloc[:,0:h2-1]
Y = df.iloc[:,h2-1]

X["Month"] = X.Date.dt.month
X["Day"] = X.Date.dt.day
X = X.drop('Date', 1)
    
X = X.fillna(0)    
X1 = X.copy()
    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


for column in X1.columns:
    le.fit(X1[column])
    X1[column] = le.transform(X1[column])
    
        
l1 = len(X1)
l2 = l1*trainpercntg
l2 = np.floor(l2)
l2 = int(l2)
l3 = l1-l2

Xtrain = X[:l2]
Xtrain_en = X1[:l2]
Ytrain = Y[:l2]

Xtest = X[l2:l1]
Xtest_en = X1[l2:l1]
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
clf = clf.fit(Xtrain_en, Ytrain)

# Xgboost--------------------------------
    
xgb = XGBRegressor()
xgb.fit(Xtrain_en, Ytrain)
    
cat = input('Enter Required {} \n'.format(title[2]))

dcat = Xtest.loc[Xtest[title[2]]==cat]

ind = dcat.index
ind = list(ind)

dcat.index = range(0,len(dcat))

X2 = Xtest_en.loc[ind]
Y2 = Ytest.loc[ind]

l4 = len(X2)
    
predtest1 = clf.predict(X2)
predtrain1 = clf.predict(Xtrain_en)

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
predtrain2 = xgb.predict(Xtrain_en)

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
    
    
