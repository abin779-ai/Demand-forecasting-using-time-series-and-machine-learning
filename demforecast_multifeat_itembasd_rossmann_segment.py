import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import math 
from xgboost import plot_tree

import seaborn

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


plt.close('all')

trainpercntg = .8

df = pd.read_csv('rossmann1.csv')
df = df[:50000]
h = df.shape
h1 = h[0]
h2 = h[1]

# removing NaT rows
n1 = df[df.Date.isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2])

title = list(df.columns.values)

df['Date'] = pd.to_datetime(df.Date)
df = df.sort_values(by='Date')


# Visualization----------------------------------------------

plt.figure()  
pl = df.groupby('Store')['Sales'].sum().plot(kind='bar')
pl.set_xlabel("Store")
pl.set_ylabel("Sales")
plt.title('Sales based on stores')


plt.figure()  
pl = df.groupby('DayOfWeek')['Sales'].sum().plot(kind='bar')
pl.set_xlabel("Store")
pl.set_ylabel("Sales")
plt.title('Sales based on day')


plt.figure()  
pl = df.groupby('Promo')['Sales'].sum().plot(kind='bar')
pl.set_xlabel("Promo")
pl.set_ylabel("Sales")
plt.title('Sales based on promo')


# segmentation store sales-------------------------------------------

UniqueNames = df.Store.unique()

DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df.Store == key]
    
d3 = []    
for key in DataFrameDict.keys():
    d = DataFrameDict[key]
    d1 = d.Sales
    d2 = np.sum(d1)
    d3= np.append(d3,d2)
    
plt.figure()    
plt.bar(UniqueNames,d3,color="blue")    


# segmentation DayOfWeek-------------------------------------------

UniqueNames = df.DayOfWeek.unique()

DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df.DayOfWeek == key]
    
d3 = []    
for key in DataFrameDict.keys():
    d = DataFrameDict[key]
    d1 = d.Sales
    d2 = np.sum(d1)
    d3= np.append(d3,d2)
    
plt.figure()    
plt.bar(UniqueNames,d3,color="blue")    





# segmentation Promo-------------------------------------------

UniqueNames = df.Promo.unique()

DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df.Promo == key]
    
d3 = []    
for key in DataFrameDict.keys():
    d = DataFrameDict[key]
    d1 = d.Sales
    d2 = np.sum(d1)
    d3= np.append(d3,d2)
    
plt.figure()    
plt.bar(UniqueNames,d3,color="blue") 







    


X = df.iloc[:,0:h2-1]
Y = df.iloc[:,h2-1]

X["Month"] = X.Date.dt.month
X["Day"] = X.Date.dt.day
X = X.drop('Date', 1)
    
X = X.fillna(0)

X1 = X.copy()
    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X1['StateHoliday'])
X1['StateHoliday'] = le.transform(X1['StateHoliday'])    
    
    

