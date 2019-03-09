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

df = pd.read_csv('mldata.csv')
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


# segmentation product & location-------------------------------------------

Un1 = df.Product.unique()

Dict1 = {elem : pd.DataFrame for elem in Un1}

for key in Dict1.keys():
    Dict1[key] = df[:][df.Product == key]
    
# prod1------------------
d1 = Dict1['prod1']

Un2 = d1.Location.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Location == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by location for prod1')
plt.show()
                  
                  
                  

# prod2------------------
d1 = Dict1['prod2']

Un2 = d1.Location.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Location == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by location for prod2')
plt.show()


    



# prod3------------------
d1 = Dict1['prod3']

Un2 = d1.Location.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Location == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by location for prod3')
plt.show()
    







# segmentation product & customer-------------------------------------------

Un1 = df.Product.unique()

Dict1 = {elem : pd.DataFrame for elem in Un1}

for key in Dict1.keys():
    Dict1[key] = df[:][df.Product == key]
    
# prod1------------------
d1 = Dict1['prod1']

Un2 = d1.Customer.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Customer == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by customer for prod1')
plt.show()
                  
                  
                  
# prod2------------------
d1 = Dict1['prod2']

Un2 = d1.Customer.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Customer == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by Customer for prod2')
plt.show()


    

# prod3------------------
d1 = Dict1['prod3']

Un2 = d1.Customer.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Customer == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by Customer for prod3')
plt.show()








# segmentation on location & product -------------------------------------------

Un1 = df.Location.unique()

Dict1 = {elem : pd.DataFrame for elem in Un1}

for key in Dict1.keys():
    Dict1[key] = df[:][df.Location == key]
    
# France------------------
d1 = Dict1['France']

Un2 = d1.Product.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Product == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by product for france')
plt.show()
                  
                  
                  
# Germany------------------
d1 = Dict1['Germany']

Un2 = d1.Product.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Product == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by product for Germany')
plt.show()


    

# UK------------------
d1 = Dict1['United Kingdom']

Un2 = d1.Product.unique()

Dict2 = {elem : pd.DataFrame for elem in Un2}

for key in Dict2.keys():
    Dict2[key] = d1[:][d1.Product == key]

frcomb = pd.Series()
for key in Dict2.keys():
    
    fr = Dict2[key] 
    fr = fr.iloc[:,3:5] 
    fr = fr.reset_index(drop=True)
    fr = fr.groupby('Date')['Sales'].sum()
    
    frcomb = pd.concat([frcomb,fr], axis=1)

frcomb = frcomb.drop(frcomb.columns[[0]], axis=1) 

fig=plt.figure()
l1 = len(frcomb.columns)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1,l1))

fkey = list(Dict2.keys())

for i in range(0,l1):
    frs = frcomb.iloc[:,i]
    ax = frs.plot(label=fkey[i]);   
    
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left');
plt.title('Sales by product for UK')
plt.show()





