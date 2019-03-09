
import numpy as np
import pandas as pd

# initial plotting and plot styling libraries (will be overriden)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# statistical modeling libraries
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
# basic functionalities
import copy
import datetime
import itertools
import math 

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

plt.close('all')


from tkinter.filedialog import askopenfilename
filename = askopenfilename()
df = pd.read_csv(filename)
df.iloc[:,0] = df.iloc[:,0].apply(lambda x: pd.to_datetime(str(x)).strftime('%Y/%m/%d %H:%M:%S'))

# Removing missing values------------------------
n1 = df[df.iloc[:,0].isnull()]
n2 = n1.index.values
n2 = list(n2)
df = df.drop(df.index[n2]) 
df = df.reset_index(drop=True)


lag = 12
trainpercntg = .8
step = 1

dff = copy.copy(df)

a1 = df.iloc[:,1]
a1 = np.log10(a1)


df.iloc[:,1] = a1

df1 = df.iloc[:,0]

df.columns = ['date','value']
df.date = pd.to_datetime(df.date)
df = df.sort_values(by='date')
df.set_index('date', inplace=True)

ts = df['value']

l1 = len(df)
l2 = l1*trainpercntg
l2 = np.floor(l2)
l2 = int(l2)
l3 = l1-l2

ts_train = ts.iloc[0:l2]
ts_train1 = copy.copy(ts_train)
ts_test = ts.iloc[l2:l1+1]  
ts_test1 = copy.copy(ts_test)





# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0,2)
 
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
 
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], lag) for x in list(itertools.product(p, d, q))]



best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None
 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(ts_train,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
            res = tmp_mdl.fit()
            
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, res.aic))
            
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:

            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))




#  PREDICTION ON TRAIN DATA-----------------------------------------------


# in-sample-prediction and confidence bounds

t3 = df1[0]
t4 = df1[l1-1]

# fit model to data
res = sm.tsa.statespace.SARIMAX(ts_train,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()


pred = res.get_prediction(start=pd.to_datetime(t3), 
                          end=pd.to_datetime(t4),
                          dynamic=False)

predtrain = pred.predicted_mean 
predtrain2 = predtrain[lag+1:l2]
predtrain_dn = np.power(10,predtrain)
#predtrain_dn = np.exp(predtrain)


# plot in-sample-prediction
fig=plt.figure()
ax = ts.plot(label='Observed',color='#006699');
predtrain2.plot(ax=ax, label='values',title='Long term Prediction on train data' ,alpha=.7, color='r',linestyle='--');

                       

# style the plot
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(t3), ts.index[-1], alpha=.15, zorder=-1, color='grey');
ax.set_xlabel('Year');
ax.set_ylabel('Monthly Orders');
plt.legend(loc='upper left');
plt.show()


# Compute the mean square error

y_truth = ts
y_forecast = predtrain


mse = ((y_forecast - y_truth) ** 2).mean()
rmse1 = math.sqrt(mse)
print('RMSE on train data is', rmse1)


# ONE YEAR AHEAD PREDICTION----------------------------------------



#
#
## in-sample-prediction and confidence bounds
#

#
#
## fit model to data
#res = sm.tsa.statespace.SARIMAX(ts_train,
#                                    order=best_pdq,
#                                    seasonal_order=best_seasonal_pdq,
#                                    enforce_stationarity=False,
#                                    enforce_invertibility=False).fit()
#
#
#pred = res.get_prediction(start=pd.to_datetime(t3), 
#                          end=pd.to_datetime(t4),
#                          dynamic=True)
#
#predvalues = pred.predicted_mean
#
#
## plot in-sample-prediction
#fig=plt.figure()
#ax = ts.plot(label='Observed',color='#006699');
#pred.predicted_mean.plot(ax=ax, label='values',title='One_Year Prediction' ,alpha=.7, color='#ff0066');
#
#                         
#                         
#
## style the plot
#ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(t3), ts.index[-1], alpha=.15, zorder=-1, color='grey');
#ax.set_xlabel('Year');
#ax.set_ylabel('Monthly Orders');
#plt.legend(loc='upper left');
#plt.show()
#
#
#
## Compute the mean square error
#
#y_truth = ts_test
#y_forecast = predvalues
#
#
#mse = ((y_forecast - y_truth) ** 2).mean()
#rmse2 = math.sqrt(mse)
#print('RMSE for long term prediction is', rmse2)


# One step ahead prediction---------------------------------------------------

tg = df1[l2-1]

predictions = pd.DataFrame()
values = pd.Series()

l22 = l2-1

for i in range(l22,l1-1,step):


    t1 = df1[i]
    
    
    if i+step>l1-1:
        
        t2 = df1[l1-1]
    else:
            
        t2 = df1[i+step]


    # fit model to data
    res = sm.tsa.statespace.SARIMAX(ts_train,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()


    # in-sample-prediction and confidence bounds
    pred = res.get_prediction(start=pd.to_datetime(t1), 
                              end=pd.to_datetime(t2),
                              dynamic=True)
    
    p1 = pred.predicted_mean
    p1 = p1[1:len(p1)]
#     predictions['']
#     predictions.append(p1)
    values = pd.concat([values,p1])
    
    
    tt = ts_test.iloc[0:step]
    
    ts_test = ts_test.drop(ts_test.index[0:step])
    
    ts_train = pd.concat([ts_train,tt])
    
predictions = pd.DataFrame(values)

predtest = predictions.iloc[:,0]
predtest_dn = np.power(10,predtest)
#predtest_dn = np.exp(predtest)


# plot in-sample-prediction
fig=plt.figure()
ax = ts.plot(label='Observed',color='#006699');
predtest.plot(ax=ax, label='Predicted',title='One_step_Ahead_Prediction', alpha=.7, color='r',linestyle='--');


# style the plot
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
ax.set_xlabel('Years');
ax.set_ylabel('Monthly Orders');
plt.legend(loc='upper left');


# Compute the mean square error

y_truth = ts.iloc[l2:l1+1] 
y_forecast = predictions.iloc[:,0]


mse = ((y_forecast - y_truth) ** 2).mean()
rmse3 = math.sqrt(mse)
print('RMSE for one step ahead prediction is', rmse3)




# Combined plot------------------------------------------------


predtrain2 = predtrain[lag+1:l2]
predtrain_dn = predtrain_dn[lag+1:l2]



# plot in-sample-prediction
fig=plt.figure()
ax = ts.plot(label='Observed',color='#006699');
predtest.plot(ax=ax, label='Prediction on test', color='r',linestyle='--');
predtrain2.plot(ax=ax, label='Prediction on train', color='m',linestyle='--');


# style the plot
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(t3), ts.index[-1], alpha=.15, zorder=-1, color='grey');
ax.set_xlabel('Years');
ax.set_ylabel('Monthly Orders');
plt.legend(loc='upper left');
plt.title('One_step_Ahead_Prediction')   
plt.show()



# Test data only--------------------------------------------------


ts_test = ts.iloc[l2:l1+1] 

# plot in-sample-prediction
fig=plt.figure()
ax = ts_test.plot(label='Observed',color='#006699');
predtest.plot(ax=ax, label='Predicted',title='One_step_Ahead_Prediction', alpha=.7, color='r',linestyle='--');


# style the plot
ax.set_xlabel('Years');
ax.set_ylabel('Monthly Orders');
plt.legend(loc='upper left');
plt.show()








# Adding relative change with orginal data-----------------------------------


dff.columns = ['date','value']
dff['date'] = pd.to_datetime(dff.date)
dff.set_index('date', inplace=True)



# plot in-sample-prediction
ax = dff.plot(label='Observed',color='#006699');
predtest_dn.plot(ax=ax, label='Predicted',title='One_step_Ahead_Prediction in Original data', alpha=.7, color='r',linestyle='--');

            
dff_test = dff.iloc[l2:l1+1] 


# plot in-sample-prediction
ax = dff_test.plot(label='Observed',color='#006699');
predtest_dn.plot(ax=ax, label='Predicted',title='One_step_Ahead_Prediction in Original data', alpha=.7, color='r',linestyle='--');


    
# plot in-sample-prediction
ax = dff.plot(label='Observed',color='#006699');
predtest_dn.plot(ax=ax, label='Predicted',title='One_step_Ahead_Prediction in Original data', alpha=.7, color='r',linestyle='--');
predtrain_dn.plot(ax=ax, label='Train data on model',title='One_step_Ahead_Prediction in Original data', alpha=.7, color='m',linestyle='--');



