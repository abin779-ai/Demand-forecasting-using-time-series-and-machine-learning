
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
from scipy.special import boxcox, inv_boxcox
from scipy.special import logit,expit
from scipy import stats


import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

plt.close('all')
cv2.destroyAllWindows()

def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))


trainpercntg = .8
step = 1    


from tkinter.filedialog import askopenfilename
filename = askopenfilename()
df = pd.read_csv(filename)
dff = copy.copy(df)
df1 = df.iloc[:,0]
dff.columns = ['date','value']
dff['date'] = pd.to_datetime(dff.date)
dff.set_index('date', inplace=True)

typ = input('Enter the typ of data: monthly,daily,hourly \n')


if typ == 'monthly':
    lag = 12
elif typ =='daily':
    lag = 30
elif typ =='hourly':
    lag = 24


def preprocess(df):

    df.iloc[:,0] = df.iloc[:,0].apply(lambda x: pd.to_datetime(str(x)).strftime('%Y/%m/%d %H:%M:%S'))
    
    # Removing missing values------------------------
    n1 = df[df.iloc[:,0].isnull()]
    n2 = n1.index.values
    n2 = list(n2)
    df = df.drop(df.index[n2]) 
    df = df.reset_index(drop=True)
    
    if not n2:
        print("There is no missing values")
    else:
        print("Missing values found and is removed")
            
    
    df.columns = ['date','value']
    df.date = pd.to_datetime(df.date)
    df = df.sort_values(by='date')
    print("Data sorted in the order of date")
    print(df)
    df2 = df.copy()
    df2.set_index('date', inplace=True)
    df2=df2.iloc[:,0]
    df2.plot(title='Original data',color='#006699')  
    plt.xlabel('Time')
    plt.ylabel('Order value')
    plt.show()
    return df


df = preprocess(df)


def visualize(df):
    
    df2 = df.copy()
    df2.set_index('date', inplace=True)
    decomposition = seasonal_decompose(df2)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.plot(df2)
    plt.title('Original')
    plt.show()
    
    plt.subplot(222)
    plt.plot(trend)
    plt.title('Trend')
    plt.legend(loc='best')
    plt.show()
    
    plt.subplot(223)
    plt.plot(seasonal)
    plt.title('Seasonality')
    plt.legend(loc='best')
    plt.show()
    
    plt.subplot(224)
    plt.plot(residual)
    plt.title('Residuals')
    plt.legend(loc='best')
    plt.show()
    print('Data Summary : ')
    print(df.describe())
    return

visualize(df)


def transform(df,trs):
    
    
    X = df.iloc[:,1]
    l1 = len(X)
    X1 = X[1:l1]
    X1.index = range(0,l1-1)
    X2 = X[0:l1-1]
    X4 = (X1-X2)
    minx = min(X4)
    minx = abs(minx)+10
    a1 = df.iloc[:,1]
    a1 = abs(a1)
    dmin = min(a1)
    dmax = max(a1)
    a1,lam = stats.boxcox(a1)

    
    
    if trs=='relative change':
    
    
        X = df.iloc[:,1]
        l1 = len(X)
        X1 = X[1:l1]
        X1.index = range(0,l1-1)
        X2 = X[0:l1-1]
        X3 = (X1-X2)/X2
        
        df = df.drop(df.index[0])
        df.index = range(0,l1-1)
        df.iloc[:,1] = X3
    
    
    elif trs=='log':

        a1 = df.iloc[:,1]
        a1 = np.log10(a1)
        df.iloc[:,1] = a1
        
    elif trs=='log-difference':
        
        X = df.iloc[:,1]
        l1 = len(X)
        X1 = X[1:l1]
        X1.index = range(0,l1-1)
        X2 = X[0:l1-1]
        
        X4 = (X1-X2)
        minx = min(X4)
        minx = abs(minx)+10
        X4 = X4+minx
        X3 = np.log10(X4)
        
        df = df.drop(df.index[0])
        df.index = range(0,l1-1)
        df.iloc[:,1] = X3
    
    elif trs=='logistic':
        
        a1 = df.iloc[:,1]
        a1 = abs(a1)
        dmin = min(a1)
        dmax = max(a1)
        a1 = (a1-dmin+1)/dmax
        a1 = logit(a1)
        df.iloc[:,1] = a1
        
    elif trs=='box-cox':
        
        a1 = df.iloc[:,1]
        a1,lam = stats.boxcox(a1)
        df.iloc[:,1] = a1
        
    elif trs=='square root':
        
        a1 = df.iloc[:,1]
        a1 = np.sqrt(a1)
        df.iloc[:,1] = a1
        
    elif trs=='arcsine':
        
        a1 = df.iloc[:,1]
        a1 = abs(a1)
        dmin = min(a1)
        dmax = max(a1)
        a1 = (a1-dmin+1)/dmax
        a1 = np.arcsin(a1)
        df.iloc[:,1] = a1
        
    elif trs=='reciprocal':   
        
        a1 = df.iloc[:,1]
        a1 = 1/a1
        df.iloc[:,1] = a1
        
    df1 = df.iloc[:,0]
    df.set_index('date', inplace=True)
    plt.figure()
    plt.plot(df,color='#006699')
    plt.title('Transformed data')
    plt.xlabel('Time')
    plt.ylabel('Order value')
    plt.show()
    return df,df1,minx,dmin,dmax,lam


trs = input('Enter transformation: relative change,log,log-difference,logistic,box-cox,square root,arcsine,reciprocal \n')
df,df1,minx,dmin,dmax,lam = transform(df,trs)
ts = df['value']


def datasplit(ts,trainpercntg):
    
    l1 = len(ts)
    l2 = l1*trainpercntg
    l2 = np.floor(l2)
    l2 = int(l2)
    l3 = l1-l2
    
    ts_train = ts.iloc[0:l2]
    ts_test = ts.iloc[l2:l1+1]  
    return ts_train,ts_test,l1,l2,l3

ts_train,ts_test,l1,l2,l3 = datasplit(ts,trainpercntg)

ts_train1 = copy.copy(ts_train)
ts_test1 = copy.copy(ts_test)



def modelling(ts_train,df1,trs,minx,dmin,dmax,lam):
    

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
                
                
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl
            except:
    
                continue
    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

#    if trs=='relative change' or trs=='log-difference':
#        t3 = df1[1]
#    else:
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

    
    # plot in-sample-prediction
#    plt.figure()
#    ax = ts.plot(label='Observed',color='#006699');
#    predtrain2.plot(ax=ax, label='Prediction',title='Train data on model - Transformed' ,alpha=.7, color='r',linestyle='--');
#    ax.set_xlabel('Time');
#    ax.set_ylabel('Order value');
#    plt.legend(loc='upper left');
#    plt.show()
    
    
    if trs=='log':
    
        predtrain_dn = np.power(10,predtrain)
        predtrain_dn = predtrain_dn[lag+1:l2]
    
    elif trs=='relative change':
    
        c3 = dff.iloc[lag+1,0]
        a5 = copy.copy(predtrain2)
        
        for k in range(0,l2-lag-1):
            
            c4 = c3+(predtrain2[k]*c3)
            a5.iloc[k] = c4
            c3 = a5[k]
        predtrain_dn = a5
        
    elif trs=='log-difference':
        
        
        c3 = dff.iloc[lag-2,0]
        predtrain_dn = np.power(10,predtrain2)
        a5 = predtrain_dn
        
        for k in range(0,l2-lag-1):
            
            c4 = c3+predtrain_dn[k]-minx
            a5.iloc[k] = c4
            c3 = a5[k]
            
        predtrain_dn = a5 
        
    elif trs=='logistic': 
         
        predtrain_dn = ((expit(predtrain))*dmax) + dmin-1
        predtrain_dn = predtrain_dn[lag+1:l2]

    elif trs=='box-cox':
        
        predtrain_dn = invboxcox(predtrain,lam)
        predtrain_dn = predtrain_dn[lag+1:l2]
     
    elif trs=='square root':
        
        predtrain_dn = np.square(predtrain)
        predtrain_dn = predtrain_dn[lag+1:l2]

    elif trs=='arcsine':
        
        predtrain_dn = ((np.sin(predtrain))*dmax) + dmin
        predtrain_dn = predtrain_dn[lag+1:l2]

    elif trs=='reciprocal':

        predtrain_dn = 1/predtrain
        predtrain_dn = predtrain_dn[lag+1:l2]

    tg = df1[l2]
    dff1 = dff.iloc[:,0]
    plt.figure()
    ax = dff1.plot(label='Original',color='#006699');
    predtrain_dn.plot(ax=ax, label='Model',title='Model output on train data', alpha=.7, color='m',linestyle='--');
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    ax.text(.9,.15,'test data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    ax.text(.5,.15,'train data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left');
    plt.show()
    return predtrain2,predtrain_dn,best_pdq,best_seasonal_pdq,res



mod = input('Enter modelling:   arima,simple average,weighted average,moving average,weighted moving average,exponential smoothing  \n')

if mod == 'arima':
    
    predtrain2,predtrain_dn,best_pdq,best_seasonal_pdq,res = modelling(ts_train,df1,trs,minx,dmin,dmax,lam)


def modeldiagnostics(res):
    res.plot_diagnostics(figsize=(12,8))
    plt.tight_layout()
    plt.show()
    return

modeldiagnostics(res)


# One step ahead prediction---------------------------------------------------

def forecast(df1,ts_train,ts_test,best_pdq,best_seasonal_pdq,predtrain2,predtrain_dn):

    
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
        values = pd.concat([values,p1])
        
        tt = ts_test.iloc[0:step]
        
        ts_test = ts_test.drop(ts_test.index[0:step])
        
        ts_train = pd.concat([ts_train,tt])
        
    predictions = pd.DataFrame(values)
    
    predtest = predictions.iloc[:,0]
        
    
    # plot in-sample-prediction
#    fig=plt.figure()
#    ax = ts.plot(label='Original',color='#006699');
#    predtest.plot(ax=ax, label='Forecasted',title='Forecasting test data - Transformed', alpha=.7, color='r',linestyle='--');
#    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
#    ax.set_xlabel('Time');
#    ax.set_ylabel('Order value');
#    plt.legend(loc='upper left');
    
    
    # Test data only--------------------------------------------------
    
#    ts_test = ts.iloc[l2:l1+1]
#    plt.figure()
#    ax = ts_test.plot(label='Original',color='#006699');
#    predtest.plot(ax=ax, label='Forecasted',title='Forecasting test data - Transformed', alpha=.7, color='r',linestyle='--');
#    ax.set_xlabel('Years');
#    ax.set_ylabel('Monthly Orders');
#    plt.legend(loc='upper left');
#    plt.show()
    
    
    # Adding relative change with orginal data-----------------------------------
    
    if trs=='log':
    
        predtest_dn = np.power(10,predtest)

    elif trs=='relative change':
    
        c1 = dff.iloc[l2,0]

        a4 = copy.copy(predtest)
        
        for k in range(0,l3):
            
            c2 = c1+(predtest[k]*c1)
            a4.iloc[k] = c2
            
            c1 = a4[k]
        predtest_dn = a4
        
    elif trs=='log-difference':
        
        c1 = dff.iloc[l2,0]
        
        predtest_dn = np.power(10,predtest)
        a4 = predtest_dn
        
        for k in range(0,l3):
            
            c2 = c1+predtest_dn[k]-minx
            a4.iloc[k] = c2
            c1 = a4[k]
                
        predtest_dn = a4
        
    elif trs=='logistic': 
         
        predtest_dn = ((expit(predtest))*dmax) + dmin-1


    elif trs=='box-cox':
        
        predtest_dn = invboxcox(predtest,lam)

    elif trs=='square root':
        
        predtest_dn = np.square(predtest)

    elif trs=='arcsine':
        
        predtest_dn = ((np.sin(predtest))*dmax) + dmin

    elif trs=='reciprocal':

        predtest_dn = 1/predtest
    
    tg = df1[l2-1]
    dff1 = dff.iloc[:,0]
    plt.figure()
    ax = dff1.plot(label='Original',color='#006699');
    predtest_dn.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting'.format(step), alpha=.7, color='r',linestyle='--');
    predtrain_dn.plot(ax=ax, label='Model', alpha=.7, color='m',linestyle='--');
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    ax.text(.9,.15,'test data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    ax.text(.5,.15,'train data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left');
    plt.show()
                
    dff_test = dff.iloc[l2:l1+1] 
    
    dff_test1 = dff_test.iloc[:,0]
    plt.figure()
    ax = dff_test1.plot(label='Original',color='#006699');
    predtest_dn.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting'.format(step), alpha=.7, color='r',linestyle='--');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    plt.legend(loc='upper left');
    plt.show()
    
    # display predicted result----------------------------
    print(predtest_dn)
    
    return predtest_dn,predictions

predtest_dn,predictions = forecast(df1,ts_train,ts_test,best_pdq,best_seasonal_pdq,predtrain2,predtrain_dn)


def forecasterror(ts,dff,predtest_dn):
    
    dff = dff[len(dff)-len(ts):len(dff)]
    y_truth = dff.iloc[l2:l1+1,:] 
    y_truth = y_truth.iloc[:,0]
    y_forecast = predtest_dn
    
    residuals = y_truth-y_forecast
    
    print(residuals.describe())
    
    plt.figure()
    plt.suptitle('Forecast Error')
    plt.subplot(221)
    plt.plot(residuals)
#    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.title('Residual')
    plt.show()
    
    plt.subplot(222)
    residuals.hist()
    plt.title('Residual histogram')
    plt.show()

    plt.subplot(223)
    residuals.plot(kind='kde')
    plt.title('Density')
    plt.show()

    
    import scipy.stats as stats
    plt.subplot(224)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Quantile-Quantile plot')
    plt.show()
    
    mse = ((y_forecast - y_truth) ** 2).mean()
    rmse = math.sqrt(mse)
    print('RMSE for prediction is', rmse)
    
    from sklearn.metrics import r2_score
    r2score = r2_score(y_truth, y_forecast)
    print('r2 score for prediction is', r2score)
    
    from sklearn.metrics import explained_variance_score
    evs = explained_variance_score(y_truth, y_forecast)
    print('explained_variance_score for prediction is', evs)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_truth, y_forecast)
    print('mean_squared_error for prediction is', mse)
    
    from sklearn.metrics import mean_squared_log_error
    msle = mean_squared_log_error(y_truth, y_forecast)
    print('mean_squared_log_error for prediction is', msle)
    
   # def mean_absolute_percentage_error(y_true, y_forecast): 
    y_truth, y_forecast = np.array(y_truth), np.array(y_forecast)
    MAPE = np.mean(np.abs((y_truth - y_forecast) / y_truth)) * 100
    print('MAPE is', MAPE)

  # def median_absolute_percentage_error(y_true, y_forecast): 
    Median_Absolute_Percentage_Error = np.median(np.abs((y_truth - y_forecast) / y_truth)) * 100
    print('Median_Absolute_Percentage_Error is', Median_Absolute_Percentage_Error)
    
    from texttable import Texttable
    t = Texttable()
    t.add_rows([['Perfomance metric','Value'],['Root Mean Squared Error',rmse], ['R squared',r2score],['Explained Variance Score',evs],['Mean Squared Log Error',msle],['Mean Absolute Percentage Error',MAPE],['Median Absolute Percentage Error', Median_Absolute_Percentage_Error]])
    print(t.draw())
    
    return 

forecasterror(ts,dff,predtest_dn)



# net forecast-------------------------------
def netforecast(predtest_dn,dff,l1,l2):
    
    pred3 = predtest_dn[0:3]
    
    per3 = [.05,.1,.15]
    
    pred4 = pred3 * per3
    
    pred5 = pred3 - pred4
    
    prednet = predtest_dn.copy()
    
    prednet[0:3] = pred5
    
    print('\n Net Forecast \n')
    print(prednet)
    
    
    tg = df1[l2-1]
    dff1 = dff.iloc[:,0]
    plt.figure()
    ax = dff1.plot(label='Original',color='#006699');
    prednet.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting - Net Forecast'.format(step), alpha=.7, color='r',linestyle='--');
    predtrain_dn.plot(ax=ax, label='Model', alpha=.7, color='m',linestyle='--');
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    ax.text(.9,.15,'test data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    ax.text(.5,.15,'train data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left');
    plt.show()
    
    
    dff_test = dff.iloc[l2:l1+1] 
    
    dff_test1 = dff_test.iloc[:,0]
    plt.figure()
    ax = dff_test1.plot(label='Original',color='#006699');
    prednet.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting - Net Forecast'.format(step), alpha=.7, color='r',linestyle='--');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    plt.legend(loc='upper left');
    plt.show()
    

    return prednet

netforecast(predtest_dn,dff,l1,l2)

# manual correction-----------------


def manualcorrection(cor,predtest_dn,dff,l1,l2):

    cor = int(cor)
    c = cor/100
    predman = predtest_dn.copy()
    predman = predman + (predman * c)
    
    
    print('\n Corrected Forecast \n')
    print(predman)
    
    tg = df1[l2-1]
    dff1 = dff.iloc[:,0]
    plt.figure()
    ax = dff1.plot(label='Original',color='#006699');
    predman.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting - Manual Correction'.format(step), alpha=.7, color='r',linestyle='--');
    predtrain_dn.plot(ax=ax, label='Model', alpha=.7, color='m',linestyle='--');
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(tg), ts.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    ax.text(.9,.15,'test data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    ax.text(.5,.15,'train data',fontsize=12,ha='center', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left');
    plt.show()
    
    dff_test = dff.iloc[l2:l1+1] 
    
    dff_test1 = dff_test.iloc[:,0]
    plt.figure()
    ax = dff_test1.plot(label='Original',color='#006699');
    predman.plot(ax=ax, label='Forecasted',title='{} Step Ahead Forecasting - Manual Correction'.format(step), alpha=.7, color='r',linestyle='--');
    ax.set_xlabel('Time');
    ax.set_ylabel('Order value');
    plt.legend(loc='upper left');
    plt.show()
    
    return predman

cor = input('Enter percentage of correction:   \n')
manualcorrection(cor,predtest_dn,dff,l1,l2)

