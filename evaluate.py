# 9.
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import wrangle
from scipy.stats import norm
import statistics
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(123)


def plot_residuals(y, yhat):
    def plot_residuals(y, yhat):
    plt.scatter(df.property_value, df.square_feet)

    #lineplot is my regression line
    plt.plot(df.property_value, df.yhat)

    plt.xlabel('property value')
    plt.ylabel('sqaure feet')
    plt.title('OLS linear model')
    plt.show()
    
     
    
def regression_errors(y, yhat):
    SSE = df.residual_2.sum()
    print('SSE =', "{:.1f}".format(SSE))
    
    ESS = sum((df.yhat - df.property_value.mean())**2)
    print('ESS =', "{:.1f}".format(ESS))
    
    TSS = ESS + SSE
    print('TSS =', "{:.1f}".format(TSS))
    
    MSE = SSE/len(df)
    print(f'MSE = {MSE:.1f}')
    
    RMSE = MSE**.5
    print("RMSE = ", "{:.1f}".format(RMSE))
    
    
    
def baseline_mean_errors(y, yhat):
    SSE_baseline = df.baseline_residual_2.sum()
    
    print("SSE Baseline =", "{:.1f}".format(SSE_baseline))
    
    MSE_baseline = SSE_baseline/len(df)
    
    print(f"MSE baseline = {MSE_baseline:.1f}")
    
    RMSE_baseline = MSE_baseline**.5
    
    print("RMSE baseline = ", "{:.1f}".format(RMSE_baseline))
    
    
    
    
def better_than_baseline(y,yhat):
    if RMSE_baseline > RMSE and SSE_baseline > SSE and MSE_baseline > MSE:
        print("is better than baseline")
    else:
        print("is not better than baseline")