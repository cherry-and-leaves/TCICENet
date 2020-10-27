#read txt method one
import os
import numpy as np
# from sklearn.metrics import mean_squared_error
from math import sqrt
# all_mae = 0
# all_mse = 0
path_ = os.path.abspath(r'D:\1WXJ\Estimate\plot\LQ_previous\WXJ')
f = open(path_ + '/3.txt')
# count = 0
# wind_real_list = []
# wind_pre_list = []

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def Index(f):
    all_mae = 0
    all_mse = 0
    count = 0
    wind_real_list = []
    wind_pre_list = []
    for line in f:
        print(line)
        line1 = line.split('(')[2]
        line2 = line1.split(")")[0]
        wind_pre = float(line2)
        wind_real = line.split('_')[2]
        wind_real = float(wind_real)
        bias = bias + wind_pre - wind_real
        all_mae = all_mae + abs(wind_pre - wind_real)
        all_mse = all_mse + (wind_pre - wind_real) ** 2
        wind_real_list.append(wind_real)
        wind_pre_list.append(wind_pre)
        count = count +1
    f.close()
    MAE = all_mae/count
    Bias = bias/count
    RMSE = all_mse/count
    RMSE = np.sqrt(RMSE)
    # RMSE = rmse(wind_pre, wind_real)
    print(MAE, RMSE, Bias)
    return MAE, RMSE, Bias


MAE, RMSE, Bias = Index(f) 
print(MAE, RMSE, Bias)