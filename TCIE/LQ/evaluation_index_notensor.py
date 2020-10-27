#read txt method one
import os
import numpy as np
# from sklearn.metrics import mean_squared_error
from math import sqrt
all_MAE = 0
all_RMSE = 0
bias = 0
path_ = os.path.abspath(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNet_A_Res_170_ZJ\tongji')
filepath = path_ + '/zong.txt'
f = open(filepath)
count = 0
wind_real_list = []
wind_pre_list = []
name_list = []
name_list_Z = []
linename_cut = []
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
for line in f:
    print(line)
    line1 = line.split(',')[1]
    line2 = line1.split(")")[0]
    line3 = line.split("'")[1]
    name_cut = line3.split('_')[0] + '_'+ line3.split('_')[1]
    wind_pre = float(line2)
    wind_real = line.split('_')[2]
    wind_real = float(wind_real)
    bias = bias + wind_pre - wind_real
    all_MAE = all_MAE + abs(wind_pre - wind_real)
    all_RMSE = all_RMSE + (wind_pre - wind_real) ** 2
    wind_real_list.append(wind_real)
    wind_pre_list.append(wind_pre)
    name_list.append(line3)
    linename_cut.append(name_cut)
    count = count +1
f.close()
MAE = all_MAE/count
Bias = bias/count
RMSE = all_RMSE/count
RMSE = np.sqrt(RMSE)
# RMSE = rmse(wind_pre, wind_real)
print(MAE, RMSE, Bias)
file = filepath.split('/')[1]
if file == 'zong.txt':
    name_list_Z.append(wind_real_list)
    name_list_Z.append(wind_pre_list)
    #
    np.savetxt(filepath.split('.')[0] + '_real.csv', wind_real_list )
    np.savetxt(filepath.split('.')[0] + '_pre.csv', wind_pre_list)
    np.savetxt(filepath.split('.')[0] + '_linename_cut.csv', linename_cut, fmt='%s')
    np.savetxt(filepath.split('.')[0] + '_name_list.csv', name_list, fmt='%s')

