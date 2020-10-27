#read txt method one
import os
import numpy as np
# from sklearn.metrics import mean_squared_error
from math import sqrt
all_MAE = 0
all_RMSE = 0
path_ = os.path.abspath(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNet_A_Res_170_ZJ_0.0005')
path_produce = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNet_A_Res_170_ZJ_0.0005\tongji'
f = open(path_ + '/result_2_6.51_8.53_15.txt')
count = 0
wind_real_list = []
wind_pre_list = []
file_write_obj_0 = open(os.path.join(path_produce, r"0.txt"), 'a') # a可以继续写
file_write_obj_1 = open(os.path.join(path_produce, r"1.txt"), 'a')
file_write_obj_2 = open(os.path.join(path_produce, r"2.txt"), 'a')
file_write_obj_3 = open(os.path.join(path_produce, r"3.txt"), 'a')
file_write_obj_4 = open(os.path.join(path_produce, r"4.txt"), 'a')
file_write_obj_Z = open(os.path.join(path_produce, r"zong.txt"), 'a')
def wind_Inten(wind):
    if wind <= 47:
        intensitytype = 0
    elif wind >= 48 and wind <= 63:
        intensitytype = 1
    elif wind >= 64 and wind <= 80:
        intensitytype = 2
    elif wind >= 81 and wind <= 100:
        intensitytype = 3
    else:
        intensitytype = 4
    return intensitytype
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
for line in f:
    file_write_obj_Z.writelines(line)
    print(line)
    line1 = line.split(',')[1]
    line2 = line1.split(")")[0]
    wind_pre = float(line2)
    wind_real = line.split('_')[2]
    wind_real = float(wind_real)
    if wind_Inten(wind_real) == 0:
        file_write_obj_0.writelines(line)

    elif wind_Inten(wind_real) == 1:
        file_write_obj_1.writelines(line)

    elif wind_Inten(wind_real) == 2:
        file_write_obj_2.writelines(line)

    elif wind_Inten(wind_real) == 3:
        file_write_obj_3.writelines(line)

    else:
        file_write_obj_4.writelines(line)


file_write_obj_0.close()
file_write_obj_1.close()
file_write_obj_2.close()
file_write_obj_3.close()
file_write_obj_4.close()