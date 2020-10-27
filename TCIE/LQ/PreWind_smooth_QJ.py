##WXj###穷举法smooth##################
import os
import numpy as np
# from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import os
path_ = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27'
file = os.path.join(path_, r'windPre_1830.xlsx')
outfile = os.path.join(path_, r'windPre_1830.csv')
def xlsx_to_csv_pd():
    data_xls = pd.read_excel(file, index_col=0)
    data_xls.to_csv(outfile, encoding='utf-8')

if __name__ == '__main__':
    # xlsx_to_csv_pd()
    # print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))
    all_MAE = 0
    all_RMSE = 0
    bias = 0
    path_ = os.path.abspath(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27')
    filepath = path_ + '/windPre_1830.txt'
    f = open(filepath)
    count = 0
    count_all = 0
    wind_real_list = []
    wind_pre_list = []
    smooth_pre_list = []
    name_list = []
    name_list_Z = []
    linename_cut = []
    tys = {} # map typhoon to its max wind
    tys_time = {} # map typhoon-time to wind
    A = [x * 0.1 for x in range(3, 11, 1)]
    A5 = [(x * 0.1 - 0.05) for x in range(3, 11, 1)]
    A4 = [(x * 0.1 - 0.04) for x in range(3, 11, 1)]
    A3 = [(x * 0.1 - 0.03) for x in range(3, 11, 1)]
    A2 = [(x * 0.1 - 0.02) for x in range(3, 11, 1)]
    A1 = [(x * 0.1 - 0.01) for x in range(3, 11, 1)]
    A = A + A1 + A2 + A3 + A4 + A5
    B = [x * 0.1 for x in range(0, 7, 1)]
    B5 = [(x * 0.1 + 0.05) for x in range(0, 7, 1)]
    B4 = [(x * 0.1 - 0.04) for x in range(3, 11, 1)]
    B3 = [(x * 0.1 - 0.03) for x in range(3, 11, 1)]
    B2 = [(x * 0.1 - 0.02) for x in range(3, 11, 1)]
    B1 = [(x * 0.1 - 0.01) for x in range(3, 11, 1)]
    B = B + B1 + B2 + B3 + B4 + B5
    min = 8.60
    for a in A:
        for b in B:
            if b > a:
                continue
            else:
                c = 1 - a - b
                if c <= 0 or c > b or c > a:
                    continue
                else:
                    filepath = path_ + '/windPre_1830.txt'
                    f = open(filepath)#################m每次都要打开
                    all_MAE = 0
                    all_RMSE = 0
                    bias = 0
                    count = 0
                    count_all = 0
                    wind_real_list = []
                    wind_pre_list = []
                    smooth_pre_list = []
                    name_list = []
                    name_list_Z = []
                    linename_cut = []
                    tys = {}  # map typhoon to its max wind
                    tys_time = {}  # map typhoon-time to wind
                    for line in f:
                        count_all +=1
                        # print(line)
                        line0 = line.split(',')[0]
                        line1 = line.split(",")[1]
                        line2 = line.split(",")[2]
                        wind_pre = float(line2)
                        wind_real = float(line1)
                        wind_real_list.append(wind_real)
                        wind_pre_list.append(wind_pre)
                        name = line0.split('_')
                        tid = name[0]
                        if tys.__contains__(tid):
                            count += 1
                            if count>=3:
                                smooth_pre = a*wind_pre_list[-1]+\
                                           b*wind_pre_list[-2]+c*wind_pre_list[-3]##################前三个
                                smooth_pre_list.append(smooth_pre)
                            else:
                                smooth_pre_list.append(wind_pre)
                            tid_time = name[0]
                            tys_time[tid_time] = count
                            if tys[tid] < wind_pre:
                                tys[tid] = wind_pre
                        else:
                            count = 1
                            smooth_pre_list.append(wind_pre)
                            tys[tid] = wind_pre

                        bias = bias + smooth_pre_list[-1] - wind_real
                        all_MAE = all_MAE + abs(smooth_pre_list[-1] - wind_real)
                        all_RMSE = all_RMSE + (smooth_pre_list[-1] - wind_real) ** 2
                    MAE = all_MAE/count_all
                    Bias = bias/count_all
                    RMSE = all_RMSE/count_all
                    RMSE = np.sqrt(RMSE)
                    # RMSE = rmse(wind_pre, wind_real)
                    print('a%.2f b%.2f c%.2f, MAE%.2f RMSE%.4f Bias%.2f' % (a, b, c, MAE, RMSE, Bias))
                    if RMSE < min:
                        min = RMSE
                        print('min:%.2f_%.2f_%.2f,%.2f_%.4f_%.2f' % (a, b, c, MAE, RMSE, Bias))
                    file = filepath.split('/')[0]
                    np.savetxt(file + '\QJ\smoothpre_%.2f_%.2f_%.2f.csv' %
                               (a, b, c), smooth_pre_list)


