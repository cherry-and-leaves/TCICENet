#只有一个进行计算
import numpy as np
import pandas as pd
import os
path_ = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27'

if __name__ == '__main__':
    # xlsx_to_csv_pd()
    print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))
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


    a = 0.65
    b = 0.25
    c = 0.1
    for line in f:
        count_all +=1
        print(line)
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



    f.close()
    MAE = all_MAE/count_all
    Bias = bias/count_all
    RMSE = all_RMSE/count_all
    RMSE = np.sqrt(RMSE)
    # RMSE = rmse(wind_pre, wind_real)
    print(MAE, RMSE, Bias)
    file = filepath.split('/')[1]
    np.savetxt(filepath.split('.')[0] + '_smoothpre_%1f_%1f_%1f.csv' %
               (a, b, c), smooth_pre_list)

