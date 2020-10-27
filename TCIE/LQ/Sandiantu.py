
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd

file = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\Sandaintu.xlsx'
outfile =r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\Sandaintu.csv'
def xlsx_to_csv_pd():
    data_xls = pd.read_excel(file, index_col=0)
    data_xls.to_csv(outfile, encoding='utf-8')

if __name__ == '__main__':
    # xlsx_to_csv_pd()
    print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))
    #将txt文件转换为csv文件
    font = {'family': 'Arial',
             'weight': 'normal',
             'size': 15,
             }

    hw = pd.read_csv(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\Sandaintu.csv')
    plt.scatter(hw['Intensity Estimation(m/s)'], hw['Best-Track (m/s)'], marker = '.')
    plt.xlabel('Intensity Estimation(m/s)', font)
    plt.ylabel('Best-Track (m/s)', font)
    plt.tick_params(labelsize=12)
    plt.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\PIC\Sandiantu.tif')
    plt.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\\PIC\Sandiantu.eps')
    plt.show()