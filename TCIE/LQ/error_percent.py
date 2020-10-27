
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd

file = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\error_percent.xlsx'
outfile =r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\error_percent.csv'
def xlsx_to_csv_pd():
    data_xls = pd.read_excel(file, index_col=0)
    data_xls.to_csv(outfile, encoding='utf-8')

if __name__ == '__main__':
    xlsx_to_csv_pd()
    print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))
    plt.rcParams['figure.figsize'] = (10.0, 9.0)
    font = {'family': 'Arial',
             'weight': 'normal',
             'size': 15,
             }

    hw1 = pd.read_csv(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\error_percent.csv')
    # 转置
    hw1.values
    data = hw1.as_matrix()
    data = list(map(list, zip(*data)))
    data = pd.DataFrame(data)
    data.to_csv(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\error_percent_ZZ.csv',header='AE', index=0)
    filename = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\error_percent_ZZ.csv'  # CSV文件路径
    lines = []
    with open(filename, 'r') as f:
        lines = f.read().split('\n')

    dataSets = []

    # for line in lines:
        # print(line)
    try:
        dataSets.append(lines[1].split(','))
    except:
        print("Error: Exception Happened... \nPlease Check Your Data Format... ")

    temp = []
    for set in dataSets:
        temp2 = []
        for item in set:
            if item != '':
                temp2.append(float(item))
        temp2.sort()
        temp.append(temp2)
    dataSets = temp

    for set in dataSets:

        plotDataset = [[], []]
        count = len(set)
        for i in range(count):
            plotDataset[0].append(float(set[i]))
            plotDataset[1].append((i + 1) / count)
        print(plotDataset)
        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 24,
                }
        fig, axes = plt.subplots()
        plt.tick_params(labelsize=18)
        plt.xlabel('Absolute Error(m/s)', font)
        plt.ylabel('Percent', font)
        plt.grid(linestyle='-.')
        axes.set_xticks([2, 4, 6, 8, 16])
        axes.set_xticklabels(['2', '4', '6', '8', '16'])
        axes.set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.0])
        axes.set_yticklabels(['0.00', '0.20', '0.40','0.60', '0.80', '1.00'])
        plt.plot(plotDataset[0], plotDataset[1], '-', c='r',linewidth=2)

    plt.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\PIC\error_P1.tif', dpi=900)
    plt.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ\tongji2_result_0_26_1_63_2_35\\PIC\error_p1.eps')
    plt.show()

