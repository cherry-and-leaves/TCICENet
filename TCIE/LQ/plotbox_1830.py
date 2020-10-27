
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pandas as pd
path_ = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27'
file = os.path.join(path_, r'plotbox_1830.xlsx')
outfile =os.path.join(path_, r'plotbox_1830.csv')
def xlsx_to_csv_pd():
    data_xls = pd.read_excel(file, index_col=0)
    data_xls.to_csv(outfile, encoding='utf-8')

if __name__ == '__main__':
    xlsx_to_csv_pd()
print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))

tips = pd.read_csv(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27\plotbox_1830.csv')
# tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

plt.rcParams['figure.figsize'] = (10.0, 9.0) # 设置figure_size尺寸
fig, axes = plt.subplots()
# colums_x = ['TS', 'STS', 'STY', 'VSTY', 'Violent TY']
# 自定义 x轴 的取值：
# plt.xticks()
# axes.set_xticklabels(range(len(colums_x)), colums_x)
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 20,
        }
axes.set_yticks([-20, -10, 0, 10, 20])
axes.set_yticklabels(['-20', '-10', '0', '10', '20'])

tips.boxplot(column='Bias(m/s)', by='TC Category', ax=axes)
axes.set_title(' ')
plt.tick_params(labelsize=18)
plt.xlabel('TC Category ', font)
plt.ylabel('Bias(m/s)', font)
plt.grid(linestyle='-.')
# tips.boxplot(column='bias', by='CAT')
# # column参数表示要绘制成箱形图的数据，可以是一列或多列
# by参数表示分组依据
fig.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27\PIC\box.tif', dpi=900)
fig.savefig(r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27\PIC\box.eps')
plt.show()