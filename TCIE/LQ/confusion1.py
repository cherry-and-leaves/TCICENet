from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import os
import matplotlib.pyplot as plt
path = r'D:\1WXJ\Estimate\plot_2\WXJ_WXJNET_256_170_ZJ2\tongji_2_27\PIC'
plt.rcParams['figure.figsize'] = (4.0, 4.0) # 设置figure_size尺寸
fig, axes = plt.subplots()
font = {'family': 'Arial', 'weight': 'normal', 'size':12}

guess = [1, 2, 1, 2, 1, 0, 1, 2, 1, 0]
fact = [0, 1, 2, 1, 2, 1, 2, 1, 0, 1]
classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
confusion = \
[[1034, 126, 3],
 [115, 229, 71],
 [0, 35, 217]]
# plt.subplots_adjust(left=-0.5, right = 2.5,
#                     bottom=-0.5, top=2.5)
# plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,
#                     bottom=0.13, top=0.91)

plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
classes = ['TS+STS', 'STY', 'VSTY+ViolentTY']
plt.tick_params(labelsize=12)
plt.xticks(indices, classes, rotation=45)
plt.yticks(indices, classes, rotation=45)
plt.colorbar()
plt.xlabel('Predict classes', font)
plt.ylabel('Actual classes', font)
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index], horizontalalignment='center')
plt.savefig(os.path.join(path,"conf.tif"),dpi=800)
plt.savefig(os.path.join(path,"conf.eps"),dpi=800)
plt.show()
