#统计原始的台风，每个类别有多少张
#训练集tys_raw_1979-2013，选择1979-2013，进行旋转，加噪声，样本均衡
#生成需要的标签txt，路径文件名 等级
from PIL import Image
import os
import shutil
path_ = os.path.abspath('.')
path_image = r'D:\deep_typhoon-master2\WXJ_deep_typhoon-master\tys_2018-2019/'
filelist = os.listdir(path_image) #该文件夹下所有的文件（包括文件夹）
TS = 0
STS = 0
TP = 0
VTP = 0
STP = 0

for file in filelist:   #遍历所有文件
    Olddir = os.path.join(path_image, file)  # 原来的文件路径
    Olddir_split = Olddir.split('/')
    if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        continue
    filename = os.path.splitext(file)[0]  # 文件名
    wind = int(filename.split('_')[2])
    if wind == 0:
        filetype=os.path.splitext(file)[1]   #文件扩展名
        Newdir=os.path.join(r'D:\deep_typhoon-master2\WXJ_deep_typhoon-master\tys_2018-2019_0/', file)  #
        os.rename(Olddir, Newdir)#重命名
print('TS={}, STS={}, TP={}, VTP={}, STP={}'.format(TS, STS, TP, VTP, STP))


