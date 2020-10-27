#每个强度台风统计，放在各自文件夹

from PIL import Image
import os
import shutil
path_ = r'D:\1WXJ\DATA\WXJ_images\train_leibie'
# path_image = os.path.join(path_, r"1and2_256")
# path_image = os.path.join(path_, r"3TP_256")
# path_image = os.path.join(path_, r"4VTP_256")
path_image = os.path.join(path_, r"5STP_256")
filelist = os.listdir(path_image) #该文件夹下所有的文件（包括文件夹）

for file in filelist:   #遍历所有文件
    Olddir = os.path.join(path_image, file)  # 原来的文件路径
    Olddir_split = Olddir.split('/')
    if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        continue
    filename = os.path.splitext(file)[0]  # 文件名
    wind = filename.split('_')[2]
    pathsave = wind
    filetype=os.path.splitext(file)[1]   #文件扩展名
    save_path = os.path.join(path_ + r"\all", pathsave)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Newdir=os.path.join(save_path, file)  #
    shutil.copy(Olddir, Newdir)#重命名



