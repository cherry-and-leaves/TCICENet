import cv2
import math
import numpy as np
from PIL import Image
import os
# pdb仅仅用于调试，不用管它
import pdb
# path_ = os.path.abspath('.')
path_image = r'D:\1WXJ\DATA\WXJ_images\train_leibie'
Old_path = path_image + r"\1and2_256/"
save_path = path_image + "/1and2_256_rotate/"
count = 0
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in os.listdir(Old_path):
    Olddir = os.path.join(path_image, i)
    if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        continue
    i_split = i.split('.')
    im = Image.open(Old_path + i)
    out = im.transpose(Image.ROTATE_90)
    # out = im.transpose(Image.FLIP_TOP_BOTTOM)
    out.save(save_path + i_split[0] + '_ROTATE_90d.jpg')
    # out.save(save_path + i_split[0] + '_TOP_BOTTOM.jpg')
