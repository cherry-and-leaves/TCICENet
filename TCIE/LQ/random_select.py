import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num',type=int,help="img numbers to random",default=2000)
args = parser.parse_args()

import random
import os
path = r'D:\1WXJ\DATA\WXJ_images\train_leibie\class1_lunwen'
pathsave = r'F:\1WXJ\train_GUO'
imgs = []
for x in os.listdir(path):
    if x.endswith('jpg'):
        imgs.append(x)
selected_imgs=random.sample(imgs, k=args.num)
print(path)

from shutil import copyfile
for img in selected_imgs:
    src=os.path.join(path, img)
    dst=os.path.join(pathsave, img)
    copyfile(src,dst)
print("copy done")