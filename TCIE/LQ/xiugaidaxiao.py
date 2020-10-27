import os
from PIL import Image
import cv2
filename = os.listdir(r"D:\1WXJ\DATA\WXJ_images\train_leibie\4VTP\\")
base_dir = r"D:\1WXJ\DATA\WXJ_images\train_leibie\4VTP\\"
new_dir = r"D:\1WXJ\DATA\WXJ_images\train_leibie\4VTP_256\\"
if not os.path.isdir(new_dir):
    os.makedirs(new_dir)
size_m = 256
size_n = 256
count = 1
y1 = 128
y2 = 384
# y1 = 58
# y2 = 198
# y1 = 105
# y2 = 405
for img in filename:
    # image = Image.open(base_dir + img)
    image = cv2.imread(base_dir + img)
    region = image[y1:y2, y1:y2]
    cv2.imwrite(new_dir + img, region)
    # image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    # image_size.save(new_dir + img)
    # count = count + 1
    # if count == 500:
    #     break

