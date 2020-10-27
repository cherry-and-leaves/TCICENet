import os
from PIL import Image
import cv2

base_dir = r"D:\1WXJ\DATA\CLASS_Japan\devide_3\test\PCJ\STP+VTP_C\\"
filename = os.listdir(base_dir)
new_dir = r"D:\1WXJ\DATA\CLASS_Japan\devide_3\test\PCJ\STP+VTP_200\\"
if not os.path.isdir(new_dir):
    os.makedirs(new_dir)
size_m = 200
size_n = 200
count = 1
y1 = 20+128
y2 = 236+128
# y1 = 20
# y2 = 236
# y1 = 58
# y2 = 198
# y1 = 105
# y2 = 405
for img in filename:

    # image = cv2.imread(base_dir + img)
    # region = image[y1:y2, y1:y2]
    # cv2.imwrite(new_dir + img, region)
    image = Image.open(base_dir + img)
    image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    image_size.save(new_dir + img)
    # count = count + 1
    # if count == 500:
    #     break

