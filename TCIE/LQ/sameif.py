import os
TS = 0
STS = 0
TP = 0
VTP = 0
STP = 0
path_image = r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\2017-2019\really_result\21/'
path_imageA = r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\2017-2019\really_result\B_1'
filewrite = open(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\2017-2019\really_result\shan_21.txt','w')
filelist = os.listdir(path_image) #该文件夹下所有的文件（包括文件夹）
for file in filelist:   #遍历所有文件

    if os.path.exists(os.path.join(path_imageA, file )): # True/False
        print('exists')
    else:
        print(file)
        filename = os.path.splitext(file)[0]  # 文件名
        wind = int(filename.split('_')[2])
        if wind <= 47:
            TS = TS + 1
            intensitytype = ' 0'

        elif wind >= 48 and wind <= 63:
            STS = STS + 1
            intensitytype = ' 1'

        elif wind >= 64 and wind <= 80:
            TP = TP + 1
            intensitytype = ' 2'

        elif wind >= 81 and wind <= 100:
            VTP = VTP + 1
            intensitytype = ' 3'

        else:
            STP = STP + 1
            intensitytype = ' 4'
        filewrite.write(file + intensitytype)
        filewrite.write('\n')
print('TS={}, STS={}, TP={}, VTP={}, STP={}'.format(TS, STS, TP, VTP, STP))


