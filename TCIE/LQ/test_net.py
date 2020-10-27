import torch
import os
import numpy as np
from my_transform import demension_reduce
from my_image_folder import ImageFolder
from torch.autograd import Variable
from my_transform import transform
# from define_net import Net
from define_net_WXJ import Net
from define_net_WXJ import Net_20
from define_net_WXJ import Net_80
from torch.autograd import Variable

if __name__ == '__main__':
        
    path_ = os.path.abspath('.')

    net = Net_80()
    net.load_state_dict(torch.load(r'D:\1WXJ\Estimate\Model\WXJNet_3_256_80/70_net.pth')) # your net

    testset = ImageFolder(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\predict_devide\3and4_256_80/',transform) # your test set

    f = open(r'D:\1WXJ\Estimate\plot\WXJNet_3_256_80_JH2/result_3_70.txt') # where to write answer

    tys = {} # map typhoon to its max wind
    tys_time = {} # map typhoon-time to wind

    for i in range(0,testset.__len__()):
        
        image, actual = testset.__getitem__(i)
        image = image.expand(1,image.size(0),image.size(1),image.size(2)) # a batch with 1 sample
        name = testset.__getitemName__(i)
        
        output = net(Variable(image))
        wind = output.data[0][0]  # output is a 1*1 tensor
        name = name.split('_')

        tid = name[0]
        if tys.__contains__(tid):
            if tys[tid] < wind:
                tys[tid] = wind
        else :
            tys[tid] = wind

        tid_time = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]
        tys_time[tid_time] = wind
        
        if i % 100 == 99 :
            print ('have processed ',i+1,' samples.')

    tys = sorted(tys.items(),key=lambda asd:asd[1],reverse=True)
    for ty in tys:
        print(ty)# show the sort of typhoons' wind

    tys_time = sorted(tys_time.items(),key=lambda asd:asd[0],reverse=False)
    for ty in tys_time:
        f.write(str(ty)+'\n') # record all result by time
    f.close()

    path_ = os.path.abspath(r'D:\1WXJ\Estimate\plot\WXJNet_3_256_80_JH2')
    f = open(path_ + '/result_3_70.txt')

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())


    def IndexNotensor(f):
        all_mae = 0
        all_mse = 0
        bias = 0
        count = 0
        wind_real_list = []
        wind_pre_list = []
        for line in f:
            print(line)
            line1 = line.split(',')[1]
            line2 = line1.split(")")[0]
            wind_pre = float(line2)
            wind_real = line.split('_')[2]
            wind_real = float(wind_real)
            bias = bias + (wind_pre - wind_real)
            all_mae = all_mae + abs(wind_pre - wind_real)
            all_mse = all_mse + (wind_pre - wind_real) ** 2
            wind_real_list.append(wind_real)
            wind_pre_list.append(wind_pre)
            count = count + 1
        f.close()
        MAE = all_mae / count
        Bias = bias / count
        RMSE = all_mse / count
        RMSE = np.sqrt(RMSE)
        # RMSE = rmse(wind_pre, wind_real)
        print(MAE, RMSE, Bias)
        return MAE, RMSE, Bias



    def Index(f):
        all_mae = 0
        all_mse = 0
        count = 0
        wind_real_list = []
        wind_pre_list = []
        for line in f:
            print(line)
            line1 = line.split(',')[1]
            line2 = line1.split(")")[0]
            wind_pre = float(line2)
            wind_real = line.split('_')[2]
            wind_real = float(wind_real)
            bias = bias + wind_pre - wind_real
            all_mae = all_mae + abs(wind_pre - wind_real)
            all_mse = all_mse + (wind_pre - wind_real) ** 2
            wind_real_list.append(wind_real)
            wind_pre_list.append(wind_pre)
            count = count + 1
        f.close()
        MAE = all_mae / count
        Bias = bias / count
        RMSE = all_mse / count
        RMSE = np.sqrt(RMSE)
        # RMSE = rmse(wind_pre, wind_real)
        print(MAE, RMSE, Bias)
        return MAE, RMSE, Bias

    #
    MAE, RMSE, Bias = IndexNotensor(f)
    print(MAE, RMSE, Bias)
