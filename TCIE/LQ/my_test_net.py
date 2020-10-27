import torch
import os
from my_transform import demension_reduce
from my_image_folder import ImageFolder
from torch.autograd import Variable
from my_transform import transform
from define_net import Net
from torch.autograd import Variable
import numpy as np
from network import inceptionresnetv2
import torch.utils.data
import torch.nn as nn
if __name__ == '__main__':
        
    path_ = os.path.abspath('.')
    net = inceptionresnetv2(pretrained=False)
    net.last_linear = nn.Linear(1536, 1)
    net.cuda()
    net.load_state_dict(torch.load(r'D:\1WXJ\Estimate\Model\LQ_ZCJ_4/440_net.pth')) # your net
    # net = torch.load(r'D:\1WXJ\Estimate\Model\LQ_MODEL/10_net_relu.pth')
    testset = ImageFolder(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\predict_devide\3/',transform) # your test set
    net.eval()
    f = open(r'D:\1WXJ\Estimate\plot\LQ_ZCJ/result_3_net_440.txt','w') # where to write answer

    tys = {} # map typhoon to its max wind
    tys_time = {} # map typhoon-time to wind

    for i in range(0,testset.__len__()):
        
        image, actual = testset.__getitem__(i)
        image = image.expand(1,image.size(0),image.size(1),image.size(2)) # a batch with 1 sample
        name = testset.__getitemName__(i)
        image = image.cuda()
        output = net(Variable(image))
        wind = output.data[0][0] # output is a 1*1 tensor
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
