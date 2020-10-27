import numpy as np
import sys

from define_net_WXJ import Deep_PHURIE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch
import os
from my_transform import transform
from my_image_folder import ImageFolder


def testset_loss(dataset, network):

    loader = torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=2)

    all_loss = 0.0
    for i,data in enumerate(loader,0):

        inputs,labels = data
        inputs = Variable(inputs)

        outputs = network(inputs)
        all_loss = all_loss + abs(labels[0]-outputs.data[0][0])

    return all_loss/i


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
        # print(line)
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

    # print(MAE, RMSE, Bias)
    return MAE, RMSE, Bias



if __name__ == '__main__':
    imgsize = 224
    path_ = os.path.abspath(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\train')
    # trainset = ImageFolder(path_+'/1and2_256/', imgsize, transform)#1and2_256_256_ZJ
    trainset = ImageFolder(path_ + '/train_Z_256_35281/', imgsize, transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32, shuffle=True, num_workers=1)
    testset = ImageFolder(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\2017-2019\Z/',imgsize, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=1)
    pathoutput = r"D:\1WXJ\Estimate\Model_2\Deep_Phure_81_0.001_drop=0.5"#
    pathlosssave = os.path.join(r'D:\1WXJ\Estimate\plot_2\Deep_Phure_81_0.001_drop=0.5')#
    tys_time = {}  # map typhoon-time to wind
    totalloss = []
    test_allloss = []
    max_RMSE = 0
    max_MR = 100
    if not os.path.exists(pathlosssave):
        os.makedirs(pathlosssave)
    if not os.path.exists(pathoutput):
        os.makedirs(pathoutput)

    net = Deep_PHURIE()
    net.cuda()
    init.xavier_uniform(net.conv1.weight.data, gain=1)
    init.constant(net.conv1.bias.data, 0.1)
    init.xavier_uniform(net.conv2.weight.data, gain=1)
    init.constant(net.conv2.bias.data, 0.1)
    init.xavier_uniform(net.conv3.weight.data, gain=1)
    init.constant(net.conv3.bias.data, 0.1)
    init.xavier_uniform(net.conv4.weight.data, gain=1)
    init.constant(net.conv4.bias.data, 0.1)
    init.xavier_uniform(net.conv5.weight.data, gain=1)
    init.constant(net.conv5.bias.data, 0.1)
    print(net)
#定義損失函數·和分類器
    criterion = nn.SmoothL1Loss()
    # criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(200):
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/100))
                totalloss.append(running_loss / 100)
                running_loss = 0.0
        if 10 < epoch < 50:
            if epoch % 2 == 1 :
                torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')
        else:
            if epoch % 20 == 19 :
                torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')
        net.eval()
        all_loss = 0.0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            test_loss = criterion(outputs, labels.float())
            all_loss += test_loss.data[0]
            wind = outputs.data[0][0]  # output is a 1*1 tensor
            # wind = wind.item()
            name = testset.__getitemName__(i)
            name = name.split('_')
            tid_time = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]
            tys_time[tid_time] = wind
        f = open(pathlosssave + r'/result_' + str(epoch + 1) + '.txt', 'a')  # where to write answer
        f2 = open(pathlosssave + r'/test_epoch_loss.txt', 'a')  # where to write answer
        test_allloss.append((all_loss / i))
        tys_time = sorted(tys_time.items(), key=lambda asd: asd[0], reverse=False)
        for ty in tys_time:
            f.write(str(ty) + '\n')  # record all result by time
        f.close()
        f_r = open(pathlosssave + r'/result_' + str(epoch + 1) + '.txt')
        MAE, RMSE, Bias = IndexNotensor(f_r)
        line = 'epoch %d, testloss: %4f MAE: %4f RMSE: %4f Bias: %4f MAE+RMSE: %4f\n' % (epoch + 1, (all_loss / i), MAE, RMSE, Bias, MAE+RMSE)
        print(line)
        if (MAE + RMSE) < max_MR:
            max_MR = MAE+RMSE
            torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')
        tys_time = {}
        f2.write(line)  # record all result by time
        net.train()
    np.savetxt(pathlosssave + r'\total_loss.csv'
               , totalloss, delimiter=',')
    print('Finished Training')
    torch.save(net.state_dict(), pathoutput+'/net_relu.pth')