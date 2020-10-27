import os
import sys
# sys.path.append('/F:/1xjie/Estimate/LQ/define_net')
from define_net import Net
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch
import torchvision
#
from my_transform import transform
from my_image_folder import ImageFolder
#
# def testset_loss(dataset, network):
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
#
#     all_loss = 0.0
#     for i, data in enumerate(loader, 0):
#         inputs, labels = data
#         inputs = Variable(inputs)
#
#         outputs = network(inputs)
#         all_loss = all_loss + abs(labels[0] - outputs.data[0][0])
#
#     return all_loss / i
#
#
# if __name__ == '__main__':
#     path_ = os.path.abspath('.')
#
#     trainset = ImageFolder(path_ + '/train_set/', transform)
#
#     print(trainset)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
#     testset = ImageFolder(path_ + '/test_set/', transform)
#
#     net = Net()
#     init.xavier_uniform(net.conv1.weight.data, gain=1)
#     init.constant(net.conv1.bias.data, 0.1)
#     init.xavier_uniform(net.conv2.weight.data, gain=1)
#     init.constant(net.conv2.bias.data, 0.1)
#     # net.load_state_dict(torch.load(path_+'net_relu.pth'))
#     print(net)
#
#     criterion = nn.L1Loss()
#
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#
#     for epoch in range(500):
#
#         running_loss = 0.0
#         for i, data in enume
#             rate(trainloader, 0):
#
#             inputs, labels = data
#             inputs, labels = Variable(inputs), Variable(labels)
#
#             optimizer.zero_grad()
#
#             outputs = net(inputs)
#             loss = criterion(outputs, labels.float())
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.data[0]
#             if i % 100 == 99:
#                 print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0
#
#     test_loss = testset_loss(testset, net)
#     print('[%d ] test loss: %.3f' % (epoch + 1, test_loss))
#
#     print('Finished Training')
#     torch.save(net.state_dict(), path_ + '/net_relu.pth')
from define_net import Net
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch
import os
from my_transform import transform
from my_image_folder import ImageFolder
import numpy as np
from network import resnet34, resnet101, inceptionresnetv2
import torch.utils.data
# def testset_loss(testloader, network):
#     # loader = torch.utils.data.DataLoader(dataset,batch_size=8,num_workers=2)
#     net.eval()
#     all_loss = 0.0
#     for i,data in enumerate(testloader,0):
#         inputs,labels = data
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         outputs = network(inputs)
#         all_loss = all_loss + abs(labels[0]-outputs.data[0][0])
#     return all_loss/i
#     net.train()
if __name__ == '__main__':
    #path_ = os.path.abspath('.')
    model_path = r"D:\1WXJ\Estimate\Model\MODEL_49946_Res34/"
    trainset = ImageFolder(r'D:\五分類train\4and5/', transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=8, shuffle=True,num_workers=2)
    testset = ImageFolder(r'D:\1WXJ\DATA\CLASS_Japan\devide_3\test\predict_devide\3and4', transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)
    pathoutput = r"D:\1WXJ\Estimate\Model\MODEL_45_Res34"
    pathlosssave = os.path.join(r'D:\1WXJ\Estimate\plot\plot_45_Res34')
    totalloss = []
    test_allloss = []
    tys_time = {}  # map typhoon-time to wind
    if not os.path.exists(pathlosssave):
        os.makedirs(pathlosssave)
    if not os.path.exists(pathoutput):
        os.makedirs(pathoutput)
    net = resnet34(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
    # model.fc = nn.Linear(512, 6)
    # net = inceptionresnetv2(pretrained=True, modelpath=model_path)
# model.fc = nn.Linear(512*4, 5)##################几类
#     net.last_linear = nn.Linear(1536, 1)
    net.fc = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(512, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1))
    net.cuda()
    # init.xavier_uniform(net.conv1.weight.data, gain=1)
    # init.constant(net.conv1.bias.data, 0.1)
    # init.xavier_uniform(net.conv2.weight.data, gain=1)
    # init.constant(net.conv2.bias.data, 0.1)
    # net.load_state_dict(torch.load(path_+'net_relu.pth'))
    print(net)

#定義損失函數·和分類器
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
    optimizer = optim.Adadelta(net.parameters(), lr=0.1, rho=0.9, eps=1e-06, weight_decay=0.1)
    for epoch in range(500):
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            ## forward
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/100))
                totalloss.append(running_loss / 100)
                running_loss = 0.0
        if epoch % 10 == 9 and epoch > 10:
        # if epoch > -1:
            torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net_0.pth')
            print("save "+ str(epoch + 1) + '_net_0.pth')
        # test_loss = testset_loss(testloader, net)
        # print('[%d ] test loss: %.3f' % (epoch + 1, test_loss))
            # loader = torch.utils.data.DataLoader(dataset,batch_size=8,num_workers=2)
        net.eval()
        all_loss = 0.0
        for i, data in enumerate(testloader,0):
            inputs,labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            test_loss = criterion(outputs, labels.float())
            all_loss += test_loss.data[0]
            wind = outputs.data[0][0] # output is a 1*1 tensor
            # wind = wind.item()
            name = testset.__getitemName__(i)
            name = name.split('_')
            tid_time = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]
            tys_time[tid_time] = wind

        f = open(pathlosssave + r'/result_' + str(epoch+1) + '.txt', 'w')  # where to write answer
        f2 = open(pathlosssave + r'/test_epoch_loss.txt', 'a')  # where to write answer
        print('epoch %d, testloss: ' % (epoch+1))
        print((all_loss / i))
        test_allloss.append((all_loss / i))
        tys_time = sorted(tys_time.items(), key=lambda asd: asd[0], reverse=False)
        for ty in tys_time:
            f.write(str(ty) + '\n')  # record all result by time
        f.close()
        tys_time = {}
        f2.write(str((all_loss / i)) + '\n')  # record all result by time

        net.train()

    np.savetxt(pathlosssave + r'\total_loss_0.csv'
               , totalloss, delimiter=',')
    print('Finished Training')
