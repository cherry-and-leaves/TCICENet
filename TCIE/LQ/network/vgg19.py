import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['vgg19', 'vgg19_bn']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

import torch.nn as nn
import torch

class vgg19_Net(nn.Module):
    def __init__(self, in_img_rgb=3, in_img_size=64, out_class=1000, in_fc_size=25088):
        super(vgg19_Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_img_rgb, out_channels=in_img_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_img_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.fc17 = nn.Sequential(
            nn.Linear(in_features=in_fc_size, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc18 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc19 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=out_class, bias=True)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8,
                          self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14, self.conv15,
                          self.conv16]

        self.fc_list = [self.fc17, self.fc18, self.fc19]

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        fc = x.view(x.size(0), -1)
        # 查看全连接层的参数：in_fc_size  的值
        # print("vgg19_model_fc:",fc.size(1))
        for fc_item in self.fc_list:
            fc = fc_item(fc)
        return fc

def vgg19(pretrained=False, modelpath='./models',**kwargs):
    model = vgg19_Net(in_img_rgb=1, in_img_size=81, out_class=1, in_fc_size=2048).cuda()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_dir=modelpath))
    return model

def vgg19_bn(pretrained=False, modelpath='./models',**kwargs):
    model = vgg19_Net(in_img_rgb=1, in_img_size=81, out_class=1, in_fc_size=2048).cuda()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn'], model_dir=modelpath))
    return model
