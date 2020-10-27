from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Deep_PHURIE(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Deep_PHURIE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(5, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool4 = nn.MaxPool2d(3, 1)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.pool5 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.pool6 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.fc1 = nn.Linear(1152, 512)#20 * 6 * 6
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        # print(x.shape)
        x = self.pool4(F.leaky_relu(self.conv4(x)))
        # print(x.shape)
        x = self.pool5(F.leaky_relu(self.conv5(x)))
        # print(x.shape)
        x = self.pool6(F.leaky_relu(self.conv6(x)))
        x = F.leaky_relu(x)
        # print(x.shape)
        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_20(nn.Module):
    def __init__(self):
        super(Net_20, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 3)
        self.conv2_2 = nn.Conv2d(20, 20, 3)
        self.pool2 = nn.MaxPool2d(1, 1)

        self.fc1 = nn.Linear(20 * 3 * 3, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_40_WXJ(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_40_WXJ, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64, 512)#20 * 6 * 6
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        # print(x.shape)
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        # print(x.shape)
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        x = F.leaky_relu(x)
        # print(x.shape)

        print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_50(nn.Module):
    def __init__(self):
        super(Net_50, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 5)
        self.conv2_2 = nn.Conv2d(20, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(20 * 5 * 5, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_80(nn.Module):
    def __init__(self):
        super(Net_80, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 7)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(20 * 4 * 4, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_110(nn.Module):
    def __init__(self):
        super(Net_110, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 7)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(20 * 8 * 8, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(6, 6)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 4)
        self.pool2 = nn.MaxPool2d(5, 5)

        self.fc1 = nn.Linear(20 * 2 * 2, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_170(nn.Module):
    def __init__(self):
        super(Net_170, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(8, 32, 6)
        self.conv2_2 = nn.Conv2d(32, 64, 6)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(self.num_flat_features(x), 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))

        x = x.view(-1, self.num_flat_features(x))
        print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_170_WXJ(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_170_WXJ, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(10816, 512)#20 * 6 * 6
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        # print(x.shape)
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        # print(x.shape)
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        x = F.leaky_relu(x)
        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_140_WXJ(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_140_WXJ, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(6400, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        # print(x.shape)
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        # print(x.shape)
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_200(nn.Module):
    def __init__(self):
        super(Net_200, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 6)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(20 * 5 * 5, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_230(nn.Module):
    def __init__(self):
        super(Net_230, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 6)
        self.pool2 = nn.MaxPool2d(5, 5)

        self.fc1 = nn.Linear(20 * 5 * 5, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_256(nn.Module):
    def __init__(self):
        super(Net_256, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 11)
        self.pool1 = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(8, 20, 6)
        self.conv2_2 = nn.Conv2d(20, 20, 6)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(20 * 8 * 8, 80)#20 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))  # better than sigmoid/tanh
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))  # better than sigmoid/tanh
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))  # better than sigmoid/tanh

        x = x.view(-1, self.num_flat_features(x))
        # print(self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_256_WXJ(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_256_WXJ, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_2 = nn.Conv2d(64, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv5_2 = nn.Conv2d(64, 64, 3)


        self.fc1 = nn.Linear(2304, 80)#64 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)

        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.conv4_2(F.leaky_relu(self.conv4(x)))
        x = self.pool4(F.leaky_relu(self.conv4_2(x)))
        x = self.conv5_2(F.leaky_relu(self.conv5(x)))
        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_170_conv5(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_170_conv5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6400, 80)#64 * 5 * 5
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)
        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))
        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Net_170_conv4(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_170_conv4, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6400, 80)#64 * 5 * 5
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)
        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))
        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_170_conv5_256(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_170_conv5_256, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(3200, 1)

    def forward(self, x):
            x = self.pool1(F.leaky_relu(self.conv1(x)))
            x = self.conv2_2(F.leaky_relu(self.conv2(x)))
            x = self.pool2(F.leaky_relu(self.conv2_2(x)))
            x = self.conv3_2(F.leaky_relu(self.conv3(x)))
            # print(x.shape)
            x = self.pool3(F.leaky_relu(self.conv3_2(x)))
            x = self.conv4_2(F.leaky_relu(self.conv4(x)))
            x = self.pool4(F.leaky_relu(self.conv4_2(x)))

            # print(self.num_flat_features(x))
            x = x.view(-1, self.num_flat_features(x))

            x = self.fc(x)
            return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_140_conv5(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_140_conv5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, 3)



        self.fc1 = nn.Linear(1600, 80)#64 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)

        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))

        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_110_conv5(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_110_conv5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)



        self.fc1 = nn.Linear(2304, 80)#64 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)

        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))

        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net_80_conv5(nn.Module):######有关3*3小卷积核
    def __init__(self):
        super(Net_80_conv5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(1600, 80)#64 * 6 * 6
        self.fc2 = nn.Linear(80, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.conv2_2(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv2_2(x)))
        x = self.conv3_2(F.leaky_relu(self.conv3(x)))
        # print(x.shape)

        x = self.pool3(F.leaky_relu(self.conv3_2(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))

        # print(x.shape)

        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features