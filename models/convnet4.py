from collections import OrderedDict

import torch.nn as nn

# import torch.nn.functional as F
# import torch
from util.reshapemodule import ReshapeBatch


class ConvNet4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4, self).__init__()
        # self.nonlin = nonlin
        self.use_bn = use_bn
        self.conv1_size = 32  # 64 #32
        self.conv2_size = 64  # 128 #64
        self.fc1_size = 1024  # 200 #500 #1024
        self.fc2_size = 10  # 1024 #200 #500 #1024

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, kernel_size=5, padding=3)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('nonlin1', nonlin())
        ])

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=5, padding=2)),
            ('maxpool2', nn.MaxPool2d(2)),
            ('nonlin2', nonlin()),
        ])

        block3 = OrderedDict([
            ('batchnorm1', nn.BatchNorm2d(self.conv2_size)),
            ('reshape1', ReshapeBatch(-1)),
            ('fc1', nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) * self.conv2_size, self.fc1_size)),
            ('nonlin3', nonlin()),
        ])

        block4 = OrderedDict([
            ('batchnorm2', nn.BatchNorm1d(self.fc1_size)),
            ('fc2', nn.Linear(self.fc1_size, self.fc2_size))
        ])

        if not self.use_bn:
            del block3['batchnorm1']
            del block4['batchnorm2']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4))
        ]))

        # self.nl1 = nonlin()
        # self.nl2 = nonlin()
        # self.nl3 = nonlin()
        # self.conv1 = nn.Conv2d(input_shape[0], self.conv1_size, kernel_size=5, padding=3)
        # self.conv2 = nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=5, padding=2)
        # if use_bn:
        #     self.conv2_bn = nn.BatchNorm2d(self.conv2_size)
        #     self.fc1_bn = nn.BatchNorm1d(self.fc1_size)
        # self.fc1 = nn.Linear(input_shape[1] // 4 * input_shape[2] // 4 * self.conv2_size, self.fc1_size)
        # if self.fc1_size != 10:
        #     self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        #     # if self.fc2_size != 10:
        #     #     self.fc3 = nn.Linear(self.fc2_size, 10)

    def forward(self, x):
        # x = F.max_pool2d(self.conv1(x), 2)
        # x1 = self.nl[0](x)  # conv1 -> maxpool -> nonlinearity
        # x2 = F.max_pool2d(self.conv2(x1), 2)
        # x3 = self.nl[1](x2)  # conv2 -> maxpool -> nonlinearity
        # if self.use_bn:  # batchnorm1
        #     x3 = self.conv2_bn(x3)
        # x3 = x3.view(x3.size(0), -1)
        # y = self.fc1(x3)
        # y2 = self.nl[2](y)  # fully-connected 1 -> nonlinearity
        # z = self.fc1_bn(y2) if self.use_bn else y2  # batchnorm2
        # if self.fc1_size != 10:
        #     # z = F.dropout(z, training=self.training) # dropout
        #     z = self.fc2(z)
        #     if self.fc2_size != 10:
        #         # z = self.nonlin(self.fc2(z))
        #         z = self.nl[3](self.fc2(z))
        #         z = F.dropout(z, training=self.training)
        #         z = self.fc3(z)  # fully-connected 2
        # return z
        x = self.all_modules(x)
        return x

    # def conv_parameters(self):
    #     for name, param in self.named_parameters():
    #         if name.startswith('conv'):
    #             yield param
    #
    # def nonconv_parameters(self):
    #     for name, param in self.named_parameters():
    #         if not name.startswith('conv'):
    #             yield param
