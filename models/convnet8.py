# architecture replicated from DoReFaNet code at
# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/svhn-digit-dorefa.py

from collections import OrderedDict
import torch
import torch.nn as nn
from activations import CAbs
from util.reshapemodule import ReshapeBatch


class ConvNet8(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=True, input_shape=(3, 40, 40), no_step_last=False):
        super(ConvNet8, self).__init__()
        self.use_bn = use_bn
        bias = not use_bn

        if input_shape[1] == 40:
            pad0 = 0
            ks6 = 5
        elif input_shape[1] == 32:
            pad0 = 2
            ks6 = 4
        else:
            raise NotImplementedError('no other input sizes are currently supported')

        # TODO: DoReFaNet uses a weird activation: f(x) = min(1, abs(x))

        block0 = OrderedDict([
            ('conv0', nn.Conv2d(3, 48, kernel_size=5, padding=pad0, bias=True)),  # padding = valid
            ('maxpool0', nn.MaxPool2d(2)),  # padding = same
            ('nonlin1', nonlin())  # 18
        ])

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=bias)),  # padding = same
            ('batchnorm1', nn.BatchNorm2d(64, eps=1e-4)),
            ('nonlin1', nonlin()),
        ])

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias)),  # padding = same
            ('batchnorm2', nn.BatchNorm2d(64, eps=1e-4)),
            ('maxpool1', nn.MaxPool2d(2)),      # padding = same
            ('nonlin2', nonlin()),  # 9
        ])

        block3 = OrderedDict([
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=bias)),  # padding = valid
            ('batchnorm3', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin3', nonlin()),  # 7
        ])

        block4 = OrderedDict([
            ('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias)),  # padding = same
            ('batchnorm4', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin4', nonlin()),
        ])

        block5 = OrderedDict([
            ('conv5', nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=bias)),  # padding = valid
            ('batchnorm5', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin5', nonlin()),  # 5
        ])

        block6 = OrderedDict([
            ('dropout', nn.Dropout2d()),
            ('conv6', nn.Conv2d(128, 512, kernel_size=ks6, padding=0, bias=bias)),  # padding = valid
            ('batchnorm6', nn.BatchNorm2d(512, eps=1e-4)),
            ('nonlin6', nonlin() if not no_step_last else CAbs()),
            # ('nonlin6', nonlin() if not relu_last_layer else nn.ReLU()),
        ])

        block7 = OrderedDict([
            ('reshape_fc1', ReshapeBatch(-1)),
            ('fc1', nn.Linear(512, 10, bias=True))
        ])

        if not self.use_bn:
            del block1['batchnorm1']
            del block2['batchnorm2']
            del block3['batchnorm3']
            del block4['batchnorm4']
            del block5['batchnorm5']
            del block6['batchnorm6']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block0', nn.Sequential(block0)),
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4)),
            ('block5', nn.Sequential(block5)),
            ('block6', nn.Sequential(block6)),
            ('block7', nn.Sequential(block7)),
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        # for m in self.all_modules:
        #     print('x:', x.size())
        #     x = m(x)
        # print('x:', x.size())
        return x
