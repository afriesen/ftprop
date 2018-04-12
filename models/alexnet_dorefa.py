# adapted from the pytorch vision AlexNet implementation found at
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# but re-structured to match the AlexNet implementation used in the DoReFaNet
# paper: https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py

from collections import OrderedDict

import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

from util.reshapemodule import ReshapeBatch

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, nonlin=nn.ReLU, no_step_last=False, use_bn=True, num_classes=1000, data_parallel=False):
        super(AlexNet, self).__init__()
        nl_name = 'nonlin'  # nonlin.__name__
        bias = not use_bn

        block0 = OrderedDict([
            ('conv0', nn.Conv2d(3, 96, kernel_size=12, stride=4, padding=0, bias=bias)),  # padding=valid
            ('{}0'.format(nl_name), nonlin()),
        ])

        # TODO: split conv2d for: conv1, conv3, conv4

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=bias)),  # padding=same
            ('batchnorm1', nn.BatchNorm2d(256, eps=1e-4)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),  # padding = same
            ('{}1'.format(nl_name), nonlin()),
        ])

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(256, 384, kernel_size=3, bias=bias, padding=1)),  # padding = same
            ('batchnorm2', nn.BatchNorm2d(384, eps=1e-4)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),  # padding = same
            ('{}2'.format(nl_name), nonlin()),
        ])

        block3 = OrderedDict([
            ('conv3', nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=bias)),  # padding = same
            ('batchnorm3', nn.BatchNorm2d(384, eps=1e-4)),
            ('{}3'.format(nl_name), nonlin()),
        ])

        block4 = OrderedDict([
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=bias)),  # padding = same
            ('batchnorm4', nn.BatchNorm2d(256, eps=1e-4)),
            ('maxpool4', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),  # padding = valid
            ('{}4'.format(nl_name), nonlin()),
        ])

        block5 = OrderedDict([
            ('reshape_fc0', ReshapeBatch(-1)),
            ('fc0', nn.Linear(256 * 6 * 6, 4096, bias=bias)),
            ('batchnorm_fc0', nn.BatchNorm2d(4096, eps=1e-4)),
            ('{}5'.format(nl_name), nonlin()),
        ])

        block6 = OrderedDict([
            ('fc1', nn.Linear(4096, 4096, bias=bias)),
            ('batchnorm_fc1', nn.BatchNorm2d(4096, eps=1e-4)),
        ])

        if not no_step_last:
            block6['{}6'.format(nl_name)] = nonlin()
            block7 = OrderedDict()
            final_block = block7
        else:
            block6['ReLU1'] = nn.ReLU()
            block7 = None
            final_block = block6

        final_block['fc2'] = nn.Linear(4096, num_classes, bias=True)

        if not use_bn:
            del block1['batchnorm1']
            del block2['batchnorm2']
            del block3['batchnorm3']
            del block4['batchnorm4']
            del block5['batchnorm_fc0']
            del block6['batchnorm_fc1']

        # self.blocks = [block0, block1, block2, block3, block4, block5, block6, block7]

        if data_parallel:
            conv_layers = nn.DataParallel(nn.Sequential(OrderedDict([
                ('block0', nn.Sequential(block0)),
                ('block1', nn.Sequential(block1)),
                ('block2', nn.Sequential(block2)),
                ('block3', nn.Sequential(block3)),
                ('block4', nn.Sequential(block4)),
            ])))

            fc_layers = OrderedDict([
                ('block5', nn.Sequential(block5)),
                ('block6', nn.Sequential(block6)),
            ])
            if block7 is not None:
                fc_layers['block7'] = nn.Sequential(block7)

            self.layers = nn.Sequential(conv_layers, nn.Sequential(fc_layers))
        else:
            od_layers = OrderedDict([
                ('block0', nn.Sequential(block0)),
                ('block1', nn.Sequential(block1)),
                ('block2', nn.Sequential(block2)),
                ('block3', nn.Sequential(block3)),
                ('block4', nn.Sequential(block4)),
                ('block5', nn.Sequential(block5)),
                ('block6', nn.Sequential(block6)),
            ])

            if block7 is not None:
                od_layers['block7'] = nn.Sequential(block7)

            self.layers = nn.Sequential(od_layers)

    def forward(self, x):

        # i = 0
        # for b in self.blocks:
        #     for l in b.values():
        #         print(i, ':', x.size())
        #         x = l(x)
        #         i += 1

        x = self.layers(x)
        return x


# def alexnet(pretrained=False, nonlin=nn.ReLU, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         nonlin (Module name): The name of the Module to create for each non-linearity instance.
#     """
#     assert nonlin == nn.ReLU or not pretrained, 'pre-trained AlexNet only supports ReLU non-linearities'
#     model = AlexNet(nonlin=nonlin, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
#     return model
