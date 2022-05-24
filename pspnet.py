'''
[description]
PSPNet
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict

# import torchvision.models.resnet18 as resnet18
# from torchvision.models.vgg import VGG
from vgg import VGGNet

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out

class PSPNet(nn.Module):
    def __init__(self, n_class=6, bn_momentum=0.01):
        super(PSPNet, self).__init__()
        # self.Resnet101 = resnet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        # self.psp_layer = PyramidPooling('psp', n_class, 2048, norm_layer=nn.BatchNorm2d)

        self.vgg16 = VGGNet(pretrained=True, model='vgg16')
        self.psp_layer = PyramidPooling('psp', n_class, 512, norm_layer=nn.BatchNorm2d)

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.vgg16(input)['x5']
        psp_fm = self.psp_layer(x)
        pred = F.interpolate(psp_fm, size=input.size()[2:4], mode='bilinear', align_corners=True)

        return pred

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    h = 256
    w = 256
    num_classes = 6

    net = PSPNet(n_class=num_classes).cuda()

    image = (3, h, w)
    f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # print(f, p)

    input = torch.randn(4, 3, h, w).cuda()
    print('input', input.shape)
    with torch.no_grad():
        output = net(input)
    print('output', output.shape)