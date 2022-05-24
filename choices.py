
from pspnet import PSPNet
from fcn import FCN16s
from resunet import ResUnet
from unetpp import UNet_Nested

import torch
import torch.nn as nn

from torchsummary import summary

from lr_scheduler import *


def get_lr_scheduler(optimizer, max_iters, sch_name):
    if sch_name == 'warmup_poly':
        return WarmupPolyLR(optimizer, max_iters=max_iters, power=0.9, warmup_factor=float(1.0/3), warmup_iters=0, warmup_method='linear')
    else:
        return None


def get_optimizer(net, optim_name):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters())
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters())
    return optimizer


def get_criterion(out_channels, class_weights=None):
    if out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    return criterion


def choose_net(name, out_channels):
    if name == 'fcn':
        return FCN16s(n_class=out_channels)
    elif name == 'pspnet':
        return PSPNet(n_class=out_channels)
    elif name == 'unetpp':
        return UNet_Nested(n_class=out_channels)
    elif name == 'resunet':
        return ResUnet(n_class=out_channels)


if __name__ == '__main__':
    net_names = [
        'fcn',
    ]
    resizes = [
        (528, 960),
    ]

    summary(choose_net(net_names[0]).cuda(), (3, resizes[0][0], resizes[0][1]))
