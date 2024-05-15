#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : dingding
# @Time :  
# @Describe :

# VGG

import torch
from torch import nn
import d2l.torch as d2l


def vgg_block(num_convs,in_channels,out_channels):
    layers1=[]
    for _ in range(num_convs):
        layers1.append(nn.Conv2d(in_channels,out_channels,(3,3),stride=1,padding=1))
        layers1.append( nn.ReLU(),)
        in_channels=out_channels
    # 只在外层加了一次池化
    layers1.append(nn.MaxPool2d(kernel_size=(2,2),stride=2))

    return nn.Sequential(*layers1)

# 下面这个list 告诉了 vgg的架构。
# 第1个块 有几个卷积，输出的通道数大小。 第2个块，。。第5个块。
# 这个块数没法变了，卷积层个数可以变。这是因为一个块用了一次池化，一次池化像素维度减半。
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 224-1 /2 =  112
# 112-1 /2 = 56
# 56-1 /2 = 28
# 28-1/2 =14
# 14-1/2 =7

def vgg(conv_arch):
    conv_blks=[]
    in_channels=1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net=nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels *7*7 ,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
    )
    return net


net = vgg(conv_arch)

x=torch.rand(size=(1,1,224,224),dtype=torch.float32)
for blk in net:
    x=blk(x)
    print(blk.__class__.__name__, 'output shape:\t', x.shape)

X = torch.randn(size=(1, 1, 12, 12))
c=nn.MaxPool2d(kernel_size=(3,3),stride=2)(X) # 这里就是很怪异，池化计算维度 （x-k+1)/2
c=nn.MaxPool2d(kernel_size=(2,2),stride=2)(X) # 这里就是 x/2 =6
c=nn.MaxPool2d(kernel_size=(4,4),stride=2)(X) # 这里就是 =5
c.shape

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

import matplotlib.pyplot as plt
plt.show()