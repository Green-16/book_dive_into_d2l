#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : dingding
# @Time :  
# @Describe :

import torch
from torch import nn
import d2l.torch as d2l


# Nin

def nin_block(input_channels, output_channels, kernel_size, stride, padding):
    a = nn.Conv2d(input_channels, output_channels, kernel_size,
                  stride, padding)
    b = nn.Conv2d(output_channels, output_channels, kernel_size=(1, 1), stride=1, padding=0)
    c = nn.Conv2d(output_channels, output_channels, kernel_size=(1, 1), stride=1, padding=0)
    return nn.Sequential(a, nn.ReLU(), b, nn.ReLU(), c, nn.ReLU())


input_channels = 1  # 黑白图片
net = nn.Sequential(
    *nin_block(input_channels, output_channels=96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    *nin_block(input_channels=96, output_channels=256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    *nin_block(input_channels=256, output_channels=384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    *nin_block(input_channels=384, output_channels=10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())

# batch_size,channel,height,width
x = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, x.shape)

# 在kaggle上运行
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

