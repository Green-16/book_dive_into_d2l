#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : ztf
# @Time :  
# @Describe :

from torch.nn import functional as F

import torch
from torch import nn
import d2l.torch as d2l

#  GoogleNet
# def googleNet_block(in_channels, c1, c2, c3, c4):
#     #  实现了一种并行(并联操作，常规都是串行）。各种型号的kernel 都要了，兵分四路，然后合到一起
#
#     p1 = nn.Conv2d(in_channels, out_channels=c1, kernel_size=(1, 1))
#     p1 = nn.Sequential(p1, nn.ReLU())
#
#     p2 = nn.Conv2d(in_channels, out_channels=c2[0], kernel_size=(1, 1))
#     p2 = nn.Sequential(p2, nn.ReLU(), nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())
#
#     p3 = nn.Conv2d(in_channels, out_channels=c3[0], kernel_size=1)
#     p3 = nn.Sequential(p3, nn.ReLU(), nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())
#
#     p4 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
#     p4 = nn.Sequential(p4, nn.Conv2d(in_channels, c4, kernel_size=(1, 1)), nn.ReLU())
#
#     # 这种写法有问题，不对。这里应该返回的是module, torch.cat 是对tensor进行操作，不是对网络结构进行操作。
#     # 所以还是应该按照书上写的，把torch.cat写道forward函数里面。
#     # 所以这场并不是 网络结构上面并行，而是tensor上合并
#     return torch.cat((p1, p2, p3, p4), dim=1)


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, out_channels=c1, kernel_size=(1, 1))
        self.p1 = nn.Sequential(self.p1_1, nn.ReLU())

        self.p2_1 = nn.Conv2d(in_channels, out_channels=c2[0], kernel_size=(1, 1))
        self.p2 = nn.Sequential(self.p2_1, nn.ReLU(), nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())

        self.p3_1 = nn.Conv2d(in_channels, out_channels=c3[0], kernel_size=1)
        self.p3 = nn.Sequential(self.p3_1, nn.ReLU(), nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.p4 = nn.Sequential(self.p4_1, nn.Conv2d(in_channels, c4, kernel_size=(1, 1)), nn.ReLU())

    def forward(self, x):
        t1 = self.p1(x)
        t2 = self.p2(x)
        t3 = self.p3(x)
        t4 = self.p4(x)
        # print(t1.shape)
        # print(t2.shape)
        # print(t3.shape)
        # print(t4.shape)
        return torch.cat((t1, t2, t3, t4), dim=1)

#
# class Inception(nn.Module):
#     # c1--c4是每条路径的输出通道数
#     def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
#         super(Inception, self).__init__(**kwargs)
#         # 线路1，单1x1卷积层
#         self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
#         # 线路2，1x1卷积层后接3x3卷积层
#         self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
#         self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
#         # 线路3，1x1卷积层后接5x5卷积层
#         self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
#         self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
#         # 线路4，3x3最大汇聚层后接1x1卷积层
#         self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
#
#     def forward(self, x):
#         p1 = F.relu(self.p1_1(x))
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
#         # 在通道维度上连结输出
#         return torch.cat((p1, p2, p3, p4), dim=1)

# in_channels = 1
b1 = nn.Sequential(
    # 第一个模块
    nn.Conv2d(1, 64, (7, 7), stride=2, padding=3), nn.ReLU(),
    nn.MaxPool2d((3, 3), stride=2, padding=1))
b2 = nn.Sequential(
    # 第二个模块
    nn.Conv2d(64, 64, kernel_size=(1, 1)), nn.ReLU(),
    nn.Conv2d(64, 64 * 3, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
)
b3 = nn.Sequential(Inception(64 * 3, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, '*****\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
