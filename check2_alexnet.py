#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : dingding
# @Time :  
# @Describe :
# check alexnet

import torch
from torch import nn
import d2l.torch as d2l


net = nn.Sequential(
    # 假如 输入是 224*224
    nn.Conv2d(1,96,kernel_size=(11,11),stride=4,padding=1),nn.ReLU(),  #( x.shape -C_k+ p*2+1 )/Stride =54
    nn.MaxPool2d(kernel_size=(3,3),stride=2), # （54-3+1）/2=26
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96,256,kernel_size=(5,5),padding=2),nn.ReLU(), # （26-5+2*2+1）/1 # 26
    nn.MaxPool2d(kernel_size=(3,3),stride=2),  # （26-3+1）/2 =12
    # 使用三个连续的卷积层和较小的卷积窗口。
    nn.Conv2d(256,384,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(), # （12-3+1+1*2）/1 = 12
    nn.Conv2d(384,384,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),  # 12
    nn.MaxPool2d(kernel_size=(3,3),stride=2), # 12-3+1 /2= 5
    nn.Flatten(), # 256*5*5 =6400  # 算不出来就先把x用随机数替代，跑一下
    nn.Linear(6400,4096),nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,10)
)

x=torch.rand((1,1,224,224))
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__, 'output shape: \t', x.shape)


batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)


lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
import matplotlib.pyplot as plt
plt.show()