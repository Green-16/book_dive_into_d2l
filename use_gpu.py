#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : dingding
# @Time :  
# @Describe :
import sys
import torch
from torch import nn
sys.modules.get("torch._meta_registrations", None)
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')

torch.cuda.device_count()
# 只有一个

torch.cuda.is_available()


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()




