# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:38:02 2019

@author: chxy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#scale = 3
class SRCNN_net(torch.nn.Module):
    def __init__(self, num_channels, f1=9, f2=1, f3=5):
        super(SRCNN_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=f1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=f2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=f3, padding=0, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out
