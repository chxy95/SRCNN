# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:03:58 2019

@author: chxy
"""

import torch
from model import SRCNN_net
import scipy.io
import pickle
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

net = SRCNN_net(num_channels=1)
net.load_state_dict(torch.load('./model/weights.pkl', map_location='cpu'))

for name in net.state_dict():
   print(name)
   print(net.state_dict()[name].data.cpu().numpy().dtype)


weight = dict()
weight['weights_conv1'] = net.state_dict()['conv1.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(81, 64)
weight['weights_conv2'] = net.state_dict()['conv2.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(64, 1, 32)
weight['weights_conv3'] = net.state_dict()['conv3.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(32, 25)

weight['biases_conv1'] = net.state_dict()['conv1.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_conv2'] = net.state_dict()['conv2.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_conv3'] = net.state_dict()['conv3.bias'].data.cpu().numpy().astype(np.float64)

   
scipy.io.savemat('./model/weights.mat', mdict=weight)

