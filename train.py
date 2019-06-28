# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:50:16 2019

@author: chxy
"""

import torch
import torch.nn as nn
from model import SRCNN_net
import torch.optim as optim
from data_loader import loadtraindata, loadtestdata
from torch.autograd import Variable

from math import ceil
def train(num_channels=1, learning_rate=1e-3, epochs=1000):
    trainloader = loadtraindata()
    testloader = loadtestdata()
    net = SRCNN_net(num_channels=num_channels).cuda()

    conv3_params = list(map(id, net.conv3.parameters()))
    base_params = filter(lambda p: id(p) not in conv3_params, net.parameters())
    optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': net.conv3.parameters(), 'lr': learning_rate * 0.1}
            ], lr=learning_rate)

    #optimizer = optim.Adam(net.parameters(), lr=learning_rate) 
    loss_fn = nn.MSELoss().cuda()
    train_loss_list = []
    test_loss_list = []
	
    for epoch in range(epochs):
	
        train_loss = 0.0
        test_loss = 0.0
		
        for i, data in enumerate(trainloader):
            imgLR, label = data
            imgLR, label = imgLR.cuda(), label.cuda()
            imgLR, label = Variable(imgLR), Variable(label)
            imgHR = net(imgLR)
            loss = loss_fn(imgHR, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / (i+1)
		
        for i, data in enumerate(testloader):
            imgLR, label = data
            imgLR, label = imgLR.cuda(), label.cuda()
            imgLR, label = Variable(imgLR), Variable(label)
            imgHR = net(imgLR)
            loss = loss_fn(imgHR, label)
            test_loss += loss.item()
        test_loss = test_loss / (i+1)
		
        print("epoch : {}, train_loss : {:.6f}, test_loss : {:.6f}".format(epoch+1, train_loss, test_loss))
        
        torch.save(net.state_dict(), './ckpt/epoch_{}_train_loss_{}_test_loss_{}_net_params.pkl'.format(epoch+1, train_loss, test_loss))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    with open('loss.txt', 'w') as f:
        for i in range(epochs):
            f.write("{} {}\n".format(train_loss_list[i], test_loss_list[i]))

train()
