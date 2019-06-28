# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:14:42 2019

@author: chxy
"""

import h5py
import torch.utils.data as data

def generate_train():
    pass

def generate_test():
    pass

class H5Dataset(data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]
    
def loadtraindata(h5file = './dataset/train.h5'):
    f = h5py.File(h5file,'r')                       
    imgLR = f['data'][:]
    label = f['label'][:]
    #print(imgLR.shape)
    #print(imgLR.dtype)
    trainset = H5Dataset(imgLR, label)
    
    train_loader = data.DataLoader(
            trainset, batch_size=128, shuffle=True,
            num_workers=4, pin_memory=True
        )
    return train_loader

def loadtestdata(h5file = './dataset/test.h5'):
    f = h5py.File(h5file,'r')                       
    imgLR = f['data'][:]
    label = f['label'][:]
    
    #print(label)
    testset = H5Dataset(imgLR, label)
    
    test_loader = data.DataLoader(
            testset, batch_size=2, shuffle=False,
            num_workers=4, pin_memory=True
        )
    return test_loader

loadtraindata()
