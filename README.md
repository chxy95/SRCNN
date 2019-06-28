# SRCNN
The project is reproduction of the paper *Learning a Deep Convolutional Network for Image Super-Resolution》(ECCV 2014)* by Pytorch.
## Requirement
Some Matlab codes provided by the paper author, url: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html.  
There are some captions for how to use that official codes to do data-preprocessing and testing in corresponding files.  
## Dependence
Matlab 2016  
Pytorch 1.0.0  
## Usage
Use ./data_pro/generate_train.m to generate train.h5.  
Train by train.py:
```
python train.py
```
Convert the Pytorch model .pkl to Matlab matrix .mat. (weights.pkl -> weights.mat)  
```
python convert.py
```
Use ./test_link/get_result.m to get the PSNR result and reconstruction RGB images.
## Result
Use the ./model/weights.mat can get the result:  
Set5 Average：reconstruction PSNR = 32.44dB VS bicubic PSNR = 30.39dB  
Set14 Average: reconstruction PSNR = 29.05dB VS bicubic PSNR = 27.54dB
