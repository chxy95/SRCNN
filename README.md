# SRCNN
The project is reproduction of the paper *Learning a Deep Convolutional Network for Image Super-Resolution》(ECCV 2014)* by Pytorch.
## Dependence
Matlab 2016  
Pytorch 1.0.0  
## Explanation
Some Matlab codes provided by the paper author, url: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html.  
The main reason for using two languages to do the project is because the different implementation of bicubic interpolation, which causes the broader difference of the results when using PSNR standard. 
## Overview
Overview of the network:  
<img src="https://raw.githubusercontent.com/chxy95/SRCNN/master/images/Overview.png" width="700"/>
## Usage
Use ./data_pro/generate_train.m to generate train.h5.  
Use ./data_pro/generate_test.m to generate test.h5.  
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
Image example:  
<img src="https://raw.githubusercontent.com/chxy95/SRCNN/master/images/Comparison.png" width="500"/>
