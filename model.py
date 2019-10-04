#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:34:20 2019

@author: dhanunjayamitta
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from configure import Config 
from Ncuts import NCutsLoss
from AttenUnet import Attention_block

config = Config()
#W-net is a combination of 2 U-nets and the output of first U-net is given as an input to the second U-net.
#Each U-net consists of 9 modules and each module consists of 2 3*3 convolution layers each followed by a ReLU non-linearity and batch normalization.
#First and last modules are normal and remaining modules are connected with separable convolution layers to reduce the computation time

class WNet(torch.nn.Module):
    
    
    def __init__(self, is_cuda):
        super(WNet, self).__init__()
        self.Att = []
        bias = True
        if is_cuda:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        

      #U-Net1:

        self.uconv1 = []
        self.uconv3 = []

      #module1:
        self.conv1 = [
        nn.Conv3d(config.ChNum[0], config.ChNum[1], config.ConvSize, padding = config.pad, bias = bias),
        nn.Conv3d(config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = bias)]
        self.ReLU1 = [nn.PReLU(), nn.PReLU()]
        self.bn1 = [nn.InstanceNorm3d(config.ChNum[1]), nn.InstanceNorm3d(config.ChNum[1])]	
        #self.drp1 = [nn.Dropout(0.1), nn.Dropout(0.1)]


        #module2-5:
        for i in range(2,config.MaxLv+1):
            self.conv1.append(nn.Conv3d(config.ChNum[i-1], config.ChNum[i], 1, bias = bias))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = bias))
            self.ReLU1.append(nn.PReLU())
            self.bn1.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp1.append(nn.Dropout(0.1))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], 1, bias = bias))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = bias))
            self.ReLU1.append(nn.PReLU())
            self.bn1.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp1.append(nn.Dropout(0.1))


      #module6-8:
        for i in range(config.MaxLv-1, 1, -1):
            self.conv1.append(nn.Conv3d(2*config.ChNum[i], config.ChNum[i], 1, bias = bias))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = bias))
            self.ReLU1.append(nn.PReLU())
            self.bn1.append(nn.InstanceNorm3d(config.ChNum[i]))
            #Attention code*********************
            self.Att.append(Attention_block(F_g=config.ChNum[i],F_l=config.ChNum[i],F_int=config.ChNum[i]//2).to(device))
            #self.Att.append(Attention_block(F_g=config.ChNum[i],F_l=config.ChNum[i],F_int=config.ChNum[i]//2))
            #self.drp1.append(nn.Dropout(0.1))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], 1, bias = bias))
            self.conv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = bias))
            self.ReLU1.append(nn.PReLU())
            self.bn1.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp1.append(nn.Dropout(0.1))


      #module9:
        self.conv1.append(nn.Conv3d(2*config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = bias))
        self.ReLU1.append(nn.PReLU())
        self.bn1.append(nn.InstanceNorm3d(config.ChNum[1]))
        #Attention code**************************
        self.Att.append(Attention_block(F_g=config.ChNum[1],F_l=config.ChNum[1],F_int=config.ChNum[1]//2).to(device))
        #self.Att.append(Attention_block(F_g=config.ChNum[1],F_l=config.ChNum[1],F_int=config.ChNum[1]//2))
        #self.drp1.append(nn.Dropout(0.1))
        self.conv1.append(nn.Conv3d(config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = bias))
        self.ReLU1.append(nn.PReLU())
        self.bn1.append(nn.InstanceNorm3d(config.ChNum[1]))
        #self.drp1.append(nn.Dropout(0.1))

      
      #module5-8:
        for i in range(config.MaxLv,1,-1):
            self.uconv1.append(nn.Conv3d(config.ChNum[i], config.ChNum[i-1], (1,1,1), bias=True))
            #self.uconv1.append(nn.ConvTranspose3d(config.ChNum[i], config.ChNum[i-1], (2,2,2),stride = (1,2,2), bias = True))
            #self.uconv3.append(nn.ConvTranspose3d(config.ChNum[i], config.ChNum[i-1], (2,2,2),stride = (2,2,2), bias = True))
        #self.predconv = nn.Conv2d(config.ChNum[1], config.K, 1, bias = bias)
        self.predconv = nn.Conv3d(config.ChNum[1], config.K, 1, bias = bias)
        #self.dropout = nn.Dropout(0.5)
        #self.dropout1 = nn.Dropout(0.8)
        ##########################self.softmax = nn.Softmax3d()
        self.pad = nn.ConstantPad3d(config.radius-1, 0)
        self.ReLU1.append(nn.PReLU())
        self.bn1.append(nn.InstanceNorm3d(config.K))
        #self.drp1.append(nn.Dropout(0.1))

        self.conv1 = torch.nn.ModuleList(self.conv1)
        self.ReLU1 = torch.nn.ModuleList(self.ReLU1)
        self.bn1 = torch.nn.ModuleList(self.bn1)
        #self.drp1.append(nn.Dropout(0.1))
        #self.maxpool1 = torch.nn.ModuleList(self.maxpool1)
        self.uconv1 = torch.nn.ModuleList(self.uconv1)


      #U-Net2
        self.conv2 = []
        self.ReLU2 = []
        self.bn2 = []
        self.uconv2 = []
        self.Att1 = []
        #self.drp2 = []


      #module10:
        self.conv2.append(nn.Conv3d(config.K, config.ChNum[1], config.ConvSize, padding = config.pad, bias = False))
        self.ReLU2.append(nn.PReLU())
        self.bn2.append(nn.InstanceNorm3d(config.ChNum[1]))
        #self.drp2.append(nn.Dropout(0.1))
        self.conv2.append(nn.Conv3d(config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = False))
        self.ReLU2.append(nn.PReLU())
        self.bn2.append(nn.InstanceNorm3d(config.ChNum[1]))
        #self.drp2.append(nn.Dropout(0.1))



      #module11-14:
        for i in range(2,config.MaxLv+1):
            self.conv2.append(nn.Conv3d(config.ChNum[i-1], config.ChNum[i], 1, bias = False))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = False))
            self.ReLU2.append(nn.PReLU())
            self.bn2.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp2.append(nn.Dropout(0.1))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], 1, bias = False))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = False))
            self.ReLU2.append(nn.PReLU())
            self.bn2.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp2.append(nn.Dropout(0.1))


      #module15-17:
        for i in range(config.MaxLv-1,1,-1):
            self.conv2.append(nn.Conv3d(2*config.ChNum[i], config.ChNum[i], 1, bias = False))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = False))
            self.ReLU2.append(nn.PReLU())
            self.bn2.append(nn.InstanceNorm3d(config.ChNum[i]))
            #Attention code************************
            self.Att1.append(Attention_block(F_g=config.ChNum[i],F_l=config.ChNum[i],F_int=config.ChNum[i]//2).to(device))
            #self.Att1.append(Attention_block(F_g=config.ChNum[i],F_l=config.ChNum[i],F_int=config.ChNum[i]//2))
            #self.drp2.append(nn.Dropout(0.1))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], 1, bias = False))
            self.conv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i], config.ConvSize, padding = config.pad, groups = config.ChNum[i], bias = False))
            self.ReLU2.append(nn.PReLU())
            self.bn2.append(nn.InstanceNorm3d(config.ChNum[i]))
            #self.drp2.append(nn.Dropout(0.1))


      #module18:
        self.conv2.append(nn.Conv3d(2*config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = False))
        self.ReLU2.append(nn.PReLU())
        self.bn2.append(nn.InstanceNorm3d(config.ChNum[1]))
        #Attention code*************************
        self.Att1.append(Attention_block(F_g=config.ChNum[1],F_l=config.ChNum[1],F_int=config.ChNum[1]//2).to(device))
        #self.Att1.append(Attention_block(F_g=config.ChNum[1],F_l=config.ChNum[1],F_int=config.ChNum[1]//2))
        #self.drp2.append(nn.Dropout(0.1))
        self.conv2.append(nn.Conv3d(config.ChNum[1], config.ChNum[1], config.ConvSize, padding = config.pad, bias = False))
        self.ReLU2.append(nn.PReLU())
        self.bn2.append(nn.InstanceNorm3d(config.ChNum[1]))
        #self.drp2.append(nn.Dropout(0.1))


      
      #module14-17:
        for i in range(config.MaxLv,1,-1):
            self.uconv2.append(nn.Conv3d(config.ChNum[i], config.ChNum[i-1], (1,1,1), bias=True))
            #self.uconv2.append(nn.ConvTranspose3d(config.ChNum[i], config.ChNum[i-1], (2,2,2),stride = (1,2,2), bias = True))
            #self.uconv4.append(nn.ConvTranspose3d(config.ChNum[i], config.ChNum[i-1], (2,2,2),stride = (2,2,2), bias = True))
        self.reconsconv = nn.Conv3d(config.ChNum[1],1,1,bias = True)
        self.ReLU2.append(nn.PReLU())
        self.bn2.append(nn.InstanceNorm3d(3))
        #self.drp2.append(nn.Dropout(0.1))
        self.conv2 = torch.nn.ModuleList(self.conv2)
        self.ReLU2 = torch.nn.ModuleList(self.ReLU2)
        self.bn2 = torch.nn.ModuleList(self.bn2)
        #self.drp2.append(nn.Dropout(0.1))
        #self.maxpool2 = torch.nn.ModuleList(self.maxpool2)
        self.uconv2 = torch.nn.ModuleList(self.uconv2)

    def forward(self, x, mode='train', downscale=True, groundtruth=None):
        if downscale:
            x = functional.interpolate(x, scale_factor=(1,0.5,0.5))
        
        feature1 = [x]
        del x
        
        #print(x.shape)

      #U-Net1
        tempf = self.conv1[0](feature1[-1])
        tempf = self.ReLU1[0](tempf)
        tempf = self.bn1[0](tempf)
        #print(tempf.shape)
        #tempf = self.drp1[0](tempf)
        tempf = self.conv1[1](tempf)
        tempf = self.ReLU1[1](tempf)
        tempf = self.bn1[1](tempf)
        feature1.append(tempf)
        #feature1.append(self.module[0](x))
        #print(feature1[-1].shape)
        

        upsample_img_size = []
        for i in range(1,config.MaxLv):
            stride_size = ()
            j = ()
            for dim in range(2,5):
                stride_size += (2 if ((feature1[-1].shape)[dim]%2 ==0) else 1, )
                j += (feature1[-1].shape[dim], )
            upsample_img_size.append(j)
            #######################tempf = self.maxpool1[i-1](feature1[-1])
            tempf = functional.max_pool3d(feature1[-1], (2,2,2), stride = stride_size)
            tempf = self.conv1[4*i-2](tempf)
            tempf = self.conv1[4*i-1](tempf)
            tempf = self.ReLU1[2*i](tempf)
            tempf = self.bn1[2*i](tempf)
            #tempf = self.drp1[2*i](tempf)
            tempf = self.conv1[4*i](tempf)
            tempf = self.conv1[4*i+1](tempf)
            tempf = self.ReLU1[2*i+1](tempf)
            tempf = self.bn1[2*i+1](tempf)
            feature1.append(tempf)
            #print(feature1[-1].shape)
        for i in range(config.MaxLv,2*config.MaxLv-2):
            j = upsample_img_size[-((i+1)%config.MaxLv)]
            tempf = functional.interpolate(feature1[-1], size=j, mode='nearest')
            tempf = self.uconv1[i-config.MaxLv](tempf)
            #print(tempf.shape)
            #print(feature1[2*config.MaxLv-i-1].shape)
            #Attention code****************
            tempf = self.Att[i-config.MaxLv](g=tempf,x=feature1[2*config.MaxLv-i-1])
 
            tempf = torch.cat((feature1[2*config.MaxLv-i-1], tempf), dim=1)
            tempf = self.conv1[4*i-2](tempf)
            tempf = self.conv1[4*i-1](tempf)
            tempf = self.ReLU1[2*i](tempf)
            tempf = self.bn1[2*i](tempf)
            
            #tempf = self.drp1[2*i](tempf)
            tempf = self.conv1[4*i](tempf)
            tempf = self.conv1[4*i+1](tempf)
            tempf = self.ReLU1[2*i+1](tempf)
            tempf = self.bn1[2*i+1](tempf)

            feature1.append(tempf)

        #print(feature1[-1].shape)
        j = upsample_img_size[0]
        tempf = functional.interpolate(feature1[-1], size=j, mode='nearest')
        tempf = self.uconv1[config.MaxLv-2](tempf)
        #tempf = self.uconv1[config.MaxLv-2](feature1[-1])
        #Attention code*******************
        tempf = self.Att[config.MaxLv-2](g=tempf,x=feature1[1])

        tempf = torch.cat((feature1[1], tempf), dim=1)

        del feature1
        tempf = self.conv1[-2](tempf)
        tempf = self.ReLU1[4*config.MaxLv-4](tempf)
        tempf = self.bn1[4*config.MaxLv-4](tempf)
        #print(tempf.shape)
        #tempf = self.drp1[4*config.MaxLv-4](tempf)
        tempf = self.conv1[-1](tempf)
        tempf = self.ReLU1[4*config.MaxLv-3](tempf)
        tempf = self.bn1[4*config.MaxLv-3](tempf)
        
        #print(feature1[-1].shape)
        tempf = self.predconv(tempf)
        tempf = self.ReLU1[-1](tempf)
        tempf = self.bn1[-1](tempf)
        
        #feature1[-1] = self.dropout(feature1[-1])
        ##########feature1[-1] = self.softmax(tempf)
        #feature1[-1] = functional.softmax(tempf, dim = 1)
        #print(feature1[-1].shape)
        ###########feature2 = [self.softmax(feature1[-1])]
        feature2 = [functional.softmax(tempf, dim = 1)]
        #print(feature2[-1].shape)
        #feature2.append(self.pad(feature2[0]))
        #feature2.append(self.loss(feature2[0],feature2[1], w, sw))
        #return feature2[0], self.pad(feature2[0])
        if(mode == 'test'):
            if downscale:
                feature2[0] = functional.interpolate(feature2[0], scale_factor=(1,2,2))
            return feature2[0], self.pad(feature2[0])
        else:

            if groundtruth is not None:
              segmentations = feature2[0]
              del feature2
              feature2 = [groundtruth]

     #U-Net2
            #print(feature2[-1].shape)
            tempf = self.conv2[0](feature2[-1])
            tempf = self.ReLU2[0](tempf)
            tempf = self.bn2[0](tempf)
            tempf = self.conv2[1](tempf)
            tempf = self.ReLU2[1](tempf)
            tempf = self.bn2[1](tempf)
            #print(tempf.shape)
            feature2.append(tempf)
            #print(feature2[-1].shape)

            upsample_img_size = []
            for i in range(1,config.MaxLv):
                stride_size = ()
                j = ()
                for dim in range(2,5):
                    stride_size += (2 if ((feature2[-1].shape)[dim]%2 ==0) else 1, )
                    j += (feature2[-1].shape[dim], )
                upsample_img_size.append(j)
                #######################tempf = self.maxpool1[i-1](feature1[-1])
                tempf = functional.max_pool3d(feature2[-1], (2,2,2), stride = stride_size)
                #tempf = self.maxpool2[i-1](feature2[-1])
                tempf = self.conv2[4*i-2](tempf)
                tempf = self.conv2[4*i-1](tempf)
                tempf = self.ReLU2[2*i](tempf)
                tempf = self.bn2[2*i](tempf)
                #tempf = self.drp2[2*i](tempf)
                tempf = self.conv2[4*i](tempf)
                tempf = self.conv2[4*i+1](tempf)
                tempf = self.ReLU2[2*i+1](tempf)
                tempf = self.bn2[2*i+1](tempf)
    
                feature2.append(tempf)
    
            #print(feature2[-1].shape)
            for i in range(config.MaxLv,2*config.MaxLv-2):
                j = upsample_img_size[-((i+1)%config.MaxLv)]
                tempf = functional.interpolate(feature2[-1], size=j, mode='nearest')
                tempf = self.uconv2[i-config.MaxLv](tempf)
                #tempf = self.uconv2[i-config.MaxLv](feature2[-1])
                #Attention code**********************
                tempf = self.Att1[i-config.MaxLv](g=tempf,x=feature2[2*config.MaxLv-i-1])
                tempf = torch.cat((feature2[2*config.MaxLv-i-1], tempf), dim=1)
                tempf = self.conv2[4*i-2](tempf)
                tempf = self.conv2[4*i-1](tempf)
                tempf = self.ReLU2[2*i](tempf)
                tempf = self.bn2[2*i](tempf)
                #tempf = self.drp2[2*i](tempf)
                tempf = self.conv2[4*i](tempf)
                tempf = self.conv2[4*i+1](tempf)
                tempf = self.ReLU2[2*i+1](tempf)
                tempf = self.bn2[2*i+1](tempf) 
                #tempf = self.drp2[2*i+1](tempf)
                feature2.append(tempf)
            
            j = upsample_img_size[0]
            tempf = functional.interpolate(feature2[-1], size=j, mode='nearest')
            tempf = self.uconv2[config.MaxLv-2](tempf)
            #tempf = self.uconv2[config.MaxLv-2](feature2[-1])
            #Attention code*****************
            tempf = self.Att1[config.MaxLv-2](g=tempf,x=feature2[1])
            tempf = torch.cat((feature2[1], tempf), dim=1)
            if groundtruth is None:
              segmentations = feature2[0]
            del feature2
            tempf = self.conv2[-2](tempf)
            tempf = self.ReLU2[4*config.MaxLv-4](tempf)
            tempf = self.bn2[4*config.MaxLv-4](tempf)
            #tempf = self.drp2[4*config.MaxLv-4](tempf)
            tempf = self.conv2[-1](tempf)
            tempf = self.ReLU2[4*config.MaxLv-3](tempf)
            tempf = self.bn2[4*config.MaxLv-3](tempf)  
            #tempf = self.drp2[4*config.MaxLv-3](tempf)  
            
            
            tempf = self.reconsconv(tempf)
            tempf = self.ReLU2[-1](tempf)
            tempf = self.bn2[-1](tempf)
            #tempf = self.dropout(tempf)
            tempf = torch.sigmoid(tempf)
            
            #return feature2[0], self.pad(feature2[0])
            
            if downscale:
                tempf = functional.interpolate(tempf, scale_factor=(1,2,2))
                segmentations = functional.interpolate(segmentations, scale_factor=(1,2,2))
            
            return tempf,segmentations, self.pad(segmentations)
