#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:26:11 2019

@author: dhanunjayamitta
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
import numpy as np
import time
from configure import Config

config = Config()

class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()
        self.gpu_list = []
        
    def forward(self, seg, padded_seg, weight):
        #too many values to unpack
        cropped_seg = []
        for m in torch.arange((config.radius-1)*2+1,dtype=torch.long):
            colrow = []
            for n in torch.arange((config.radius-1)*2+1,dtype=torch.long):
                column = []
                for i in torch.arange((config.radius-1)*2+1,dtype=torch.long):
                    column.append(padded_seg[:,:,m:m+seg.size()[2],n:n+seg.size()[3],i:i+seg.size()[4]].clone())
                colrow.append(torch.stack(column,5))
            cropped_seg.append(torch.stack(colrow,6))
        cropped_seg = torch.stack(cropped_seg,6)
        #for m in torch.arange(50,70,dtype=torch.long):

        #    print(m)
        #    for n in torch.arange(50,70,dtype= torch.long):
        #        print(weight[5,0,m,n])
        multi1 = cropped_seg.mul(weight)
        multi2 = multi1.sum(-1).sum(-1).sum(-1).mul(seg)
        sum_weight = weight.sum(-1).sum(-1).sum(-1)
        multi3 = sum_weight.mul(seg)
        #print("=============================================================================")
        #for a in [0,1]:
        #    print(multi2[5,0,a*10+50:a*10+60,50:60])
        #    print(multi2[5,0,a*10+50:a*10+60,60:70])
        assocA = multi2.view(multi2.shape[0],multi2.shape[1],-1).sum(-1)
        assocV = multi3.view(multi3.shape[0],multi3.shape[1],-1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)
        
        return torch.add(-assoc,config.K)