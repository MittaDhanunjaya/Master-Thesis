#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:16:59 2019

@author: dhanunjayamitta
"""

class Config:
    
    def __init__(self):
        #Run Configure
        self.datasetMode = 1 #1 = InPhase, 2 = OutPhase, 3 = T2, 4 = In+OutPhase (Not Implimented yet)
        #self.interpFactor = None
        self.interpFactor = (0.5,0.5,0.5)
        self.runName = '3DWNetv2'
        self.combineLoss = False
        #network configure
        #self.ModelDownscale = True
        self.ModelDownscale = False
        self.InputCh=1
        self.ScaleRatio = 2
        self.ConvSize = 3
        self.pad = (self.ConvSize - 1) // 2 
        self.MaxLv = 5
        self.ChNum = [self.InputCh,64]
        for i in range(self.MaxLv-1):
            self.ChNum.append(self.ChNum[-1]*2)
        #data configure
        self.BatchSize = 1
        self.Shuffle = False
        self.LoadThread = 0
        self.inputsize = [224,224]
        #partition configure
        self.K = 10
        #training configure
        self.init_lr = 0.001
        self.lr_decay = 0.1
        self.lr_decay_iter = 5
        self.max_iter = 100
        self.checkpoint_frequency = 10
        self.cuda_dev = 0 
        self.cuda_dev_list = "4,5"
        self.check_iter = 1000
        self.useSSIMLoss = True
        #self.model_tested = "checkpoints/checkpoint_8_8_2_44_epoch_50"
        #self.model_tested = "/Users/dhanunjayamitta/Downloads/pretrained/checkpoint_3DWNetv2_SepLoss_ds2_ep100_epoch_100"
        self.model_tested = "checkpoints/checkpoint_9_16_17_8_epoch_990"
        #Ncuts Loss configure
        self.radius = 4
        self.sigmaI = 10
        self.sigmaX = 4

    def initiate(self):
        #pre-calculations
        if self.combineLoss:
            self.runName += '_CombLoss'
        else:
            self.runName += '_SepLoss'
        self.runName += '_ds'+str(self.datasetMode)+'_ep'+str(self.max_iter)
        