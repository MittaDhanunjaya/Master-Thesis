#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:16:48 2019

@author: dhanunjayamitta
"""

import torch
import numpy as np
from configure import Config
from model import WNet
from Ncuts import NCutsLoss
#from DataLoader import DataLoader
from AbdomenDS import AbdomenDS
from torch.utils.data import DataLoader
import time
import os
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib


config = Config()
os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_dev_list
if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    #ds_test = AbdomenDS("/Users/dhanunjayamitta/Desktop/single_val","test", config.datasetMode, config.interpFactor)
    #ds_val = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRI_Val","train",(0.5,0.5,0.5))
    ds_train = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRITTemp","test",config.interpFactor)
    # ds_val = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRITTemp","train",(1,0.5,0.5))

    checkname = None #'/raid/scratch/schatter/Dataset/dhanun/checkpoints/checkpoint_9_14_13_52_epoch_15' #Set None if no need to load

    result_upsample = True
    #upunterp_fact = None
    upinterp_fact = (1,1,1)
    #model_downscale = False
    dataloader = DataLoader(ds_test, batch_size=config.BatchSize, shuffle=True)
    model = WNet(is_cuda)
    #model = WNet()
    if is_cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    model.to(device)
    #model.cuda()
    model.eval()
    #model_downscale = False
    mode = 'test'
    optimizer = torch.optim.Adam(model.parameters(),lr = config.init_lr)
    #optimizer
    with open(config.model_tested,'rb') as f:
        para = torch.load(f, "cpu")
        #para = torch.load(f,"cuda:0")
        model.load_state_dict(para['state_dict'])
    for step,[x] in enumerate(dataloader):
        print('Step' + str(step+1))
        #print(x.shape)
        x = x.to(device)
        pred, pad_pred = model(x, mode, config.ModelDownscale)
        print(pred.shape)
        seg = pred.argmax(dim = 1)
        #seg = pred
        
        print(seg.shape)
        #seg = np.reshape(seg, (seg.shape[1],seg.shape[2],seg.shape[3]))
        #x = np.reshape(x, (x.shape[1],x.shape[2],x.shape[3], x.shape[4]))
        #print(x.max())
        print(x.shape)
        if result_upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=upinterp_fact, mode='nearest')
            seg = torch.nn.functional.interpolate(torch.unsqueeze(seg,0).float(), scale_factor=upinterp_fact, mode='nearest')

        seg = np.squeeze(seg,0)
        seg = np.squeeze(seg,0)
        x = np.squeeze(x,0)
        #x = np.squeeze(x,0)
        seg = seg.cpu().detach().numpy()
        # seg = seg.astype(np.float32)
        x = x.cpu().detach().numpy()
        #x = np.transpose(x.astype(np.float32),(0,2,3,1))
        #print(seg.shape)
        #print(seg.max())
        #print(x.shape)
        seg = np.transpose(seg, (0,2,1))
        # import scipy.io as sio
        # sio.savemat('a.mat', {'data': x})
        # sio.savemat('b.mat', {'data': seg})
        """color_map = lambda c: config.color_lib[c]
        cmap = np.vectorize(color_map)
        seg = np.moveaxis(np.array(cmap(seg)),0,-1)"""
        #seg = np.moveaxis(np.array(seg),0,-1).astype(np.uint8)
        for i in range(seg.shape[0]):
            #print(x.shape)
            #print(x[0][:,:,i].shape)
            #print(seg[i,:,:].shape)
            #seg[i,:,:] = np.where(seg[i,:,:] == 5, 6, seg[i,:,:])
            img = nib.Nifti1Image(seg[i,:,:], affine = np.eye(4))
            img1 = nib.Nifti1Image(x[0][i,:,:], affine = np.eye(4))
            #nib.save(img, "/Users/dhanunjayamitta/Desktop/output/seg_"+str(step+1)+"_"+str(i).zfill(2)+".nii")
            nib.save(img, "seg_"+str(step+1)+"_"+str(i).zfill(2)+".nii")
            #nib.save(img1, "/Users/dhanunjayamitta/Desktop/input/input_"+str(step+1)+"_"+str(i).zfill(2)+".nii")
            """plt.title(i)
            plt.subplot(2,2,1)
            plt.imshow(x[0][:,:,0], cmap = 'gray')
            plt.subplot(2,2,2)
            plt.imshow(seg[0,:,:], cmap = 'gray')
            plt.show()"""
            i+=1
            #Image.fromarray(x[i]).save("./input_"+str(step+1)+"_"+str(i)+".jpg")
            #for j in range(seg.shape[-1]):
            #pdb.set_trace()
            #Image.fromarray(seg[i,:,:]).save("./seg_"+str(step+1)+"_"+str(i)+".jpg")"""
        