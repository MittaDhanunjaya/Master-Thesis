import os
import numpy as np
import torch
import torch.utils.data as Data
from glob import glob
import pydicom
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from configure import Config

config = Config()
class AbdomenDS(Data.Dataset):
    """description of class"""

    def __init__(self, path, mode, ds_mode, interp_fact=None, readGT=False):
        self.mode = mode
        print('ds mode: '+ str(ds_mode))
        self.listOfItems = glob(path+"/*")
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            import cupy as cp
            self.numpack = cp
        else:
            self.numpack = np
        self.interp_fact = interp_fact
        self.GetVols(ds_mode, readGT=readGT)

    def __len__(self):
        """For returning the length of the dataset"""
        return len(self.listOfItemVols)

    def __getitem__(self, index):
        return self.listOfItemVols[index]

    def GetVols(self, ds_mode, normalize = True, readGT=False):
        #ds_mode: 1 = InPhase, 2 = OutPhase, 3 = T2, 4 = In+OutPhase (Not Implimented yet)
        self.listOfItemVols = []
        for index in range(len(self.listOfItems)):
            if "30" in self.listOfItems[index]:
                print('ignored 30 because of its size is too big for GPU')
                continue
            if ds_mode == 1:
                self.lstFilesDCM = sorted(glob(self.listOfItems[index]+"**/T1DUAL/DICOM_anon/InPhase/*"))
                self.lstFilesGround = sorted(glob(self.listOfItems[index]+"**/T1DUAL/Ground/*"))
            elif ds_mode == 2:
                self.lstFilesDCM = sorted(glob(self.listOfItems[index]+"**/T1DUAL/DICOM_anon/OutPhase/*"))
                self.lstFilesGround = sorted(glob(self.listOfItems[index]+"**/T1DUAL/Ground/*"))
            elif ds_mode == 3:
                self.lstFilesDCM = sorted(glob(self.listOfItems[index]+"**/T2SPIR/DICOM_anon/*"))
                self.lstFilesGround = sorted(glob(self.listOfItems[index]+"**/T2SPIR/Ground/*"))
            else:
                print('Invalid or Not Implimented ds_mode')
            print(self.listOfItems[index])
            dicoms = []
            for ite in self.lstFilesDCM:
                #print(ite)
                RefDs = pydicom.read_file(ite)
                pixels = RefDs.pixel_array
                dicoms.append(pixels)  
            dicoms = np.asarray(dicoms)

            if len(dicoms.shape) == 3:
                if self.interp_fact is not None:
                    tensor = torch.from_numpy(np.expand_dims(np.expand_dims(dicoms, 0), 0)/1.0).float()
                    dicoms = torch.nn.functional.interpolate(tensor, scale_factor=self.interp_fact, mode='nearest').numpy()
                    dicoms = np.squeeze(dicoms, (0,1))
                dicoms = np.expand_dims(dicoms, 0)
                if normalize:
                    dicoms = dicoms/dicoms.max()
            
                if readGT:
                    gtvols = []
                    for ite in self.lstFilesGround:
                        #print(ite)
                        gt = cv2.imread(ite,0)
                        gtvols.append(gt)                
                    gtvols = np.asarray(gtvols)
                    gt_uniq = np.unique(gtvols)
                    gt_processed = np.zeros((len(gt_uniq),gtvols.shape[0],gtvols.shape[1], gtvols.shape[2]))
                    for i in range(len(gt_uniq)):
                        gt_processed[i,gtvols == gt_uniq[i]] = 1
                    if self.interp_fact is not None:
                        tensor = torch.from_numpy(np.expand_dims(gt_processed, 0)/1.0).float()
                        gt_processed = torch.nn.functional.interpolate(tensor, scale_factor=self.interp_fact, mode='nearest').numpy()
                        gt_processed = np.squeeze(gt_processed, 0)
                    self.listOfItemVols.append((torch.from_numpy(dicoms/1.0).float(), torch.from_numpy(gt_processed/1.0).float()))
                else:
                    if(self.mode == "train"):    
                        #dicoms = np.expand_dims(dicoms,0)            
                        weight = self.cal_weight(dicoms)   
                        self.listOfItemVols.append((torch.from_numpy(dicoms/1.0).float(), torch.from_numpy(weight/1.0).float()))
                    else:
                        dicoms = np.expand_dims(dicoms, 0)
                        self.listOfItemVols.append(torch.from_numpy(dicoms/1.0).float())
            else:
                print('ERROR: DICOM shape error for '+ self.listOfItems[index] + ' shape found : ' + str(dicoms.shape))


    def cal_weight(self, raw_data):
        data = self.numpack.asarray(raw_data)
        shape = data.shape
        #print("calculating weights.")
        dissim = self.numpack.zeros((shape[0],shape[1],shape[2],shape[3],(config.radius-1)*2+1,(config.radius-1)*2+1,(config.radius-1)*2+1))
        padded_data = self.numpack.pad(data,((0,0),(config.radius-1,config.radius-1),(config.radius-1,config.radius-1),(config.radius-1,config.radius-1)),'reflect')
        for m in range(2*(config.radius-1)+1):
            for n in range(2*(config.radius-1)+1):
                for i in range(2*(config.radius-1)+1):
                    dissim[:,:,:,:,m,n,i] = data-padded_data[:,m:shape[1]+m,n:shape[2]+n,i:shape[3]+i]
        temp_dissim = self.numpack.exp(-self.numpack.power(dissim,2).sum(0,keepdims = True)/config.sigmaI**2)  
        dist = self.numpack.zeros((2*(config.radius-1)+1,2*(config.radius-1)+1,2*(config.radius-1)+1))
        for m in range(1-config.radius,config.radius):
            for n in range(1-config.radius,config.radius):
                for i in range(1-config.radius,config.radius):
                    if m**2+n**2+i**2<config.radius**2:
                        dist[m+config.radius-1,n+config.radius-1,i+config.radius-1] = self.numpack.exp(-(m**2+n**2+i**2)/config.sigmaX**2)
        print("weight calculated.")
        res = self.numpack.multiply(temp_dissim,dist)
        if self.is_cuda:
            weight = self.numpack.asnumpy(res)
        else:
            weight = np.asarray(res)   
        del data, shape, dissim, padded_data, temp_dissim, dist, res
        if self.is_cuda:
            self.numpack.get_default_memory_pool().free_all_blocks()
            self.numpack.get_default_pinned_memory_pool().free_all_blocks()
        return weight