#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:14:04 2019

@author: dhanunjayamitta
"""
import torch
import numpy as np
from tensorboardX import SummaryWriter
from configure import Config
from model import WNet
from torch.utils.data import DataLoader
from Ncuts import NCutsLoss
#from AttenUnet import Attention_block
import time
import os
import argparse
import sys

from saveimg_helper import makegridimg3D

from AbdomenDS import AbdomenDS
if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='3D Attention WNet')
    parser.add_argument("-d","--dataset", required=False, help="Dataset Type: 1 = InPhase, 2 = OutPhase, 3 = T2, 4 = In+OutPhase")
    parser.add_argument("-l","--losscombine", required=False, help="Loss Combine: 1 = True, 0 = False")
    parser.add_argument("-c","--cuda", required=False, help="List of CUDA Devices. It needs to device atleast. Device cardinals coma separated without space")
    args = vars(parser.parse_args())
    if args['dataset'] is not None:
        config.datasetMode = int(args['dataset'])
    if args['losscombine'] is not None:
        config.combineLoss = True if args['losscombine']=="1" else False
    if args['cuda'] is not None:
        config.cuda_dev_list = args['cuda']

    config.initiate()
    os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_dev_list
    writer = SummaryWriter(os.path.join('runs',config.runName))


    is_cuda = torch.cuda.is_available()
    #is_cuda = False
    #ds_train = AbdomenDS("/Users/dhanunjayamitta/Desktop/single_train","train", config.datasetMode, config.interpFactor)
    ds_train = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRI_Train","train", config.datasetMode, config.interpFactor)
    # ds_train = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRITTemp","train",config.interpFactor)
    #ds_val = AbdomenDS("/Users/dhanunjayamitta/Desktop/single_train","train", config.datasetMode, config.interpFactor)
    ds_val = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRI_Val","train",config.datasetMode, config.interpFactor)
    # ds_val = AbdomenDS("/raid/scratch/schatter/Dataset/dhanun/MRI/MRITTemp","train",config.datasetMode, config.interpFactor)

    checkname = None #"/Users/dhanunjayamitta/Desktop/Archive 8 fresh/checkpoints/checkpoint_9_16_17_8_epoch_990" #Set None if no need to load

    dataloader = DataLoader(ds_train, batch_size=config.BatchSize, shuffle=True)
    dataloader1 = DataLoader(ds_val, batch_size=config.BatchSize, shuffle=True)
    #eval_set = DataLoader("MRI/new_test","train")
    
    #eval_loader = eval_set.torch_loader()
    model = WNet(is_cuda)
    #model = torch.nn.DataParallel(WNet())
    if is_cuda:
        device1 = torch.device("cuda:1")
        device2 = torch.device("cuda:0")
    else:
        device1 = torch.device("cpu")
        device2 = torch.device("cpu")
    model.to(device1)
    #model_eval = torch.nn.DataParallel(WNet())
    #model.cuda()
    #model_eval.cuda()
    #model_eval.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr = config.init_lr)
    #reconstr = torch.nn.MSELoss().cuda(config.cuda_dev)
    if config.useSSIMLoss:
        import pytorch_ssim
        reconstr = pytorch_ssim.SSIM()
    else:
        reconstr = torch.nn.MSELoss()
    Ncuts = NCutsLoss()
    mode = 'train'
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)

    if checkname is not None:
        with open(checkname,'rb') as f:
            checkpoint_data = torch.load(f, "cpu")
            model.load_state_dict(checkpoint_data['state_dict'])
            if "pretrain" not in checkname:
                print('a non-pretrain checkpoint is now being loaded')
                start_epoch = int(checkpoint_data['epoch'])
                optimizer.load_state_dict(checkpoint_data['optimizer'])
                scheduler.load_state_dict(checkpoint_data['scheduler'])
            else:
                print('a pretrain checkpoint is now being loaded')
                start_epoch = 0
        print('checkpoint loaded')
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, config.max_iter):
        print("Epoch: "+str(epoch+1))
        Ave_Ncuts = 0.0
        Ave_Rec = 0.0
        Ave_Ncuts1 = 0.0
        Ave_Rec1 = 0.0
        t_load = 0.0
        t_forward = 0.0
        t_loss = 0.0
        t_backward = 0.0
        t_adjust = 0.0
        t_reset = 0.0
        t_inloss = 0.0
        
        model.train()
        print('training')
        for step,[x,w] in enumerate(dataloader):
            print(step)
            print(x.shape)
            print(w.shape)
            x = x.to(device1)
            w = w.to(device2)
            
            #try:
            optimizer.zero_grad()
            rec_image, pred, pad_pred = model(x, mode, config.ModelDownscale)
            ncuts_loss = Ncuts(pred.to(device2),pad_pred.to(device2),w)
            ncuts_loss = ncuts_loss.sum()/config.BatchSize 
            Ave_Ncuts = (Ave_Ncuts * step + ncuts_loss.item())/(step+1) 
            if not config.combineLoss:
                ncuts_loss.backward(retain_graph=True)
                optimizer.step()            
                optimizer.zero_grad()

            if config.useSSIMLoss:
                rec_loss = -reconstr(rec_image[:,:,0,...], x[:,:,0,...]) #1st slice
                for d in range(1, x.shape[2]):
                    rec_loss += -reconstr(rec_image[:,:,d,...], x[:,:,d,...])
                rec_loss /= x.shape[2]
            else:
                rec_loss = reconstr(rec_image, x)
            if not config.combineLoss:
                rec_loss.backward()
            else:
                Loss = ncuts_loss + rec_loss
                Loss.backward()
            optimizer.step()

            if config.useSSIMLoss:
                rec_loss = -rec_loss.data.item()
            else:
                rec_loss = rec_loss.item()
            Ave_Rec = (Ave_Rec * step + rec_loss)/(step+1)

            writer.add_scalar('Ncuts_loss_train', ncuts_loss.item(), (epoch*config.max_iter)+step)
            writer.add_scalar('Recon_loss_train', rec_loss, (epoch*config.max_iter)+step)           

            del x, w, rec_image, pred, pad_pred, ncuts_loss, rec_loss
            # except Exception as ex:
            #     print(ex)
            #     del x, w
            #     try: 
            #         del rec_image, pred, pad_pred, ncuts_loss, rec_loss
            #     except:
            #         print('')

            
        model.eval()
        print('validating')
        for step,[x,w] in enumerate(dataloader1):
            print(step)
            x = x.to(device1)
            w = w.to(device2)
            print(x.shape)

            # try:
            
            rec_image1, pred1, pad_pred1 = model(x, mode, config.ModelDownscale)
            ncuts_loss1 = Ncuts(pred1.to(device2),pad_pred1.to(device2),w)
            ncuts_loss1 = ncuts_loss1.sum()/config.BatchSize
            Ave_Ncuts1 = (Ave_Ncuts1 * step + ncuts_loss1.item())/(step+1)
            

            if config.useSSIMLoss:
                rec_loss1 = -reconstr(rec_image1[:,:,0,...], x[:,:,0,...]) #1st slice
                for d in range(1, x.shape[2]):
                    rec_loss1 += -reconstr(rec_image1[:,:,d,...], x[:,:,d,...])
                rec_loss1 /= x.shape[2]
                rec_loss1 = rec_loss1.data
            else:
                rec_loss1 = reconstr(rec_image1, x)
            Ave_Rec1 = (Ave_Rec1 * step + rec_loss1.item())/(step+1)

            writer.add_scalar('Ncuts_loss_val', ncuts_loss1.item(), (epoch*config.max_iter)+step)
            writer.add_scalar('Recon_loss_val', rec_loss1.item(), (epoch*config.max_iter)+step)     

            writer.add_image('Reconstruction_Img', makegridimg3D(rec_image1, False), (epoch*config.max_iter)+step)     
            #writer.add_image('Pred_Seg', makegridimg3D(pred1, True), (epoch*config.max_iter)+step)          

            del x, w, rec_image1, pred1, pad_pred1, ncuts_loss1, rec_loss1
            # except Exception as ex:
            #     print(ex)
            #     del x, w
            #     try: 
            #         del rec_image1, pred1, pad_pred1, ncuts_loss1, rec_loss1
            #     except:
            #         print('')
            
        print("Ncuts loss_train: "+str(Ave_Ncuts))
        print("Ncuts loss_val: "+str(Ave_Ncuts1))
        #N_cuts_train.append(Ave_Ncuts)
        print("rec_ loss_train: "+str(Ave_Rec))
        print("rec_ loss_val: "+str(Ave_Rec1))
        orig_stdout = sys.stdout
        f = open('out'+config.runName+'.txt', 'a')
        sys.stdout = f
        print("Ncuts loss_train: "+str(Ave_Ncuts))
        print("Ncuts_loss_val: "+str(Ave_Ncuts1))
        print("rec_ loss_train: "+str(Ave_Rec))
        print("rec_ loss_val: "+str(Ave_Rec1))
        sys.stdout = orig_stdout
        f.close()
        writer.add_scalar('AvgNcuts_loss_train', Ave_Ncuts, epoch)
        writer.add_scalar('AvgNcuts_loss_val', Ave_Ncuts1, epoch)
        writer.add_scalar('AvgReconstruction_loss_train', Ave_Rec, epoch)
        writer.add_scalar('AvgReconstruction_loss_val', Ave_Rec1, epoch)
        #Rec_Loss_train.append(Ave_Rec)
        """print("Ncuts loss_eval: "+str(Ave_Ncuts1))
        N_cuts_val.append(Ave_Ncuts1)
        print("rec_ loss_eval: "+str(Ave_Rec1))
        Rec_Loss_val.append(Ave_Rec1)"""
        if (epoch+1)%config.checkpoint_frequency == 0:
            checkname = './checkpoints'
            os.makedirs(checkname, exist_ok=True)
            checkname+='/checkpoint'+'_'+config.runName
            checkname += '_epoch_'+str(epoch+1)
            with open(checkname,'wb') as f:
                torch.save({
                'epoch': epoch +1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'Segloss': Ave_Ncuts,
                'reconloss': Ave_Rec
                },f)
            print(checkname+' saved')
    
        scheduler.step()