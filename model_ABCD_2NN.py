from __future__ import print_function, division

import torch
import torch.nn as nn
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json
import h5py
import csv

from torch.nn import init

from sklearn.metrics import roc_curve
from scipy.stats import entropy # This is the KL divergence

from evaluation import JSD, JSDvsR,ABCD_metrics

from dcor import u_distance_correlation_sqr
from disco import distance_corr,distance_corr_unbiased,dcovsq_unbiased_slow,dcorrsq_unbiased_slow


    
state={}

def calculate_loss(net1,net2,data,target,weight,alpha=0.,decorr_mode='none'):
                   
                   
    output1 = net1(data)
    output1 = F.softmax(output1)[:,1]
    output2 = net2(data)
    output2 = F.softmax(output2)[:,1]

    # backward
#        print(data.shape,target.shape,output.shape)

#        loss = F.cross_entropy(output, target)

    loss1=0.
    loss2=0.
    loss3=0.
    if(decorr_mode=='dist_unbiased' or decorr_mode=='dist' or decorr_mode=='dist_unbiased2'):
        loss1 = F.binary_cross_entropy(output1,target.float(),weight)
        loss2 = F.binary_cross_entropy(output2,target.float(),weight)
#            y1qcd=output1[(target==0)]
#            y2qcd=output2[(target==0)]                      
        y1qcd0=output1[(target==0)]
        y2qcd0=output2[(target==0)]
#            y1cut=np.percentile(y1qcd0.detach().cpu().numpy(),100-25)
#            y2cut=np.percentile(y2qcd0.detach().cpu().numpy(),100-25)
#            y1qcd2=y1qcd0[(y1qcd0>y1cut) & (y2qcd0>y2cut)]
#            y2qcd2=y2qcd0[(y1qcd0>y1cut) & (y2qcd0>y2cut)]

        if(len(y1qcd0)>2):
#                wqcd=weight[target==0]
#                normedweight=wqcd/(wqcd.sum())*len(wqcd) # weights should sum to n=size of sample
#                normedweight=wqcd/(wqcd.sum())
#                normedweight=torch.ones(len(yqcd2)).float().cuda()/len(yqcd2)
            normedweight=torch.autograd.Variable(torch.ones(len(y1qcd0)).cuda())
            if(decorr_mode=='dist'):
                dCorr=distance_corr(y1qcd0,y2qcd0,normedweight,power=1)
                loss3+=alpha*(dCorr)
            elif(decorr_mode=='dist_unbiased' or decorr_mode=='dist_unbiased2'):
                dCorr=distance_corr_unbiased(y1qcd0,y2qcd0,normedweight,power=1)
                loss3+=alpha*(dCorr)

            if(decorr_mode=='dist_unbiased2'):
                y1cut=np.percentile(y1qcd0.detach().cpu().numpy(),100-50)
                y2cut=np.percentile(y2qcd0.detach().cpu().numpy(),100-50)
                y1qcd1=y1qcd0[(y1qcd0>y1cut) & (y2qcd0>y2cut)]
                y2qcd1=y2qcd0[(y1qcd0>y1cut) & (y2qcd0>y2cut)]
                if(len(y1qcd1)>2):
                    normedweight=torch.autograd.Variable(torch.ones(len(y2qcd1)).cuda())
                    dCorr2=distance_corr_unbiased(y1qcd1,y2qcd1,normedweight,power=1)
                    loss3+=alpha*(dCorr2)
#                    if(len(y1qcd2)>2):
#                        normedweight=torch.autograd.Variable(torch.ones(len(y2qcd2)).cuda())
#                        dCorr3=distance_corr_unbiased(y1qcd2,y2qcd2,normedweight,power=1)
#                        loss3+=alpha*(dCorr3)
    else:
        print('Decorrelation mode not supported')
                 
    return loss1,loss2,loss3,output1,output2


# train function (forward, backward, update)
def train(net1,net2,optimizer1,optimizer2,dataloader,decorr_mode='none',alpha=0.1):
    print('Training')
    t0 = time.time()
    Ntrain=len(dataloader.dataset)
    print(Ntrain)
    net1.train()
    net2.train()
    loss_avg = 0.0
    loss2_avg = 0.0
    loss2=0.
    Ndata=0
    for batch_idx, (data, target,weight,binnum,mass) in enumerate(dataloader):
        Ndata+=len(data)
        if(Ndata>Ntrain):
            break
#        data = (expand_array(data)).astype('float32')
#        target = int(target)

        data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        weight=torch.autograd.Variable(weight.cuda())
        binnum=torch.autograd.Variable(binnum.cuda())
        mass=torch.autograd.Variable(mass.cuda())
#        data, target = data.to(device), target.to(device)

        print('minibatch',batch_idx+1,'/',int(len(dataloader)), end='\r')
        # forward

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss1,loss2,loss3,_,_=calculate_loss(net1,net2,data,target,weight,alpha=alpha,decorr_mode=decorr_mode)
        
        loss=loss1+loss2+loss3
        
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + float(loss) * 0.8
        loss2_avg = loss2_avg * 0.2 + float(loss2) * 0.8

#    print('std',std)
    print('\n')
    
    t1 = time.time()
    print('Training time: {} seconds'.format(t1 - t0))

    return loss_avg,loss2_avg

    
# val function (forward only)
def val(net1,net2,dataloader,decorr_mode='none',alpha=0.1):
    print('Validating')
    t1 = time.time()
    Nval=len(dataloader.dataset)
    print(Nval)
    net1.eval()
    net2.eval()
    loss1_avg = 0.0
    loss2_avg = 0.0
    loss3_avg = 0.0
    correct = 0
    Nvalcount=0
    Nbatchcount=0
    output1all=torch.empty(0)
    output2all=torch.empty(0)
    labels=torch.empty(0)
    weights=torch.empty(0)
    binnums=torch.empty(0)
    masses=torch.empty(0)

    for batch_idx, (data, target,weight,binnum,mass) in enumerate(dataloader):
        if(len(data)*batch_idx>Nval):
            break
#        data = (expand_array(data)).astype('float32')
#        target = int(target)
        data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        weight=torch.autograd.Variable(weight.cuda())
        mass=torch.autograd.Variable(mass.cuda())

        print('minibatch',batch_idx+1,'/',int(len(dataloader)), end='\r')

        labels=torch.cat((labels,target.float().data.cpu()))
        weights=torch.cat((weights,weight.float().data.cpu()))
        binnums=torch.cat((binnums,binnum.float().data.cpu()))
        masses=torch.cat((masses,mass.float().data.cpu()))

        with torch.no_grad():
            # forward
            loss1,loss2,loss3,output1,output2=calculate_loss(net1,net2,data,target,weight,alpha=alpha,decorr_mode=decorr_mode)
            # val loss average
            loss1_avg += float(loss1)
            loss2_avg += float(loss2)
            loss3_avg += float(loss3)
            Nvalcount += len(data)
            Nbatchcount += 1
            output1all=torch.cat((output1all,output1.float().data.cpu()))
            output2all=torch.cat((output2all,output2.float().data.cpu()))
    print('\n')
    
    loss1_avg = loss1_avg / Nbatchcount
    loss2_avg = loss2_avg / Nbatchcount
    loss3_avg = loss3_avg / Nbatchcount
#    state['val_accuracy'] = correct / Nvalcount

    t2 = time.time()
    print('Validation time: {} seconds'.format(t2 - t1))
    
    return output1all,output2all,labels,weights,binnums,masses,loss1_avg,loss2_avg,loss3_avg


# val function (forward only)
def val_slow(net1,net2,dataloader,decorr_mode='none',alpha=0.1):
    print('Validating, slow version')
    t1 = time.time()
    Nval=len(dataloader.dataset)
    print(Nval)
    net1.eval()
    net2.eval()
    loss1_avg = 0.0
    loss2_avg = 0.0
    loss3_avg = 0.0
    correct = 0
    Nvalcount=0
    Nbatchcount=0
    output1all=torch.empty(0)
    output2all=torch.empty(0)
    labels=torch.empty(0)
    weights=torch.empty(0)
    binnums=torch.empty(0)
    masses=torch.empty(0)
    loss=0.
    for batch_idx, (data, target,weight,binnum,mass) in enumerate(dataloader):
        if(len(data)*batch_idx>Nval):
            break
#        data = (expand_array(data)).astype('float32')
#        target = int(target)
        data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        weight=torch.autograd.Variable(weight.cuda())
        mass=torch.autograd.Variable(mass.cuda())

        print('minibatch',batch_idx+1,'/',int(len(dataloader)), end='\r')

        labels=torch.cat((labels,target.float().data.cpu()))
        weights=torch.cat((weights,weight.float().data.cpu()))
        binnums=torch.cat((binnums,binnum.float().data.cpu()))
        masses=torch.cat((masses,mass.float().data.cpu()))

        with torch.no_grad():
            # forward
            output1 = net1(data)
            output1 = F.softmax(output1)[:,1]
            output2 = net2(data)
            output2 = F.softmax(output2)[:,1]
        output1all=torch.cat((output1all,output1.float().data.cpu()))
        output2all=torch.cat((output2all,output2.float().data.cpu()))
        
    if(decorr_mode=='dist_unbiased'):
        loss1 = F.binary_cross_entropy(output1all,labels,weights) # this is averaged by default
        loss2 = F.binary_cross_entropy(output2all,labels,weights) # this is averaged by default
        y1qcd=output1all[(labels==0)].numpy().astype('float64')
        y2qcd=output2all[(labels==0)].numpy().astype('float64')
        loss3 = alpha*u_distance_correlation_sqr(y1qcd,y2qcd)  # this function is 100x faster than my version!      
    else:
        print("Requested decorrelation mode not supported!")
        
            
    print('\n')
  
#    state['val_accuracy'] = correct / Nvalcount

    t2 = time.time()
    print('Validation time: {} seconds'.format(t2 - t1))
    
    return output1all,output2all,labels,weights,binnums,masses,float(loss1),float(loss2),float(loss3)



#def JSD(hist1,hist2):
#    output=0.5*(entropy(hist1,0.5*(hist1+hist2))+entropy(hist2,0.5*(hist1+hist2)))
#    return output



def train_model(Nepochs,lr,net1,net2,optimizer1,optimizer2,train_loader,val_loader,decorr_mode='none',alpha=0.1,lrschedule=[],logfile='log',label=''):
    
    val_acc1_list=[]
    val_acc2_list=[]
    best_loss = 9999999999999999.
    patience=0

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    for epoch in range(Nepochs):
        state={}
        if epoch in lrschedule:
            lr *= 0.5
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr
        print('epoch',epoch,'lr',param_group['lr'])

        state['epoch'] = epoch

        train_loss,train_loss2=train(net1,net2,optimizer1,optimizer2,train_loader,decorr_mode,alpha=alpha)

        output1,output2,labels,_,_,masses,val_loss1,val_loss2,val_loss3=val(net1,net2,val_loader,decorr_mode=decorr_mode,alpha=alpha)

        t2=time.time()
        
        labels=labels.numpy()
        output1=output1.numpy()
        output2=output2.numpy()
        masses=masses.numpy()

        fpr1,tpr1,_=roc_curve(labels,output1)
        val_acc1=np.max(0.5*(tpr1+1-fpr1))
        val_acc1_list.append(val_acc1)

        fpr2,tpr2,_=roc_curve(labels,output2)
        val_acc2=np.max(0.5*(tpr2+1-fpr2))
        val_acc2_list.append(val_acc2)

        Wout=output1[labels==1.]
        qcdout1=output1[labels==0.]
        qcdout2=output2[labels==0.]

        JSDR50=JSDvsR(Wout,qcdout1,qcdout2,sigeff=50,minmass=0,maxmass=1)
        JSDR30=JSDvsR(Wout,qcdout1,qcdout2,sigeff=30,minmass=0,maxmass=1)
        JSDR10=JSDvsR(Wout,qcdout1,qcdout2,sigeff=10,minmass=0,maxmass=1)
              
        R50=JSDR50[0]
        JSD50=JSDR50[1]

        R30=JSDR30[0]
        JSD30=JSDR30[1]

        R10=JSDR10[0]
        JSD10=JSDR10[1]

        eBlist,eRlist,eRsiglist=ABCD_metrics(output1,output2,labels)
        np.save('eBlist_'+label+'_'+str(alpha)+"_"+str(epoch)+'_DD.npy',eBlist)
        np.save('eRlist_'+label+'_'+str(alpha)+"_"+str(epoch)+'_DD.npy',eRlist)
        np.save('eRsiglist_'+label+'_'+str(alpha)+"_"+str(epoch)+'_DD.npy',eRsiglist)

        t3 = time.time()
        
        print('Calculating metrics time: {} seconds'.format(t3 - t2))
        

        print("Train loss, val loss, val accuracy, val R50, val R30: ",[train_loss,train_loss2],[val_loss1,val_loss2,val_loss3],val_acc1,val_acc2,R50,R30)
        print("JSD50, JSD30, JSD10: ", JSD50, JSD30, JSD10)
        state['JSD50',alpha,epoch]=JSD50
        state['JSD30',alpha,epoch]=JSD30
        state['JSD10',alpha,epoch]=JSD10
        state['val_accuracy1',alpha,epoch]=val_acc1
        state['val_accuracy2',alpha,epoch]=val_acc2
        state['val_loss1',alpha,epoch]=val_loss1
        state['val_loss2',alpha,epoch]=val_loss2
        state['val_loss3',alpha,epoch]=val_loss3
        state['R50',alpha,epoch]=R50
        state['R30',alpha,epoch]=R30
        state['R10',alpha,epoch]=R10
        
        torch.save(net1.state_dict(), "ABCD_model1_"+label+'_'+str(alpha)+"_"+str(epoch)+".dict")
        torch.save(net2.state_dict(), "ABCD_model2_"+label+'_'+str(alpha)+"_"+str(epoch)+".dict")

        
#        if(epoch>10 and np.mean(val_acc1_list[-10:])<0.52):
#            break
        val_loss=val_loss1+val_loss2+val_loss3
        if val_loss < best_loss and val_loss3>=0:
            best_loss=val_loss
            best_epoch=epoch
            torch.save(net1.state_dict(),  "ABCD_model1_"+label+'_'+str(alpha)+"_bestvalloss.dict")
            torch.save(net2.state_dict(),  "ABCD_model2_"+label+'_'+str(alpha)+"_bestvalloss.dict")
        #    log.write('%s\n' % json.dumps(state))
        #    log.flush()
        #    print(state)
            print("Best loss, alpha, epoch: %f" % best_loss,val_loss1,val_loss2,val_loss3, alpha, best_epoch)

        if(logfile=='log'):
            logfile=open('log', "w")

        w = csv.writer(logfile)
        for key, value in state.items():
            w.writerow([key, value])

#        if(patience>5): break


    return output1,output2,labels

        
