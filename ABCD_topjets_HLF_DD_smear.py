from __future__ import print_function, division

import sys,os
import argparse

parser = argparse.ArgumentParser(description='Train DNN with decorrelation using tau32 and frec as input.')
#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
parser.add_argument('--gpunum', default='0',
                    help='gpu number (default 0)')
#parser.add_argument('--alphamin', default='0',
#                    help='minimum alpha (default 0)')
#parser.add_argument('--alphamax', default='1',
#                    help='maximum alpha (default 1)')
#parser.add_argument('--nalpha', default='20',
#                    help='number of alpha (default 20)')
parser.add_argument('--logfile', default='log.csv',
                    help='log file')
parser.add_argument('--smear', default='25',
                    help='gaussian smear mass sigma')

results = parser.parse_args(sys.argv[1:])
print(results)
#print(results.decorr_mode)
#print(results.gpunum)

gpunum=results.gpunum
#alphamin=float(results.alphamin)
#alphamax=float(results.alphamax)
#nalpha=int(results.nalpha)
logfilename=results.logfile
smear=float(results.smear)

os.environ["CUDA_VISIBLE_DEVICES"]=gpunum

import torch
import torch.nn as nn
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')
import pandas as pd
from skimage import io, transform
import numpy as np
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

from torch.nn import init

from data_loader import TopTaggingDataset
import model
from model_ABCD_2NN import train,val,train_model
from networks import DNNclassifier

#decorr_mode=sys.argv[1]
#print('decorr_mode',decorr_mode)

import pandas as pd

import glob

from torch.optim import Optimizer


traindata_raw=np.loadtxt('../toptagging/topsample_train_tau.dat',delimiter=',',skiprows=15)
valdata_raw=np.loadtxt('../toptagging/topsample_val_tau.dat',delimiter=',',skiprows=15)
testdata_raw=np.loadtxt('../toptagging/topsample_test_tau.dat',delimiter=',',skiprows=15)

alldata=np.concatenate((traindata_raw,valdata_raw,testdata_raw))
# Smear masses by a gaussian to degrade the power of the mass variable
alldata[:,1]+=smear*np.random.normal(size=len(alldata))

alldata2=(alldata[:,1:]-np.min(alldata[:,1:],axis=0))/(np.max(alldata[:,1:],axis=0)-np.min(alldata[:,1:],axis=0))
alllabels=alldata[:,0].reshape((-1,1))
allweights=np.ones(len(alldata)).reshape((-1,1))
allbinnums=np.ones(len(alldata)).reshape((-1,1))
allmasses=alldata[:,1].reshape((-1,1))

alldata3=torch.from_numpy(np.hstack((alldata2,alllabels,allweights,allbinnums,allmasses)).astype('float32'))

Ntrain=200000
Nval=900000
Ntest=900000
traindata=alldata3[:Ntrain]
valdata=alldata3[Ntrain:(Ntrain+Nval)]
#testdata=alldata3[(Ntrain+Nval):(Ntrain+Nval+Ntest)]

########
# 
    
trainset = TopTaggingDataset(traindata[:,:-4],traindata[:,-4],traindata[:,-3],traindata[:,-2],traindata[:,-1])
valset = TopTaggingDataset(valdata[:,:-4],valdata[:,-4],valdata[:,-3],valdata[:,-2],valdata[:,-1])

#num_workers=4
my_batch_size=10000
train_loader = DataLoader(trainset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)
val_loader = DataLoader(valset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)

logfile=open(logfilename, "w")
#alphalist=np.linspace(alphamin,alphamax,nalpha,endpoint=False)

# alphalist=[50,100,200,400,800]
# alphalist=[50,100,200]
#alphalist=[400,800]
alphalist=[100]
print(alphalist)
for alpha in alphalist:
    alpha=float(alpha)
    print('alpha',alpha)

    net1 = DNNclassifier(13,2)
    net2 = DNNclassifier(13,2)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    
    net1.to(device)
    net2.to(device)

    optimizer1 = torch.optim.Adam(net1.parameters())
    optimizer2 = torch.optim.Adam(net2.parameters())

    output1,output2,labels=train_model(200,1e-3,net1,net2,optimizer1,optimizer2,train_loader,val_loader
                          ,decorr_mode='dist_unbiased',alpha=alpha
                          ,logfile=logfile,label='smear'+str(smear))


logfile.close()

