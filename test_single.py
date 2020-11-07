# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import cv2
import glob, os
import sys
import time
import random
import shutil
from scipy import ndimage
from tqdm import trange
# import pydicom
from metrices import *
import utils
from PIL import Image
from skimage import io

import pdb


current_dir = os.path.dirname(os.path.abspath(__file__))
src_folder = '/cvdata/VesselNN/test/volume'
src_folder_GT = '/cvdata/VesselNN/test/GT'
src_folder_files = os.listdir(src_folder)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='/cvdata/model/model_epoch_100.pth',
                    help='set the checkpoint name')
parser.add_argument('--input-channels', default=9, type=int, help='input channels')
parser.add_argument('-p', '--plane', help='direction of projection', default=0, type=int)


args = parser.parse_args()


plane = args.plane

model_path = args.checkpoint
model = utils.load_checkpoint(model_path)

model = nn.DataParallel(model).cuda()
model.eval()


# F1_score = []
# JI = []
# Dice = []
# IoU = []
# Sensi = []
# Speci = []

for vol_id, vol in enumerate(tqdm((src_folder_files), desc='patient cases in test folder')):

    start_time = time.time()
    srcfile_path = os.path.join(src_folder, vol)
    image_volume = io.imread(srcfile_path)

    srcfile_path_GT = os.path.join(src_folder_GT, vol)
    image_volume_GT = io.imread(srcfile_path_GT)/255
    C, W, L = image_volume.shape[0], image_volume.shape[1], image_volume.shape[2]
    
    if W % 32:
        Wk = W + (32 - W % 32)
        image_volume = np.pad(image_volume, ((0,0),(0,Wk-W), (0,0)), 'constant', constant_values=0)
        image_volume_GT = np.pad(image_volume_GT, ((0,0),(0,Wk-W), (0,0)), 'constant', constant_values=0)
    if L % 32:
        Lk = L + (32 - L % 32)
        image_volume = np.pad(image_volume, ((0,0),(0,0), (0,Lk-L)), 'constant', constant_values=0)
        image_volume_GT = np.pad(image_volume_GT, ((0,0),(0,0), (0,Lk-L)), 'constant', constant_values=0)
    
    C, W, L = image_volume.shape[0], image_volume.shape[1], image_volume.shape[2]
    image_volume_output = np.empty(shape=(0,W,L))
    
    input_volume = np.empty(shape = (0,3,W,L), dtype=np.float32)
    for i in range(C):
       image = np.stack((image_volume[i,:,:],image_volume[i,:,:],image_volume[i,:,:]),axis = 0)
       Img = image.astype('float32')/255
       input_volume = np.append(input_volume, Img[np.newaxis,...], axis=0)

    first_image = np.stack((image_volume[0,:,:],image_volume[0,:,:],image_volume[0,:,:]),axis = 0).astype('float32')/255
    last_image = np.stack((image_volume[C-1,:,:],image_volume[C-1,:,:],image_volume[C-1,:,:]),axis = 0).astype('float32')/255     

    input_volume = np.insert(input_volume, 0, first_image[np.newaxis, ...], axis = 0)
    input_volume = np.append(input_volume, last_image[np.newaxis, ...], axis = 0)
    
    CC = input_volume.shape[0]
    for i in range(CC-2):
        Img = input_volume[i:i+3,:,:,:]
        Img = torch.from_numpy(Img).unsqueeze(0).cuda()
        out = model(Img, Img[:,1,:,32:64,32:64])[1].detach().squeeze(0).squeeze(0)
        out = torch.sigmoid(out).cpu().numpy()
        image_volume_output = np.append(image_volume_output,out[np.newaxis,...], axis=0)

        






       










