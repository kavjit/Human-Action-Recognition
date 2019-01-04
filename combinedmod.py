import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2

from multiprocessing import Pool

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)


##### save predictions directory
pred_dir_sf = 'UCF-101-predictions/'
pred_dir_3d = 'UCF-101-3d-predictions/'


acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)



for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')
    
    hs = h5py.File(filename,'r')
    nFrames = len(hs['video'])


    filename1 = filename.replace(data_directory+'UCF-101-hdf5/',pred_dir_sf)
    if(os.path.isfile(filename1)):
        with h5py.File(filename1,'r') as h:
            pred_sf = h['predictions'][()] 
    filename2 = filename.replace(data_directory+'UCF-101-hdf5/',pred_dir_3d)
    if(os.path.isfile(filename2)):
        with h5py.File(filename2,'r') as h:
            pred_3d = h['predictions'][()]


    for j in range(pred_sf.shape[0]):
        pred_sf[j] = np.exp(pred_sf[j])/np.sum(np.exp(pred_sf[j]))
    

    for j in range(pred_3d.shape[0]):
        pred_3d[j] = np.exp(pred_3d[j])/np.sum(np.exp(pred_3d[j]))

    pred_sf = np.sum(np.log(pred_sf),axis=0)
    pred_3d = np.sum(np.log(pred_3d),axis=0)
    
    predcmb = (pred_sf + pred_3d)/2
    argsort_pred = np.argsort(-predcmb)[0:10]

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))



number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('confmatrix_combinedmod.npy',confusion_matrix)
