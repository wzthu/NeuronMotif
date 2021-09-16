import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import h5py
layer = int(sys.argv[1])
#kernel = int(sys.argv[2])
print('layer' + str(layer))
#print('kernel' + str(kernel))
#layer = 2
#kernel = 0

import keras


from modeldef import *
from utils import *

import h5py
import numpy as np
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, MaxPooling1D, BatchNormalization
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras.regularizers import l1,l2, l1_l2
from keras.constraints import MaxNorm
from keras.optimizers import SGD

from keras.activations import relu


import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

kernel=0

kernel_nb,kernel_sz,pool_sz,input_bp, input_bps, model_list, act_model_list, gd = get_model_list(layer = layer, kernel = kernel, weight_file='weight.hdf5')

submodel = model_list[-1]


ppmlist = []
smppmlist = []
label = []
countlist = []
actlist = []
conactlist = []
total = 1
for i in range(layer-1):
    total *= pool_sz[i]

for kernel in range(kernel_nb[layer-1]):
    print(kernel)
    with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.ppm.h5','r') as f:
        for j in range(total):
            if 'ppm' + str(j) in f.keys():
                spnumb = f['act' + str(j)][:].shape[0]
                countlist.append(spnumb)
                actlist.append(f['act' + str(j)][:].max())
                ppmlist.append(f['ppm' + str(j)][:][np.newaxis,:,:])
                smooth_ppm = (f['ppm' + str(j)][:][np.newaxis,:,:] * spnumb + np.ones((1,input_bp,4))*0.25)/(spnumb + 1)
                smppmlist.append(smooth_ppm)
                conactlist.append(f['conact'+ str(j)][:])
                print(f['conact'+ str(j)][:])
            else:
                ppmlist.append(np.ones((1,input_bp,4))*0.25)
                smppmlist.append(np.ones((1,input_bp,4))*0.25)
                countlist.append(0)
                actlist.append(0)
                conactlist.append(np.array([0]))



with h5py.File('layer'+str(layer)+'/allppm.h5','w') as f:
    f['allppm']=np.concatenate(ppmlist,axis=0)
    f['smoothppm']=np.concatenate(smppmlist,axis=0)
    f['act'] = np.array(actlist)
    f['spnumb'] = np.array(countlist)
    print(conactlist)
    f['conact'] = np.concatenate(conactlist,axis=0)
                
                    
                

