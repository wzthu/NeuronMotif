import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import h5py
#layer = 3
#kernel = 0
import pandas as pd
import keras
import os

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





def dec2four(num, seq_size=9):
    l = []
    while True:
        num, remainder = divmod(num, 4)
        l.append(remainder)
        if num == 0:
            if seq_size is None:
                return l[::-1]
            else:
                s = len(l)
                s = seq_size - s
                if s > 0:
                   other = [0] * s
                else:
                   other = []
                other.extend(l[::-1])
                return other

def to_onehot(seq_code, seq_size, step=9):
    s = int(seq_size/step)
    l = []
    for i in range(s):
        l.extend(dec2four(seq_code[i],step))
    if s*step != seq_size:
        l.extend(dec2four(seq_code[s],seq_size-s*step))
    l = np.array(l,dtype=int)
    sp = np.zeros((1,seq_size,4))
    sp[0,l==0,0] = 1
    sp[0,l==1,1] = 1
    sp[0,l==2,2] = 1
    sp[0,l==3,3] = 1
    return sp

def gen_sample_code(num ,seq_size, step=9):
    lst = []
    for i in range(int(seq_size/step)):
       lst.append(np.random.choice(pow(4,step), num,replace=True)[:,np.newaxis])
    s = seq_size - int(seq_size/step) * step
    if s != 0:
       lst.append(np.random.choice(pow(4,s), num,replace=True)[:,np.newaxis])
    return np.concatenate(lst,axis=1)



def gen_sample(batchsize, sample_code, seq_size, seq_step=9):
    steps = int(sample_code.shape[0] / batchsize)
    if sample_code.shape[0] != batchsize * steps:
        steps += 1
    while True:
        for i in range(steps):
            sz = batchsize
            if i == steps -1:
                sz = sample_code.shape[0] - batchsize * (steps-1)
            code = sample_code[(i*batchsize):(i*batchsize+sz),:]
            datalist = [to_onehot(code[j,:], seq_size, seq_step) for j in range(sz)]
            data = np.concatenate(datalist, axis=0)
            yield [data]

def val_uniq_seq(allseqs, allvals, max_sample=None):
    allseqs = allseqs[allvals>0,:,:]
    allvals = allvals[allvals>0]
    allseqseq = allseqs.reshape((allseqs.shape[0], allseqs.shape[1]*allseqs.shape[2]))
    allseqseq = np.array(np.array(allseqseq, dtype=int), dtype=str)
    allseqseq = np.array([''.join(allseqseq[i,:]) for i in range(allseqseq.shape[0])], dtype='str')
    df = pd.DataFrame({'seq':allseqseq,'index':range(allseqseq.shape[0])})
    df.drop_duplicates(subset='seq',keep='first', inplace=True)
    allvals = allvals[np.array(df['index'])]
    allseqs = allseqs[np.array(df['index']),:,:]
    if max_sample is not None and allseqs.shape[0] > max_sample:
        maxpvalues = np.max(allvals)
        sellist = []
        sizes = []
        sizes1 = []
        for j in range(50,0,-1):
            low = (maxpvalues * (j-1))/50
            high = (maxpvalues * j)/50
            sel = np.logical_and(low < allvals, allvals <= high)
            sel = np.arange(allseqs.shape[0])[sel]
            sizes.append(sel.shape[0])
            if sel.shape[0] < max_sample /50:
                sellist.append(sel)
            else:
                sellist.append(np.random.choice(sel,int(max_sample/50),replace=False))
            sizes1.append(sellist[-1].shape[0])
        sellist = np.concatenate(sellist, axis=0)        
        allseqs = allseqs[sellist,:,:]
        allvals = allvals[sellist]
        print(sizes)
        print(sizes1)
    return allseqs, allvals



