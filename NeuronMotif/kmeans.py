import h5py
import numpy as np
np.random.seed(0)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import h5py
layer = int(sys.argv[1])
kernel = int(sys.argv[2])
motif_nb = 1
if len(sys.argv) > 3:
    motif_nb = int(sys.argv[3])
print('layer' + str(layer))
print('kernel' + str(kernel))
print('motif_nb'+ str(motif_nb))
#layer = 2
#kernel = 0

import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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




# 指定第一块GPU可用
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分>配
config.gpu_options.per_process_gpu_memory_fraction = 0.30
sess = tf.Session(config=config)

KTF.set_session(sess)



batch_size=128

kernel_nb,kernel_sz,pool_sz,input_bp, input_bps, model_list, act_model_list, gd = get_model_list(layer = layer, kernel = kernel, weight_file='weight.hdf5')

submodel = model_list[-1]

new_pool_sz = [1] * len(pool_sz)

for i in range(motif_nb):
    new_pool_sz = [ new_pool_sz[j] * pool_sz[j] for j in range(len(pool_sz))]

pool_sz = new_pool_sz

import tensorflow as tf
from keras import backend  as K
sess = K.get_session()

with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.h5','r') as f:
    allseqs = f['seq1'][:]
    allvs = f['value'][:]

print('finish1')

from sklearn.cluster import KMeans,MiniBatchKMeans
def cluster_motif(allseqs, layer, index):
    if layer==0 or index.shape[0]< pool_sz[layer-1] or index.shape[0]< 50:
        return [index]
    thisseqs = allseqs[index,:,:]
    thisvs = allvs[index]
    actvalue = model_list[layer-1].predict(thisseqs)
    actvalue = actvalue.reshape((actvalue.shape[0],actvalue.shape[1]*actvalue.shape[2]))
#    actvalue[actvalue<0] = 0
    kms = MiniBatchKMeans(n_clusters=pool_sz[layer-1])
    labels = kms.fit_predict(actvalue,sample_weight=thisvs)
#    labels = kms.fit_predict(actvalue)
    if layer > 1:
        lst = []
        for i in range(pool_sz[layer-1]):
            lst.extend(cluster_motif(allseqs, layer-1, index=index[labels==i]))
        return lst
    else:
        return [index[labels==i] for i in range(pool_sz[layer-1])]

clist = cluster_motif(allseqs = allseqs, layer = layer-1, index = np.arange(allseqs.shape[0]))

print('finish2')

def resample_top(ponehots, pvalues):
    maxpvalues = np.max(pvalues)
    vlist = []
    sellist = []
    for j in range(20,0,-1):
        low = (maxpvalues * (j-1))/20
        high = (maxpvalues * j)/20
        sel = np.logical_and(low < pvalues, pvalues <= high)
        sellist.append(sel)
        if sel.sum()>0:
            singleV = pvalues[sel]
            vlist.append(singleV.mean())
        else:
            vlist.append((low + high)/2)
    vlist = np.array(vlist)
    vlist = vlist / vlist.sum()
    sm = 0
    allsel = sellist[0]
    for i in range(1,vlist.shape[0]):
        sm += vlist[i]
        if sm > 0.05:
            break
        else:
            allsel = np.logical_or(allsel,sellist[i])
    print(allsel.shape)
    print(ponehots.shape)
    resampleseqs = ponehots[allsel,:,:]
    splist = []
    spvlist = []
    for i in range(4):
        for j in range(ponehots.shape[1]):
            newseq = resampleseqs.copy()
            newseq[:,j,:] = 0
            newseq[:,j,i] = 1
            newvalue =model_list[-1].predict(newseq)[:,0,kernel]
            splist.append(newseq[newvalue>0,:,:])
            spvlist.append(newvalue[newvalue>0])
    newseqs = np.concatenate(splist, axis=0)
    newvals = np.concatenate(spvlist,axis=0)
    return val_uniq_seq(np.concatenate([ponehots, newseqs], axis=0 ), np.concatenate([pvalues,newvals],axis=0 ))
    
    
print('finish3')

f=h5py.File('layer' + str(layer) + '/kernel-' + str(kernel) + '.ppm.h5' ,'w')
for i in range(len(clist)):
    print('finish3'+str(i))
    if len(clist[i])==0:
        continue
    f['index' + str(i)] = clist[i]
    f['act' + str(i)] = allvs[clist[i]]
    f['pfma' + str(i)] = allseqs[clist[i],:,:].sum(axis=0)
    pvalues = allvs[clist[i]]
    ponehots = allseqs[clist[i],:,:]
#    ponehots, pvalues = resample_top(ponehots, pvalues)
    maxpvalues = np.max(pvalues)
    pfmlist = []
    vlist = []
    for j in range(20,0,-1):
        low = (maxpvalues * (j-1))/20 
        high = (maxpvalues * j)/20 
        sel = np.logical_and(low < pvalues, pvalues <= high)
        if sel.sum()>0:
            singlePfm = ponehots[sel,:,:]
            singleV = pvalues[sel]
            vlist.append(singleV.mean())
            singlePfm = singlePfm.sum(axis=0)
            singlePfm = singlePfm.transpose((1,0))
            singlePfm = singlePfm / singlePfm.sum(axis=0)
            singlePfm = singlePfm.transpose((1,0))
            pfmlist.append(singlePfm[np.newaxis,:,:])
        else:
            pfmlist.append(pfmlist[-1])
            vlist.append((low + high)/2)
    ppm = [ pfmlist[i]*vlist[i] for i in range(len(vlist))]
    ppm = np.concatenate(ppm, axis=0)
    ppm = ppm.sum(axis=0)
    ppm = ppm.transpose((1,0))
    ppm = ppm / ppm.sum(axis=0)
    ppm = ppm.transpose((1,0))
    f['ppm' + str(i)] = ppm
    con = ppm.argmax(axis=1)
    conseq = np.zeros(ppm.shape)
    for m in range(con.shape[0]):
        conseq[m,con[m]] = 1
    rs = model_list[-1].predict(conseq[np.newaxis,:,:])[:,0,kernel] 
    print(rs.shape)
    f['conact' + str(i)] = rs


f.close()

