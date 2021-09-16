import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import h5py
layer = int(sys.argv[1])
kernel = int(sys.argv[2])
print('layer' + str(layer))
print('kernel' + str(kernel))
#layer = 3
#kernel = 0

import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import h5py
import numpy as np


import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF



# 指定第一块GPU可用
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分>配
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)

KTF.set_session(sess)

from modeldef import *
from utils import * 

kernel_nb,kernel_sz,pool_sz,input_bp, input_bps, model_list, act_model_list, gd = get_model_list(layer = layer, kernel = kernel,  weight_file='weight.hdf5')

act_submodel =act_model_list[-1]
submodel = model_list[-1]

import tensorflow as tf
from keras import backend  as K
sess = K.get_session()



if input_bp < 11:
#    wmtx = submodel.layers[-1].get_weights()[0][:,:,kernel]
    spsize = pow(4,input_bp)
    code = np.arange(spsize)[:,np.newaxis]
    seq0 = []
    for x_batch,i in zip(gen_sample(spsize, code, input_bp, 10),range(1)):
        seq0.append(x_batch[0])
    seq0 = np.concatenate(seq0,axis=0)
    value0 = submodel.predict(seq0)[:,0,kernel]
    if not os.path.exists('layer'+str(layer)+'/'):
        os.mkdir('layer'+str(layer)+'/')
    with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.h5','w') as f:
        f['seq0']=seq0
        f['seq1']=seq0[value0>0,:,:]
        f['value']=value0[value0>0]
    exit()

        
if layer == 1:
    allppm = np.ones((1,1,4)) * 0.25
    conact = np.ones(1)
    acts = np.ones(4)
else:    
    with h5py.File('layer' + str(layer-1) + '/allppm.h5','r') as f:
        allppm=f['allppm'][:]
        acts = f['act'][:]
        spnumb = f['spnumb'][:]
        conact = f['conact'][:]

spsize  = int(input_bp/20) * 1000

if spsize> 5000:
    spsize = 5000

if spsize<2000:
    spsize = 2000

code = gen_sample_code(spsize,input_bp, step=9)
seq0 = []

for x_batch,i in zip(gen_sample(spsize, code, input_bp, 9),range(1)):
    seq0.append(x_batch[0])
    print(i)

seq0 = np.concatenate(seq0,axis=0)


seq = seq0.copy()

rev = False

#reverse complementary
if rev:
    ppms = np.concatenate([allppm,allppm[::-1,::-1,::-1]], axis=0)
    conacts = np.r_[conact,conact[::-1]]
    ic = np.zeros((ppms.shape[0],ppms.shape[1]))
else:
    ppms = allppm.copy()
    conacts = conact.copy()
    ic = np.zeros((ppms.shape[0],ppms.shape[1]))

for i in range(ppms.shape[0]):
    for j in range(ppms.shape[1]):
        if ppms[i,j,:].sum() < 0.001:
            ppms[i,j,:] = 0.25
            ic[i,j]=0
        else:
            ppms[i,j,:] /= ppms[i,j,:].sum()
            addsmall = ppms[i,j,:] + 0.000001
            ic[i,j] = 2 - (- addsmall * np.log2(addsmall)).sum()

ismotif = (ic>1).sum(axis=1)>3
motifpos = []
for i in range(ic.shape[0]):
    if ismotif[i]:
        motifpos.append([np.where(ic[i,:]>1)[0][0],np.where(ic[i,:]>1)[0][-1]])
    else:
        motifpos.append([-1,-1])

motifpos = np.array(motifpos)

lastpool = 1
for i in range(layer-2):
    lastpool *=pool_sz[i]

wmtx = act_submodel.layers[-1].get_weights()[0][:,:,kernel]
actsmx = acts.reshape((int(acts.shape[0]/lastpool),lastpool))
actsmx = actsmx.sum(axis=1)
if rev:
    actsmx = np.r_[actsmx,actsmx[::-1]]


wmtx *= actsmx

import h5py
if wmtx.max() <= 0:
    if not os.path.exists('layer'+str(layer)+'/'):
        os.mkdir('layer'+str(layer)+'/')
    with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.h5','w') as f:
        f['seq0']=np.zeros((0,seq0.shape[1],seq0.shape[2]))
        f['seq1']=np.zeros((0,seq0.shape[1],seq0.shape[2]))
        f['value']=np.zeros((0))
    exit()

spidx = np.where(wmtx>0)
spw = wmtx[spidx]
normspw = np.zeros(spw.shape)
top = spw.max()
sel_list = []
for i in range(20,0,-1):
    sel = np.logical_and(spw > top*(i-1)/20, spw <= top*i/20)
    if sel.sum()==0:
        sel = sel_list[-1]
    sm = spw[sel].sum()
    normspw[sel] += ((spw[sel]/sm)/20)
    sel_list.append(sel)

normspw = normspw/normspw.sum()

if layer>1:
    poolsz = pool_sz[layer-2]
else:
    poolsz = 1



selidx = [np.random.choice(normspw.shape[0],poolsz,replace=False,p=normspw) for i in range(seq0.shape[0])]


selidx = np.array(selidx,dtype=int)

rgsz = ppms.shape[1]
rgstep = 1
for i in range(layer-1):
    rgstep *= pool_sz[i]

prgstep = 1
for i in range(layer-2):
    prgstep *= pool_sz[i]



for i in range(seq0.shape[0]):
    if layer == 1:
        break
    ifput = np.zeros((seq0.shape[1]))
#    print(i)
    for j in range(poolsz):#position
#        print(spidx[0][selidx[i,j]],spidx[1][selidx[i,j]])
        st = spidx[1][selidx[i,j]]*prgstep
        ed = st + prgstep
#        ismotif[st:ed].sum()>0
#        selpfmid = np.random.choice(np.arange(st,ed,1)[ismotif[st:ed]],ismotif[st:ed].sum(),replace=False)
        selpfmid = np.random.choice(np.arange(st,ed,1),prgstep,replace=False)
        for k in range(selpfmid.shape[0]):#same motifs at the same position
            seqst = int(spidx[0][selidx[i,j]] * rgstep + prgstep*np.random.choice(poolsz,1))
            seqed = int(seqst + input_bps[layer-2])
            mtfst = int(seqst + motifpos[selpfmid[k],0])
            mtfed = int(seqst + motifpos[selpfmid[k],1])
            if ifput[mtfst:mtfed].sum()>0:
                pass
            else:
#                ifput[mtfst:mtfed] = 1
                spppm = ppms[selpfmid[k],:,:]
#                spseqidx = [np.random.choice(4,1,p=sppfm[t,:]) for t in range(sppfm.shape[0])]
                spseqidx = spppm.argmax(axis=1)
                for p in range(spppm.shape[0]):
                    if ifput[seqst +p] > 0:
                        pass
                    else:
                        seq0[i,seqst+p,:] = 0
                        seq0[i,seqst+p,spseqidx[p]] = 1
                if conacts[selpfmid[k]]>0:
                    ifput[mtfst:mtfed] = 1
#                print(md_4.predict(seq0[i:(i+1),:,:])[0,spidx[0][selidx[i,j]], spidx[1][selidx[i,j]]])
#                print(md_4_1.predict(seq0[i:(i+1),seqst:seqed,:])[0,0,spidx[1][selidx[i,j]]])





import pandas as pd


v0 = submodel.predict(seq0[:,:,:])[:,0,kernel]
seq0bk = seq0.copy()

allseqs = seq0.copy()
allvs = v0.copy()



onemore = False
#maxmutate = 10
#maxvalue = allvs.copy()

#tmpallseqs.append(seq0)
#tmpallvs.append(v0)
batch_size = spsize

alpha = 1

step = int(spsize/batch_size)
maxmeanvalue = -10000000 
keepmax = 0
for it in range(200):
    if keepmax > 10:
        break
    else:
#if True:
#    for i in range(step):
        #print(i)
#        old = seq0[i*batch_size:(i+1)*batch_size,:,:]
        old = seq0.copy()
        seqbatch = old.copy() 
#        md_2.predict(seqbatch)[:,0,0].mean()
        newgd = sess.run(gd, {'subseqInput:0':seqbatch})[0]
        seqbatch =  seqbatch + alpha*newgd
#        sel = seqbatch.argmax(axis=2)
#        seqbatch[:]=0
        seqbatch[seqbatch<0]=0
        sm = seqbatch.sum(axis=2)
        sm1 = np.concatenate([sm[:,:,np.newaxis]]*4,axis=2)
        seqbatch[sm1==0] = 0.25
        sm1[sm1==0] = 1
        seqbatch /= sm1
        sseqbatch = seqbatch.reshape((seqbatch.shape[0]*seqbatch.shape[1],seqbatch.shape[2]))
        idx = [np.random.choice(4,p=sseqbatch[t,:]) for t in range(sseqbatch.shape[0])]
        newseqbatch = np.zeros(sseqbatch.shape)
        newseqbatch[(tuple(range(sseqbatch.shape[0])),tuple(idx))]=1
        newseqbatch = newseqbatch.reshape(old.shape)
#        md_2.predict(seqbatch)[:,0,0].mean()
        newvalue = submodel.predict(newseqbatch)[:,0,kernel]
#        oldvalue = submodel.predict(old)[:,0,kernel]
#        old[newvalue>oldvalue,:,:]=seqbatch[newvalue>oldvalue,:,:]
#        seq0[i*spsize:(i+1)*spsize,:,:] = tmpseqbatch
#        newv = md_4.predict(seqbatch)[:,0,kernel]
        if allseqs.shape[0] > 100000 and it % 5 == 4: 
            allseqs, allvs = val_uniq_seq(np.concatenate([allseqs,newseqbatch[newvalue>0,:,:].copy()]), np.concatenate([allvs,newvalue[newvalue>0].copy()]), 100000)
        else:
            allseqs = np.concatenate([allseqs,newseqbatch[newvalue>0,:,:].copy()])
            allvs = np.concatenate([allvs,newvalue[newvalue>0].copy()])
        prob = newvalue.copy()
        prob[prob<0] = np.exp(prob[prob<0])
        prob[prob>=0] = prob[prob>=0] + 1
        prob /= prob.sum()
        keepseq = newseqbatch[newvalue.argsort()[::-1][0:int(batch_size*0.1)],:,:]
        cross_rate = 0.45
        shiftseqs = []
        if  rgstep>1:
            cross_rate = 0.35
            shift_range = np.arange(1,rgstep)
            shift_size = np.random.choice(shift_range, int(batch_size*0.1))
            shiftseq = np.zeros(keepseq.shape)
            for s in range(shiftseq.shape[0]):
                shiftseq[s, 0:(input_bp-shift_size[s]), :] = keepseq[s, shift_size[s]:input_bp,:]
                shiftseq[s, (input_bp-shift_size[s]):input_bp, :] = keepseq[s, 0:shift_size[s],:]
            shiftseqs.append(shiftseq)
            shift_size = np.random.choice(shift_range, int(batch_size*0.1))
            shiftseq = np.zeros(keepseq.shape)
            for s in range(shiftseq.shape[0]):
                shiftseq[s, shift_size[s]:input_bp, :] = keepseq[s,0:(input_bp-shift_size[s]) ,:]
                shiftseq[s, 0:shift_size[s], :] = keepseq[s, (input_bp-shift_size[s]):input_bp,:]
            shiftseqs.append(shiftseq)
            shiftseqs.append(keepseq)
            keepseq = np.concatenate(shiftseqs,axis=0)
#        print(submodel.predict(keepseq)[:,0,kernel].max())
        selpair = [newseqbatch[np.random.choice(prob.shape[0],2,p=prob),:,:] for i in range(int(batch_size*cross_rate))]
        selpair = np.concatenate(selpair,axis=0)
        newselpair = selpair.copy()
        changest = np.array(np.random.rand(int(newselpair.shape[0]/2))*newselpair.shape[1],dtype=int)
        for j in range(0,int(newselpair.shape[0]/2)):
            newselpair[j*2,changest[j]:,:] = selpair[j*2+1,changest[j]:,:]
            newselpair[j*2+1,changest[j]:,:] = selpair[j*2,changest[j]:,:]
#        seq0[i*spsize:(i+1)*spsize,:,:] = np.concatenate([keepseq,newselpair],axis=0)
        seq0[:] = np.concatenate([keepseq,newselpair],axis=0)
        m = newvalue.mean()
        if m > maxmeanvalue:
            maxmeanvalue = m
            keepmax = 0 
        else:
            keepmax += 1
        print(newvalue.mean(),keepmax)
        print(newvalue.max())
        print(np.sum(newvalue>0))

allseqs, allvs = val_uniq_seq(allseqs, allvs, 100000)


import h5py
if not os.path.exists('layer'+str(layer)+'/'):
    os.mkdir('layer'+str(layer)+'/')

with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.h5','w') as f:
    f['seq0']=seq0bk
    f['seq1']=allseqs
    f['value']=allvs

