import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py


import keras


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

from data.util import Randseq, fill_oh, gen_samples

input_bp = 600

batch_size=128

seqInput = Input(shape=(input_bp, 4), name='seqInput')



seq = Conv1D(300, 19)(seqInput)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(pool_size=3)(seq)
seq = Conv1D(200, 11)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(pool_size=4)(seq)
seq = Conv1D(200, 7)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(pool_size=4)(seq)
seq = Flatten()(seq)
seq = Dense(1000)(seq)
seq = Activation('relu')(seq)
seq = Dropout(0.3)(seq)
seq = Dense(1000)(seq)
seq = Activation('relu')(seq)
seq = Dropout(0.3)(seq)
seq = Dense(164)(seq)
seq = Activation('sigmoid')(seq)


model = Model(inputs = [seqInput], outputs = [seq])
model.load_weights('weight.hdf5')


chroms = ['chr'+str(i) for i in range(1,23)]
chroms.append('chrX')
chroms.append('chrY')

with h5py.File('data/onehot.h5', 'r') as f:
    onehot = dict()
    for chrom in chroms:
        onehot[chrom] = f[chrom][:]



model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])


import pandas as pd

bed = pd.read_csv('data/encode_roadmap.bed', sep='\t', header=None)

label = np.zeros((bed.shape[0],164))

for i in range(bed.shape[0]):
    label[i,np.array(bed.iloc[i,6].split(','),dtype=int)] = 1


with h5py.File('data/sample_sel.h5','r') as f:
    seltest = f['seltest'][:]


test_data = {'sample':bed.iloc[seltest,:], 'label':label[seltest,:],'seq_onehot': onehot }

test_randseq = Randseq(test_data['sample'].shape[0], True)

test_steps = int(test_randseq.seqsize / batch_size)
if test_randseq.seqsize != batch_size * test_steps:
    test_steps += 1

sample_generator = gen_samples



test_gen = sample_generator(batchsize=batch_size,
                              randseq=test_randseq,
                              data=test_data)


result = model.predict_generator(generator=test_gen,
                                      steps = test_steps,
                                      verbose=1)



from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
auprc=np.array([average_precision_score(test_data['label'][:,i],result[:,i]) for i in range(test_data['label'].shape[1])])
auroc=np.array([roc_auc_score(test_data['label'][:,i],result[:,i]) for i in range(test_data['label'].shape[1])])

with h5py.File('test.h5','w') as f:
    f['test_result'] = result
    f['test_label'] = test_data['label']
    f['seltest'] = seltest
    f['auprc'] = auprc
    f['auroc']=auroc
