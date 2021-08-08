import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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


seq = Conv1D(64, 3)(seqInput)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(128, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(256, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(384, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(512, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Flatten()(seq)
seq = Dense(1024)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Dropout(0.2)(seq)
seq = Dense(164)(seq)
seq = Activation('sigmoid')(seq)

model = Model(inputs = [seqInput], outputs = [seq])


from keras.optimizers import RMSprop
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

chroms = ['chr'+str(i) for i in range(1,23)]
chroms.append('chrX')
chroms.append('chrY')

with h5py.File('data/onehot.h5', 'r') as f:
    onehot = dict()
    for chrom in chroms:
        onehot[chrom] = f[chrom][:]




import pandas as pd

bed = pd.read_csv('data/encode_roadmap.bed', sep='\t', header=None)

label = np.zeros((bed.shape[0],164))

for i in range(bed.shape[0]):
    label[i,np.array(bed.iloc[i,6].split(','),dtype=int)] = 1

with h5py.File('data/sample.h5','r') as f:
    seltest = f['seltest'][:]
    selval = f['selval'][:]
    seltrain = f['seltrain'][:]

test_data = {'sample':bed.iloc[seltest,:], 'label':label[seltest,:],'seq_onehot': onehot }

val_data = {'sample':bed.iloc[selval,:], 'label':label[selval,:] ,'seq_onehot': onehot}
train_data = {'sample':bed.iloc[seltrain,:], 'label':label[seltrain,:],'seq_onehot': onehot }

train_randseq = Randseq(train_data['sample'].shape[0])
val_randseq = Randseq(val_data['sample'].shape[0], True)
steps_per_epoch = int(train_randseq.seqsize / batch_size)
if train_randseq.seqsize != batch_size * steps_per_epoch:
    steps_per_epoch += 1

validation_steps = int(val_randseq.seqsize / batch_size)
if val_randseq.seqsize != batch_size * validation_steps:
    validation_steps += 1

sample_generator = gen_samples


train_gen = sample_generator(batchsize=batch_size,
                                randseq=train_randseq,
                                data=train_data)

val_gen = sample_generator(batchsize=batch_size,
                              randseq=val_randseq,
                              data=val_data)

callbacks=[]
callbacks.append(ModelCheckpoint(filepath='weight.hdf5',save_best_only=True))
callbacks.append(EarlyStopping(patience=12))

epochs=100

history = model.fit_generator(train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=val_gen,
                                      validation_steps=validation_steps, epochs=epochs,
                                      verbose=1,
                                      callbacks=callbacks,
                                      max_queue_size=10)




history_dict=history.history

    #Plots model's training cost/loss and model's validation split cost/loss
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.figure()
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='val training loss')

plt.savefig('history.pdf')
#rs = model.predict(oh)[0,:]


with h5py.File('history.h5','w') as f:
    f['loss_values'] =loss_values
    f['val_loss'] = val_loss_values
    f['seltest'] = seltest
