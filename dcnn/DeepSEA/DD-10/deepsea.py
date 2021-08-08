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

from atac2d.train.genome_layers import RevCompConv1D, FlipConv2D, DenseAfterRevcompConv1D, DenseAfterFlipConv2D,RevCompConv1DBatchNorm


input_bp = 1000
conv_kernel_size = 8
pool_kernel_size = 4

maxnrom = MaxNorm(max_value=0.9, axis=0)
l1l2 = l1_l2(l1=1e-8, l2=1e-6)

def crelu(x, alpha=0.0, max_value=None, threshold=1e-6):
    return relu(x, alpha, max_value, threshold)

batch_size=128

seqInput = Input(shape=(input_bp, 4), name='seqInput')


seq = Conv1D(128, 7)(seqInput)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Conv1D(128, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(160, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Conv1D(160, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(256, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Conv1D(320, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(512, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Conv1D(640, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(1024, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Conv1D(1280, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Flatten()(seq)
seq = Dense(925)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = Dropout(0.2)(seq)
seq = Dense(919)(seq)
seq = Activation('sigmoid')(seq)

model = Model(inputs = [seqInput], outputs = [seq])


from scipy.io import loadmat

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

import h5py
with h5py.File('data/train.mat', 'r') as trainmat:
    trainy = trainmat['traindata'][:]
    trainx = trainmat['trainxdata'][:]

#trainmat = loadmat('data/train.mat')
valmat = loadmat('data/valid.mat')
#trainy = trainmat['traindata']
#trianx = trainmat['trainxdata']
validx = valmat['validxdata']
validy = valmat['validdata']

callbacks=[]
callbacks.append(ModelCheckpoint(filepath='weight-{epoch:02d}-{val_loss:.4f}.hdf5',save_best_only=False))
callbacks.append(EarlyStopping(patience=5))

trainx = trainx.transpose((2,0,1))
trainy = trainy.transpose((1,0))
validx = validx.transpose((0,2,1))
#validy = validy.transpose((1,0))


history = model.fit(x = trainx,
          y = trainy,
          epochs = 50,
          verbose = 1,
          batch_size=batch_size,
          validation_data = (validx, validy),
#          validation_batch_size = batch_size*2,
          callbacks = callbacks
)
"""
history = model.fit(x = validx,
          y = validy,
          epochs = 1,
          batch_size=batch_size,
          validation_data = (validx, validy),
#          validation_batch_size = batch_size*2,
          callbacks = callbacks
)
"""

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
