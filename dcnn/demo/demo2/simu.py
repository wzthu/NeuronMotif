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




input_bp = 82

batch_size=128




seqInput = Input(shape=(input_bp, 4), name='seqInput')

seq = Conv1D(5, 7)(seqInput)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(5, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(6, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(6, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(1, 3)(seq)
seq = BatchNormalization()(seq)
seq = Activation('sigmoid')(seq)
seq = Flatten()(seq)

model = Model(inputs = [seqInput], outputs = [seq])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#from keras.optimizers import RMSprop
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])


PWM0 = np.loadtxt('PWM')
 
PWM = np.ones((4,input_bp))*0.25
PWM1 = np.zeros((4,5))*0.25
PWM1[1:2,:] = 0.5

print(PWM0.shape)
print(PWM.shape)

def pwm_to_sample(PWM, n = 1000):
    PWM /= PWM.sum(axis=0)
    PWM = PWM.T
    PWM = PWM[::-1,:]
    PWM = PWM[:,::-1]
    sample = np.zeros((n,PWM.shape[0],PWM.shape[1]))
    for i in range(n):
        for j in range(sample.shape[1]):
            sample[i,j,np.random.choice(4,1,p=PWM[j,:])] = 1
    return sample

size = 10000

sp0 = pwm_to_sample(PWM0,n=size)
sp1 = pwm_to_sample(PWM0,n=size)
sp2 = pwm_to_sample(PWM0,n=size)
sp3 = pwm_to_sample(PWM1,n=size)
sp4 = pwm_to_sample(PWM0,n=size)
spp = pwm_to_sample(PWM,n=size)
spn = pwm_to_sample(PWM,n=size)
pos0 = np.random.randint(0,16,size)
pos1 = np.random.randint(44,60,size)
pos2 = np.r_[np.random.randint(0,16,int(size/2)),np.random.randint(46,62,int(size/2))]
pos4 = np.random.randint(17,45,size)
pos3 = np.random.randint(0,76,size)

print(sp0.shape)
print(sp1.shape)
print(spp.shape)


for i in range(size):
    spp[i,pos0[i]:(pos0[i]+PWM0.shape[1]),:] = sp0[i,:,:]
    spp[i,pos1[i]:(pos1[i]+PWM0.shape[1]),:] = sp1[i,:,:] 

for i in range(size):
    spn[i,pos2[i]:(pos2[i]+PWM0.shape[1]),:] = sp2[i,:,:]
    spn[i,pos4[i]:(pos4[i]+PWM0.shape[1]),:] = sp4[i,:,:]
#    spn[i,pos3[i]:(pos3[i]+PWM1.shape[1]),:] = sp3[i,:,:]

sp = np.concatenate([spp,spn],axis=0)

label = np.r_[np.ones(size),np.zeros(size)] 

callbacks=[]
callbacks.append(ModelCheckpoint(filepath='weight.hdf5',save_best_only=True))
callbacks.append(EarlyStopping(patience=15))


history = model.fit(x= sp, y=label, epochs=100,validation_split=0.1,callbacks=callbacks) 


history_dict=history.history
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
    f['sample'] = sp
    f['label'] = label
