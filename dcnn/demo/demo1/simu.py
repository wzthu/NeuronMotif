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




input_bp = 600

batch_size=128




seqInput = Input(shape=(8, 4), name='seqInput')


seq = Conv1D(3, 5)(seqInput)
seq = Activation('relu')(seq)
seq = MaxPooling1D(2)(seq)
seq = Conv1D(1, 2)(seq)
seq = Activation('sigmoid')(seq)
seq = Flatten()(seq)

model = Model(inputs = [seqInput], outputs = [seq])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#from keras.optimizers import RMSprop
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])


PWM0 = np.loadtxt('PWM0') 
PWM1 = np.loadtxt('PWM1')
PWM = np.ones(PWM1.shape)*0.25


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

sp0 = pwm_to_sample(PWM0)
sp1 = pwm_to_sample(PWM1)
spn = pwm_to_sample(PWM,n=2000)

sp = np.concatenate([sp0,sp1,spn],axis=0)

label = np.r_[np.ones(2000),np.zeros(2000)] 

callbacks=[]
callbacks.append(ModelCheckpoint(filepath='weight.hdf5',save_best_only=True))
callbacks.append(EarlyStopping(patience=15))


history = model.fit(x= sp, y=label, epochs=100,validation_split=0.1,callbacks=callbacks) 

print(model.layers[1].get_weights()[0][:,:,0])
print(model.layers[1].get_weights()[0][:,:,1])
print(model.layers[1].get_weights()[0][:,:,2])

print(model.layers[4].get_weights()[0][:,:,0] )

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
