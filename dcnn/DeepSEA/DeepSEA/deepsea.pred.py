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


input_bp = 1000
conv_kernel_size = 8
pool_kernel_size = 4

maxnrom = MaxNorm(max_value=0.9, axis=0)
l1l2 = l1_l2(l1=0, l2=1e-6)

def crelu(x, alpha=0.0, max_value=None, threshold=1e-6):
    return relu(x, alpha, max_value, threshold)

batch_size=1000

seqInput = Input(shape=(input_bp, 4), name='seqInput')

seq = Conv1D(320, conv_kernel_size,kernel_regularizer=l1l2, kernel_constraint=maxnrom)(seqInput)
seq = Activation(crelu)(seq)
seq = MaxPooling1D(pool_size=pool_kernel_size,strides=pool_kernel_size)(seq)
seq = Dropout(0.2)(seq)
seq = Conv1D(480, conv_kernel_size,kernel_regularizer=l1l2, kernel_constraint=maxnrom)(seq)
seq = Activation(crelu)(seq)
seq = MaxPooling1D(pool_size=pool_kernel_size,strides=pool_kernel_size)(seq)
seq = Dropout(0.2)(seq)
seq = Conv1D(960, conv_kernel_size,kernel_regularizer=l1l2, kernel_constraint=maxnrom)(seq)
seq = Activation(crelu)(seq)
seq = Dropout(0.5)(seq)
seq = Flatten()(seq)
seq = Dense(925,kernel_regularizer=l1l2, kernel_constraint=maxnrom)(seq)
seq = Activation(crelu)(seq)
seq = Dense(919,kernel_regularizer=l1l2, kernel_constraint=maxnrom, activity_regularizer=l1_l2(l1=1e-8,l2=0))(seq)
seq = Activation('sigmoid')(seq)


model = Model(inputs = [seqInput], outputs = [seq])
model.load_weights('weight.hdf5')

from scipy.io import loadmat



#trainmat = loadmat('data/train.mat')
testmat = loadmat('data/test.mat')
#trainy = trainmat['traindata']
#trianx = trainmat['trainxdata']
testx = testmat['testxdata']
testy = testmat['testdata']




testidx = testx.transpose((0,2,1))
#validy = validy.transpose((1,0))
result = model.predict(testidx,batch_size=1000)


from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
auprc = np.zeros((testy.shape[1]))
auroc = np.zeros((testy.shape[1]))
for i in range(testy.shape[1]):
    print(i)
    if testy[:,i].sum() !=0 and testy[:,i].sum() !=testy.shape[0]:
        auprc[i] = average_precision_score(testy[:,i],result[:,i])
        auroc[i] = roc_auc_score(testy[:,i],result[:,i])


with h5py.File('test.h5','w') as f:
    f['test_result'] = result
    f['test_label'] = testy
    f['auprc'] = auprc
    f['auroc']=auroc

