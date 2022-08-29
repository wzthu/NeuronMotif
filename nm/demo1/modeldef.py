import h5py
import sys
import keras
import os

import h5py
import numpy as np
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, MaxPooling1D, BatchNormalization
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model


from keras.activations import relu


import os
import tensorflow as tf

def getRF(layer,kernel_sz, pool_sz):
    if layer == 0:
        return kernel_sz[0]
    sz = 1
    for i in range(layer,0,-1):
        sz += (kernel_sz[i]-1)
        sz *=  pool_sz[i-1]
    sz += (kernel_sz[0]-1)
    return sz



from keras.activations import relu


##################################################
#Fill get_whole_model function with your own model
# We take BD-10 model as an example
input_bp = 600
def get_whole_model():
    seqInput = Input(shape=(input_bp, 4), name='seqInput')
    seq = Conv1D(3, 5)(seqInput)
    seq = Activation('relu')(seq)
    seq = MaxPooling1D(2)(seq)
    seq = Conv1D(1, 2)(seq)
    seq = Activation('sigmoid')(seq)
    seq = Flatten()(seq)
    return Model(inputs = [seqInput], outputs = [seq])
#
##################################################

def get_model_list(layer, kernel, weight_file='weight.hdf5'):
    model = get_whole_model()
    model.load_weights(weight_file)
####################################################
# Fill kernel_nb, kernel_sz and pool_sz according your model
    kernel_nb = [3,1]
    kernel_sz = [5,2]
    pool_sz = [2]
# If there is no max-pooling operation, the pool_sz is 1
#####################################################
    input_bps = [getRF(i,kernel_sz, pool_sz) for i in range(len(kernel_sz))] # [8, 39, 163]
    input_bp = input_bps[layer-1]
    pre_model_list = []
    model_list = []
    act_model_list = []
    out_list = []
####################################################################
# Build substructures for the convolutional neurons in your own model
    seqInput = Input(shape=(input_bp, 4), name='subseqInput')
# Fill your own deep convolutional neural network structure and,
# Before activation function, add  'model_list.append(Model(inputs = [seqInput], outputs = [seq]))'
# After convolution function, add 'act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))'
    seq = Conv1D(3, 5)(seqInput)
    act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 1:
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(1, 2)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
# the convolutional neuron output is seq
####################################################################
    out = seq
    for submodel in model_list:
        for i in range(len(submodel.layers)):
                submodel.layers[i].set_weights(model.layers[i].get_weights())
    for submodel in act_model_list:
        for i in range(len(submodel.layers)):
                submodel.layers[i].set_weights(model.layers[i].get_weights())
    gd = tf.gradients(seq[:,:,kernel],seqInput)
    return kernel_nb,kernel_sz,pool_sz,input_bp,input_bps,model_list, act_model_list, gd


