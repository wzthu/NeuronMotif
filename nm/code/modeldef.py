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
        seq = Conv1D(256, 3)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(384, 3)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = Conv1D(384, 3)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(512, 3)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = Conv1D(512, 3)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = Flatten()(seq)
        seq = Dropout(0.2)(seq)
        seq = Dense(768)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('relu')(seq)
        seq = Dense(164)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation('sigmoid')(seq)
        return Model(inputs = [seqInput], outputs = [seq])
#
##################################################

def get_model_list(layer, kernel, weight_file='weight.hdf5'):
    model = get_whole_model()
    model.load_weights(weight_file)
####################################################
# Fill kernel_nb, kernel_sz and pool_sz according your model
    kernel_nb = [128,128,160,160,256,256,384,384,512,512]
    kernel_sz = [7,3,3,3,3,3,3,3,3,3]
    pool_sz = [1,2,1,2,1,2,1,2,1,2]
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
    seq = Conv1D(128, 7)(seqInput)
    act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    seq = BatchNormalization()(seq)
    model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 1:
        seq = Activation('relu')(seq)
        seq = Conv1D(128, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 2:
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(160, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 3:
        seq = Activation('relu')(seq)
        seq = Conv1D(160, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 4:
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(256, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 5:
        seq = Activation('relu')(seq)
        seq = Conv1D(256, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 6:
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(384, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 7:
        seq = Activation('relu')(seq)
        seq = Conv1D(384, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 8:
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(2)(seq)
        seq = Conv1D(512, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
        model_list.append(Model(inputs = [seqInput], outputs = [seq]))
    if layer > 9:
        seq = Activation('relu')(seq)
        seq = Conv1D(512, 3)(seq)
        act_model_list.append(Model(inputs = [seqInput], outputs = [seq]))
        seq = BatchNormalization()(seq)
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


