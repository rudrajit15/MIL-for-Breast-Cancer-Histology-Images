# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:09:25 2018

@author: Rudrajit Das
"""

from __future__ import print_function

import numpy as np
import os
import math
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Lambda, Flatten, Conv2D, MaxPooling2D, Dropout, merge, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.optimizers import SGD,Adam
#from keras import backend as K
from keras import metrics
import scipy as sc
import tensorflow as tf
import scipy.io as sio
#from sklearn.utils import class_weight

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

batch_size = 4
epochs = 8

centering_mean = False
centering_std = False
img_rows, img_cols = 128,128
img_channels = 1
rescale_fact = 1.0
train_samples = 9600#4800
num_classes = 8#94

train_mat = sio.loadmat('train.mat')
#train_mat = sio.loadmat('mini_test2.mat')

x_train_all = train_mat['x_train']
x_train_all = x_train_all.reshape((train_samples, 128, 128, 1))
#x_train_all = train_mat['x_test']
#x_train_all = x_train_all.reshape((48, 128, 128, 1))

x_test = x_train_all[train_samples-96:train_samples,:,:,:]

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def UNet(x):
      
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization(axis = -1)(conv1)
    #conv1 = keras.activations.relu(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(axis = -1)(conv1)
    #drop1 = Dropout(0.2)(conv1)
    drop1 = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization(axis = -1)(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization(axis = -1)(conv2)
    #drop2 = Dropout(0.2)(conv2)
    drop2 = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization(axis = -1)(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization(axis = -1)(conv3)
    #drop3 = Dropout(0.2)(conv3)
    drop3 = conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization(axis = -1)(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization(axis = -1)(conv4)
    #drop4 = Dropout(0.4)(conv4)
    drop4 = conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization(axis = -1)(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization(axis = -1)(conv5)
    #conv5 = BatchNormalization(axis=-1)(conv5)
    #drop5 = Dropout(0.4, name='encoder')(conv5)
    drop5 = conv5
    
    '''
    conv51 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    conv51 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv51)
    #conv5 = BatchNormalization(axis=-1)(conv5)
    #drop5 = Dropout(0.4, name='encoder')(conv5)
    drop51 = conv51
    
    conv52 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop51)
    conv52 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv52)
    #conv5 = BatchNormalization(axis=-1)(conv5)
    #drop5 = Dropout(0.4, name='encoder')(conv5)
    drop52 = conv52
    '''
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)

    #merge6 = keras.layers.Concatenate(axis=3)([drop4, up6])
    merge6 = up6
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization(axis = -1)(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization(axis = -1)(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #merge7 = merge([drop3,up7], mode = 'concat', concat_axis = 3)

    #merge7 = keras.layers.Concatenate(axis=3)([drop3, up7])
    merge7 = up7
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization(axis = -1)(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization(axis = -1)(conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #merge8 = merge([drop2,up8], mode = 'concat', concat_axis = 3)

    #merge8 = keras.layers.Concatenate(axis=3)([drop2, up8])
    merge8 = up8
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization(axis = -1)(conv8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization(axis = -1)(conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = merge([drop1,up9], mode = 'concat', concat_axis = 3)
    
    #merge9 = keras.layers.Concatenate(axis=3)([drop1, up9])
    merge9 = up9
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization(axis = -1)(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization(axis = -1)(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization(axis = -1)(conv9)
    #conv9 = BatchNormalization(axis=-1)(conv9)
    conv10 = Conv2D(num_classes, 1, activation = 'softmax')(conv9)
    
    model = Model(input = inputs, output = conv10)
    model.load_weights('wts_red_tf_big_unconn.h5')
    
    #model2 = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    y = model.predict(x, batch_size = 36, verbose=1)

    return y


y_test = UNet(x_test)
print(y_test.shape)
#y_test = np.reshape(y_test,(train_samples,1024))
#y_test = np.reshape(y_test,(train_samples,3072))
#np.save('test_features.npy',y_test)

sio.savemat('y_pred2.mat', {'y_pred':y_test})
sio.savemat('x_pred2.mat', {'x_pred':x_test})
