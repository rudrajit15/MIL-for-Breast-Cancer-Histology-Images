# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:56:07 2018

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
epochs = 10

centering_mean = False
centering_std = False
img_rows, img_cols = 256,256
img_channels = 1
rescale_fact = 1.0
train_samples = 30#89#170
num_classes = 10#94#107

train_mat = sio.loadmat('x_all_R.mat')

x_train_all = train_mat['x_all_R']
x_train_all = x_train_all.reshape((train_samples, 256, 256, img_channels))
#y_train_all = train_mat['y_train']
#y_train_all = y_train_all.reshape((train_samples, 256, 256, 1))

x_train = x_train_all#[0:train_samples-5,:,:,:]
#y_train = y_train_all[0:train_samples-5,:,:]

#val_mat = sio.loadmat('val2.mat')
#x_val = val_mat['x_val']
#x_val = x_val.reshape((train_samples, 256, 256, img_channels))
#x_val = x_train_all[train_samples-5:train_samples,:,:,:]
#y_val = y_train_all[train_samples-5:train_samples,:,:]

def SSIM_objective(y_true, y_pred):
    ssim_val = 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    return ssim_val

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def UNet(x_train):
      
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(inputs)
    conv1 = BatchNormalization(axis = -1, name = 'conv1_2')(conv1)
    #conv1 = keras.activations.relu(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_3')(conv1)
    conv1 = BatchNormalization(axis = -1, name = 'conv1_4')(conv1)
    #drop1 = Dropout(0.2)(conv1)
    drop1 = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = BatchNormalization(axis = -1, name = 'conv2_2')(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_3')(conv2)
    conv2 = BatchNormalization(axis = -1, name = 'conv2_4')(conv2)
    #drop2 = Dropout(0.2)(conv2)
    drop2 = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = BatchNormalization(axis = -1, name = 'conv3_2')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_3')(conv3)
    conv3 = BatchNormalization(axis = -1, name = 'conv3_4')(conv3)
    #drop3 = Dropout(0.2)(conv3)
    drop3 = conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_1')(pool3)
    conv4 = BatchNormalization(axis = -1, name = 'conv4_2')(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_3')(conv4)
    conv4 = BatchNormalization(axis = -1, name = 'conv4_4')(conv4)
    #drop4 = Dropout(0.4)(conv4)
    drop4 = conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_1')(pool4)
    conv5 = BatchNormalization(axis = -1, name = 'conv5_2')(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_3')(conv5)
    conv5 = BatchNormalization(axis = -1, name = 'conv5_4')(conv5)
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
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_1')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)

    merge6 = keras.layers.Concatenate(axis=3)([drop4, up6])
    #merge6 = up6
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_2')(merge6)
    conv6 = BatchNormalization(axis = -1, name = 'conv6_3')(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_4')(conv6)
    conv6 = BatchNormalization(axis = -1, name = 'conv6_5')(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_1')(UpSampling2D(size = (2,2))(conv6))
    #merge7 = merge([drop3,up7], mode = 'concat', concat_axis = 3)

    merge7 = keras.layers.Concatenate(axis=3)([drop3, up7])
    #merge7 = up7
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_2')(merge7)
    conv7 = BatchNormalization(axis = -1, name = 'conv7_3')(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_4')(conv7)
    conv7 = BatchNormalization(axis = -1, name = 'conv7_5')(conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(UpSampling2D(size = (2,2))(conv7))
    #merge8 = merge([drop2,up8], mode = 'concat', concat_axis = 3)

    merge8 = keras.layers.Concatenate(axis=3)([drop2, up8])
    #merge8 = up8
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_2')(merge8)
    conv8 = BatchNormalization(axis = -1, name = 'conv8_3')(conv8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_4')(conv8)
    conv8 = BatchNormalization(axis = -1, name = 'conv8_5')(conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_1')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = merge([drop1,up9], mode = 'concat', concat_axis = 3)
    
    merge9 = keras.layers.Concatenate(axis=3)([drop1, up9])
    #merge9 = up9
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
    conv9 = BatchNormalization(axis = -1, name = 'conv9_3')(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_4')(conv9)
    conv9 = BatchNormalization(axis = -1, name = 'conv9_5')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_6')(conv9)
    conv9 = BatchNormalization(axis = -1, name = 'conv9_7')(conv9)
    #conv9 = BatchNormalization(axis=-1)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid', name = 'conv10_1')(conv9)
    
    model = Model(input = inputs, output = conv10)
    
    model.load_weights('wts_ssim_small.h5')
    
    model.summary()
    
    #model2 = Model(inputs=inputs, outputs = drop5)
    #model2 = Model(inputs=model.input, outputs = model.get_layer('encoder').output)
    
    #model.compile(optimizer = Adam(lr = 5*1e-5), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    #model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.compile(loss = SSIM_objective, optimizer = 'adam')#, metrics = ['accuracy'])
    
    y = model.predict(x_train, batch_size = 2, verbose=1)
    
    return y


y_test = UNet(x_train_all)
print(y_test.shape)
#y_test = np.reshape(y_test,(train_samples,1024))
#y_test = np.reshape(y_test,(train_samples,3072))
np.save('y_ssim_small.npy',y_test)
np.save('x_ssim_small.npy',x_train_all)

sio.savemat('y_pred_small.mat', {'y_pred':y_test})
sio.savemat('x_pred_small.mat', {'x_pred':x_train_all})