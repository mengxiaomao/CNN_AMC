# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:16:53 2017
Training of N=500/incoherent
@author: Fan Meng, Southeast University
email: mengxiaomaomao@outlook.com
"""
import os
import numpy as np
import scipy.io as sc 
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense
from keras.layers.core import Flatten
from keras.layers import Conv1D, AveragePooling1D
from sklearn.utils import shuffle

signal_size=500
BATCH_SIZE=128
L_1=128
class_name=['2PSK','4PSK','8PSK','16QAM']
data_file = os.getcwd() #data_file: generated from corresponding matlab codes
base_train_filepath_step1 = data_file+'/train_data/train_N500_inco_step1.mat'
base_train_filepath_step2 = data_file+'/train_data/train_N500_inco_step2.mat'
best_weights_filepath_step1 = data_file+'/weight_data/base_weight_N500.hdf5'
best_weights_filepath_step2 = data_file+'/weight_data/best_weight_N500_ultimate.hdf5'
base_test_filepath = data_file+'/test_data/test_N500_'

# step 1
print('\ntraining process step1...') 
class_num=5
EPOCH=20

y = Input(shape=(signal_size,2), dtype='float32', name='Input')

x = Conv1D(12, 3, padding='same', activation='relu')(y)
x = Conv1D(12, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(24, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(24, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)

snr = Input(shape=(1,), dtype='float32', name='snr')
n = Dense(10, activation='relu')(snr)
m = keras.layers.concatenate([x, n], axis=-1)
f = Dense(L_1, activation='relu')(m)
p = Dense(class_num, activation='softmax', name='p')(f)

model = Model(inputs=[y, snr], outputs=[p])
model.compile(optimizer='adam', loss={'p': 'categorical_crossentropy'}, metrics=['accuracy'])
saveBestModel = ModelCheckpoint(best_weights_filepath_step1, 
                                            monitor='val_loss', 
                                            verbose=0, 
                                            save_best_only=True, 
                                            mode='auto')

train_data=np.array(sc.loadmat(base_train_filepath_step1)['train_data'],np.float32)
train_data=shuffle(train_data) 
signal_data=train_data[:,:signal_size*2]
snr_data=train_data[:,signal_size*2]
label_data=train_data[:,signal_size*2+1:]
signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
train_data=[]
hist_step1=model.fit([signal_data, snr_data], label_data, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCH, 
          verbose=1,
          callbacks=[saveBestModel],
          validation_split=0.1)
history_step1=hist_step1.history
model.load_weights(best_weights_filepath_step1)
print('\ntraining process step1 is finished.')

# step 2
class_num=4
EPOCH=240
print('\ntraining process step2...') 

p = Dense(class_num, activation='softmax', name='p')(f)

model = Model(inputs=[y, snr], outputs=[p])
model.compile(optimizer='adam', loss={'p': 'categorical_crossentropy'}, metrics=['accuracy'])

train_data=np.array(sc.loadmat(base_train_filepath_step2)['train_data'],np.float32)
train_data=shuffle(train_data) 
signal_data=train_data[:,:signal_size*2]
snr_data=train_data[:,signal_size*2]
label_data=train_data[:,signal_size*2+1:]
signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
train_data=[]

saveBestModel = ModelCheckpoint(best_weights_filepath_step2, 
                                            monitor='val_loss', 
                                            verbose=0, 
                                            save_best_only=True, 
                                            mode='auto')
hist_step2=model.fit([signal_data, snr_data], label_data, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCH, 
          verbose=1,
          callbacks=[saveBestModel],
          validation_split=0.1)
history_step2=hist_step2.history
print('\ntraining process step2 is finished.')

# testing
model.load_weights(best_weights_filepath_step2)
print('\ntesting...')
SNR=[-6,-4,-2,0,2,4,6]
Pc=[]
k=0
p_c=np.zeros((len(SNR),class_num))
coff=1  # N/500 
for snr in SNR:
    train_data=sc.loadmat(base_test_filepath+str(snr)+'dB_inco.mat')['train_data']
    signal_data=train_data[:,:signal_size*2]
    snr_data=train_data[:,signal_size*2]
    label_data=train_data[:,signal_size*2+1:]
    signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
    cou=int(signal_data.shape[0]/coff)
    proba=model.predict([signal_data, snr_data])
    proba1=np.ones((cou,class_num))
    for u in range(coff):
        proba1*=proba[u:signal_data.shape[0]:coff,:]
    proba=proba1
    Ylabel=label_data[0:signal_data.shape[0]:coff,:].argmax(axis=-1)
    Ypred=proba.argmax(axis=-1)
    indx=[]
    for i in range(cou):
        if Ylabel[i]!=Ypred[i]:
            indx.append(i)
        else:
            p_c[k][Ypred[i]]+=1
    p_c[k,:]=p_c[k,:]/cou*class_num
    pc=1.0-1.0*len(indx)/cou
    Pc.append(pc)
    k+=1
    print("SNR: %.1f, Pc: %.3f"%(snr,pc))
