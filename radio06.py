# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:52:30 2017
Training of N=500
@author: mengxiaomao
"""
import h5py
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers.core import Flatten
from keras.layers import Conv1D, AveragePooling1D,MaxPooling1D
from sklearn.utils import shuffle

signal_size=500
BATCH_SIZE=128
L_1=128
class_name=['2PSK','4PSK','8PSK','16QAM','16APSK','32APSK','64QAM']
weight_file='C:/Software/workshop/python/weights/'
data_file='C:/Software/workshop/python/Datasets/radio_recognition/'
base_train_filepath_step1 = data_file+'train_data_N500p_step1.mat'
base_train_filepath_step2 = data_file+'train_data_N500p_step2.mat'
best_weights_filepath_step1 = weight_file+'base_weights_N500p.hdf5'
best_weights_filepath_step2 = weight_file+'best_weights_N500p_ultimate.hdf5'

# step 1
print('\ntraining process step1...') 
class_num=8
EPOCH=10

model_s = Sequential()  
model_s.add(Conv1D(12, 6, padding='same', activation='relu', input_shape=(signal_size,2)))
model_s.add(Conv1D(12, 6, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(24, 6, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(24, 4, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(32, 4, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Flatten())
model_s.add(Dense(256, activation='relu'))

model_c = Sequential() 
model_c.add(Dense(10, activation='relu', input_dim=1))

model = Sequential() 
model.add(Merge([model_s, model_c], mode='concat'))
model.add(Dense(L_1, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])#adam
saveBestModel = ModelCheckpoint(best_weights_filepath_step1, 
                                            monitor='val_loss', 
                                            verbose=0, 
                                            save_best_only=True, 
                                            mode='auto')

train_data=np.array(h5py.File(base_train_filepath_step1)['train_data'],np.float16).T
train_data=shuffle(train_data) 
signal_data=train_data[:,:signal_size*2]
snr_data=train_data[:,signal_size*2]
label_data=train_data[:,signal_size*2+1:]
label_data=label_data[:,:class_num]
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
class_num=7
EPOCH=120
print('\ntraining process step2...') 

model.pop()
top_model = Sequential()
top_model.add(Dense(class_num, activation='softmax', input_dim=L_1))
model.add(top_model)

train_data=np.array(h5py.File(base_train_filepath_step2)['train_data'],np.float16).T
#train_data[:,signal_size*2+1:]=np.kron(np.eye(class_num),np.ones((int(train_data.shape[0]/class_num),1)))
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
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
hist_step2=model.fit([signal_data, snr_data], label_data, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCH, 
          verbose=1,
          callbacks=[saveBestModel],
          validation_split=0.1)
history_step2=hist_step2.history
model.load_weights(best_weights_filepath_step2)
print('\ntraining process step2 is finished.')
