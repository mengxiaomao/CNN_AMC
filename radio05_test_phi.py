# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:08:06 2017
testing module of radio_recognition05/ phi tesing
@author: mengxiaomao
"""
import numpy as np
import scipy.io as sc 

from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers.core import Flatten
from keras.layers import Conv1D, AveragePooling1D

signal_size=1000
L_1=128
class_num=7
class_name=['2PSK','4PSK','8PSK','16QAM','16APSK','32APSK','64QAM']

model_s = Sequential()  
model_s.add(Conv1D(12, 3, padding='same', activation='relu', input_shape=(signal_size,2)))
model_s.add(Conv1D(12, 3, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(24, 3, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(24, 3, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Conv1D(32, 3, padding='same', activation='relu'))
model_s.add(AveragePooling1D(pool_size=2))
model_s.add(Flatten())
model_s.add(Dense(256, activation='relu'))

model_c = Sequential() 
model_c.add(Dense(10, activation='relu', input_dim=1))

model = Sequential() 
model.add(Merge([model_s, model_c], mode='concat'))
model.add(Dense(L_1, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

best_weights_filepath = 'C:/Software/workshop/python/weights/best_weights_N1000_ultimate.hdf5'
model.load_weights(best_weights_filepath)

print('\ntesting...')
base_test_filepath = 'C:/Software/workshop/python/Datasets/radio_recognition/test_data_'
SNR=[-6,-4,-2,0,2,4,6,8,10]
PHI=[3,6,9,12,15]
Pc=[]

p_c=np.zeros((len(PHI),len(SNR),class_num))
k=0
for phi in PHI: 
    kk=0
    for snr in SNR:
        train_data=sc.loadmat(base_test_filepath+str(snr)+'dB_'+str(phi)+'.mat')['train_data']
        signal_data=train_data[:,:signal_size*2]
        snr_data=train_data[:,signal_size*2]
        label_data=train_data[:,signal_size*2+1:]
        label_data=label_data[:,:class_num]
        signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
        proba=model.predict([signal_data, snr_data])
        Ylabel=label_data.argmax(axis=-1)
        Ypred=proba.argmax(axis=-1)
        indx=[]
        for i in range(signal_data.shape[0]):
            if Ylabel[i]!=Ypred[i]:
                indx.append(i)
            else:
                p_c[k][kk][Ypred[i]]+=1
        p_c[k,kk,:]=p_c[k,kk,:]/signal_data.shape[0]*class_num
        pc=1.0-1.0*len(indx)/signal_data.shape[0]
        Pc.append(pc)
        kk+=1
        print('\n'+'SNR: '+str(snr)+'dB')
        print(classification_report(Ylabel, Ypred, target_names=class_name))
    k+=1
