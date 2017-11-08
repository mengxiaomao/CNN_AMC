# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:40:36 2017

@author: mengxiaomao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:33:35 2017
testing module of radio_recognition06
@author: mengxiaomao
"""
import time
import numpy as np
import scipy.io as sc 

from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers.core import Flatten
from keras.layers import Conv1D, AveragePooling1D,MaxPooling1D

signal_size=500
L_1=128
class_num=7
class_name=['2PSK','4PSK','8PSK','16QAM','16APSK','32APSK','64QAM']

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

best_weights_filepath = 'C:/Software/workshop/python/weights/best_weights_N500_ultimate.hdf5'
model.load_weights(best_weights_filepath)
    
print('\nMerging testing...')
base_test_filepath = 'C:/Software/workshop/python/Datasets/radio_recognition/test_snr/test_data_'
SNR=[0]
DELTA=[0]
Pc=[]
k=0
p_c=np.zeros((len(SNR),class_num))
coff=16
p_c=np.zeros((len(SNR),len(DELTA),class_num))
k=0
for snr in SNR: 
    kk=0
    for delta in DELTA:
        train_data=sc.loadmat(base_test_filepath+str(snr)+'dB_'+str(kk+1)+'.mat')['train_data']
        signal_data=train_data[:,:signal_size*4]
        Len=signal_data.shape[0]*2
        snr_data=train_data[:,signal_size*4]
        snr_data=np.kron(snr_data,np.ones((1,2))).T
        label_data=train_data[:,signal_size*4+1:]
        label_data=np.kron(label_data,np.ones((2,1)))
    
        signal_data=signal_data.reshape(Len, signal_size, 2)
        cou=int(Len/coff)
        start = time.time()
        proba=model.predict([signal_data, snr_data])        
        time_cost=time.time()-start
        proba1=np.ones((cou,class_num))
        for u in range(coff):
            proba1*=proba[u:Len:coff,:]
        proba=proba1
        Ylabel=label_data[0:Len:coff,:].argmax(axis=-1)
        Ypred=proba.argmax(axis=-1)
        indx=[]
        for i in range(cou):
            if Ylabel[i]!=Ypred[i]:
                indx.append(i)
            else:
                p_c[k][kk][Ypred[i]]+=1
        p_c[k,kk,:]=p_c[k,kk,:]/cou*class_num
        pc=1.0-1.0*len(indx)/cou
        Pc.append(pc)
        kk+=1
        print('\n'+'SNR: '+str(snr)+'dB')
        print(classification_report(Ylabel, Ypred, target_names=class_name))
    k+=1