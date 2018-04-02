# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:56:11 2018
tensorflow Training of N=500/coherent 
cumulants+two-hidden layer NN
using 2,4,6-order cumulants
@author: mengxiaomao
"""
import os
import time
import numpy as np
import tensorflow as tf
import scipy.io as sc  
from sklearn.utils import shuffle

reuse=tf.AUTO_REUSE
dtype=np.float32
epochs=200
batch_size=1000
signal_size=21
validation_split=0.1
class_num=7
class_name=['2PSK','4PSK','8PSK','16QAM','16APSK','32APSK','64QAM']
data_file = os.getcwd() #data_file: generated from corresponding matlab codes
best_weights_filepath = data_file+'/weight_data/best_weight_N1000_cumulant.mat'

def Read_test_data(filename):
    train_data=np.array(sc.loadmat(filename)['train_data'],dtype)
    signal_data=train_data[:,:signal_size]
    signal_data=np.concatenate((signal_data[:,0:9],signal_data[:,10:19],signal_data[:,20:21]),axis=1)
    label_data=train_data[:,signal_size:]
    return signal_data,label_data
    
def Read_train_data(filename):
    train_data=np.array(sc.loadmat(filename)['train_data'],dtype)
    train_data=shuffle(train_data)
    train_size=train_data.shape[0]
    signal_data=train_data[:,:signal_size]
    signal_data=np.concatenate((signal_data[:,0:9],signal_data[:,10:19],signal_data[:,20:21]),axis=1)
    label_data=train_data[:,signal_size:]
    signal_valid=signal_data[:int(validation_split*train_size)][:]
    label_valid=label_data[:int(validation_split*train_size)][:]
    label_train=label_data[int(validation_split*train_size):][:]
    signal_train=signal_data[int(validation_split*train_size):][:]
    return signal_train,label_train,signal_valid,label_valid
    
def Variable(shape):  
    weight=tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias=tf.get_variable('b', shape=[shape[-1]], initializer=tf.zeros_initializer)
    return weight, bias   

def Network(x):
    with tf.variable_scope('l0', reuse=reuse):
        w,b=Variable([19,40])
        l=tf.matmul(x,w)+b
        l=tf.nn.tanh(l)
    with tf.variable_scope('l1', reuse=reuse):
        w,b=Variable([40,40])
        l=tf.matmul(l,w)+b
        l=tf.nn.tanh(l)
    with tf.variable_scope('l2', reuse=reuse):
        w,b=Variable([40,class_num])
        l=tf.matmul(l,w)+b
        p=tf.nn.softmax(l)
    return p      
    
def Network_ini(theta):
    update=[]
    for var in tf.trainable_variables():
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
    return update
    
def Loss(y,y_):
#    cost=tf.reduce_mean(tf.square(y_ - y))
    cost=tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    return cost
    
def Optimizer(cost):
    with tf.variable_scope('opt', reuse=reuse):
        train_op=tf.train.AdamOptimizer().minimize(cost)
    return train_op  
    
def Accuracy(y,y_):
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype))
    return accuracy
    
def Get_prob(y):
    return y
    
def Train_batch(sess, signal_data, label_data):
    train_values = []
    train_acc = []
#    tf.train.shuffle_batch()
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        _, loss_val, acc_val = sess.run([optimizer,cost,acc], feed_dict={x: batch_x, p_: batch_p_})
        train_values.append(loss_val)
        train_acc.append(acc_val)
    return np.mean(train_values)/batch_size, np.mean(train_acc)
    
def Valid_batch(sess, signal_data, label_data):
    val_values = []
    valid_acc = []
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        loss_val, acc_val = sess.run([cost,acc], feed_dict={x: batch_x, p_: batch_p_})
        val_values.append(loss_val)
        valid_acc.append(acc_val)
    return np.mean(val_values)/batch_size, np.mean(valid_acc)
    
def Test_batch(sess, signal_data,label_data):
    test_values = []
    test_acc = []
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        loss_test, acc_test = sess.run([cost,acc], feed_dict={x: batch_x, p_: batch_p_})
        test_values.append(loss_test)
        test_acc.append(acc_test)
    return np.mean(test_values)/batch_size, np.mean(test_acc)

def Train(filepath):
    signal_train,label_train,signal_valid,label_valid=Read_train_data(filepath)
    print('\ntraining process step...')     
    with tf.Session() as sess:        
        tf.global_variables_initializer().run()    
        best_val_loss, val_acc = Valid_batch(sess, signal_valid,label_valid)
        Save(best_weights_filepath)
        print("Initial session model saved in file: %s, Validation Loss: %.3f, Validation Acc: %.3f" %(best_weights_filepath,best_val_loss,val_acc))     
        for i in range(epochs):        
            start_time = time.time()          
            train_loss, train_acc = Train_batch(sess, signal_train,label_train)
            valid_loss, valid_acc = Valid_batch(sess, signal_valid,label_valid)
            time_taken = time.time() - start_time
            print("Epoch %d Training Loss: %.3f, Training Acc: %.3f, Validation Loss: %.3f, Validation Acc: %.3f, Time Cost: %.2f s"
                  % (i+1,train_loss,train_acc,valid_loss,valid_acc,time_taken))
            if(valid_loss < best_val_loss):
                best_val_loss = valid_loss
                Save(best_weights_filepath)
        print("\nTraining is finished.") 

def Test(filepath):
    signal_data,label_data=Read_test_data(filepath)
    with tf.Session() as sess: 
        tf.global_variables_initializer().run()       
        sess.run(update)
        start_time = time.time()
        loss_test, acc_test = Test_batch(sess, signal_data, label_data)
        time_taken = time.time() - start_time
        print("Testing Loss: %.3f, Testing Acc: %.3f, Time Cost: %.2f s"% (loss_test, acc_test,time_taken))
    return acc_test
  
def Test_Rayleigh(filepath, coff):
    signal_data,label_data=Read_test_data(filepath)
    with tf.Session() as sess: 
        tf.global_variables_initializer().run()       
        sess.run(update)
        prob=np.zeros(label_data.shape)
        for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
            batch_x = signal_data[offset:offset + batch_size][:]
            batch_p_ = label_data[offset:offset + batch_size][:]
            prob_test = sess.run([p], feed_dict={x: batch_x, p_: batch_p_})
            prob[offset:offset + batch_size][:] = np.array(prob_test).reshape(-1,class_num)
        cou=signal_data.shape[0]//coff
        acc_test=np.ones((cou,class_num))
        for u in range(coff):
            acc_test*=prob[u:signal_data.shape[0]:coff,:]
        Ylabel=label_data[0:signal_data.shape[0]:coff,:].argmax(axis=-1)
        Ypred=acc_test.argmax(axis=-1)
        indx=[]
        for i in range(cou):
            if Ylabel[i]==Ypred[i]:
                indx.append(i)
        pc=1.0*len(indx)/cou
    return pc
    
def Save(filepath):
    dict_name={}
    for var in tf.trainable_variables():  
        dict_name[var.name]=var.eval()
    sc.savemat(filepath, dict_name) 
        
def Read():
    for var in tf.trainable_variables():
        print(var.name,var.shape)
       
if __name__ == '__main__':
    graph = tf.Graph()       
    with graph.as_default() as g:  
        with tf.Session(graph=g): 
            x=tf.placeholder(dtype,shape=[None,19])
            p_=tf.placeholder(dtype,shape=[None,class_num])
            p=Network(x)   
            cost=Loss(p,p_)
            optimizer=Optimizer(cost)
            acc=Accuracy(p,p_)
            train_filepath = data_file+'/train_data/train_N1000_co_cumulant'             
            Train(train_filepath)
            
#    #test
    SNR_test=[-6,-4,-2,0,2,4,6,8,10]
#    SNR_test=[-6,-4,-2,0,2,4,6]
    Pc=np.zeros((len(SNR_test)))
    for k1 in range(len(SNR_test)):
        tf.reset_default_graph() 
        graph = tf.Graph()       
        with graph.as_default() as g:  
            with tf.Session(graph=g):  
                x=tf.placeholder(dtype,shape=[None,19])
                p_=tf.placeholder(dtype,shape=[None,class_num])
                p=Network(x)     
                update=Network_ini(sc.loadmat(best_weights_filepath))
                cost=Loss(p,p_)
                acc=Accuracy(p,p_)     
                test_filepath=data_file+'/test_data/test_N1000_'+str(SNR_test[k1])+'dB_cumulant.mat'
                Pc[k1]=Test(test_filepath)
            
    # fading test
#    coff=1
#    SNR_test=[-18,-14,-10,-6,-2,2,6,10,14,18]
#    Pc=np.zeros((len(SNR_test)))
#    for k1 in range(len(SNR_test)):
#        tf.reset_default_graph() 
#        graph = tf.Graph()       
#        with graph.as_default() as g:  
#            with tf.Session(graph=g):  
#                x=tf.placeholder(dtype,shape=[None,19])
#                p_=tf.placeholder(dtype,shape=[None,class_num])
#                p=Network(x)     
#                update=Network_ini(sc.loadmat(best_weights_filepath))
#                cost=Loss(p,p_)
#                acc=Accuracy(p,p_) 
#                test_filepath=data_file+'test_'+str(SNR_test[k1])+'dB_2.mat' 
#                Pc[k1]=Test_Rayleigh(test_filepath,coff)          
            