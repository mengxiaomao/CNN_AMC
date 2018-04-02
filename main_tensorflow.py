# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:51:54 2018
tensorflow Training of N=500/coherent
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
kernel_size=3
batch_size=1000
signal_size=1000
L_1=128
validation_split=0.1
#lr=0.001
class_name=['2PSK','4PSK','8PSK','16QAM']
data_file = os.getcwd() #data_file: generated from corresponding matlab codes
best_weights_filepath_step1 = data_file+'/weight_data/base_weights_N500_co_cnn.mat'
best_weights_filepath_step2 = data_file+'/weight_data/best_weights_N500_co_cnn_ultimate.mat'

base_train_filepath_step1 = data_file+'/train_data/train_N1000_co_step1.mat'
base_train_filepath_step2 = data_file+'/train_data/train_N1000_co_step2.mat'
base_test_filepath = base_train_filepath_step2

def Read_test_data(filename):
    train_data=np.array(sc.loadmat(filename)['train_data'],dtype)
    signal_data=train_data[:,:signal_size*2]
    snr_data=train_data[:,signal_size*2]
    label_data=train_data[:,signal_size*2+1:]
    signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
    snr_data=snr_data.reshape(snr_data.shape[0], 1)
    return signal_data,snr_data,label_data
    
def Read_train_data(filename):
    train_data=np.array(sc.loadmat(filename)['train_data'],dtype) #sc.loadmat(filename)
    train_data=shuffle(train_data)
    train_size=train_data.shape[0]
    signal_data=train_data[:,:signal_size*2]
    snr_data=train_data[:,signal_size*2]
    label_data=train_data[:,signal_size*2+1:]
    signal_data=signal_data.reshape(signal_data.shape[0], signal_size, 2)
    snr_data=snr_data.reshape(snr_data.shape[0], 1)
    signal_valid=signal_data[:int(validation_split*train_size)][:]
    valid_snr=snr_data[:int(validation_split*train_size)][:]
    label_valid=label_data[:int(validation_split*train_size)][:]
    signal_train=signal_data[int(validation_split*train_size):][:]
    snr_train=snr_data[int(validation_split*train_size):][:]
    label_train=label_data[int(validation_split*train_size):][:]
    return signal_train,snr_train,label_train,signal_valid,valid_snr,label_valid
    
def Conv(x,w,b):
    l=tf.nn.conv1d(x, w, 1,padding='SAME')+b
    return l

def Pool(x):
    l = tf.layers.average_pooling1d(x, pool_size=2, strides=2)
    return l
    
def Variable(shape):  
    weight=tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias=tf.get_variable('b', shape=[shape[-1]], initializer=tf.zeros_initializer)
    return weight, bias   

def Network_l(x, snr):
    with tf.variable_scope('l0', reuse=reuse):
        w,b=Variable([kernel_size,2,12])
        l=Conv(x,w,b)
        l=tf.nn.relu(l)
    with tf.variable_scope('l1', reuse=reuse):
        w,b=Variable([kernel_size,12,12])
        l=Conv(l,w,b)
        l=tf.nn.relu(l)
        l=Pool(l)
    with tf.variable_scope('l2', reuse=reuse):
        w,b=Variable([kernel_size,12,24])
        l=Conv(l,w,b)
        l=tf.nn.relu(l)
        l=Pool(l)
    with tf.variable_scope('l3', reuse=reuse):
        w,b=Variable([kernel_size,24,24])
        l=Conv(l,w,b)
        l=tf.nn.relu(l)
        l=Pool(l)
    with tf.variable_scope('l4', reuse=reuse):
        w,b=Variable([kernel_size,24,32])
        l=Conv(l,w,b)
        l=tf.nn.relu(l)
        l=Pool(l)
    with tf.variable_scope('l5', reuse=reuse):
        l=tf.layers.flatten(l)
        w,b=Variable([l.shape[1],256])
        l=tf.matmul(l,w)+b
        l=tf.nn.relu(l)
    with tf.variable_scope('s0', reuse=reuse):
        w,b=Variable([1,10])
        n=tf.matmul(snr,w)+b
        n=tf.nn.relu(n)
    m=tf.concat([l, n],1)
    with tf.variable_scope('f', reuse=reuse):
        w,b=Variable([m.shape[1],L_1])
        f=tf.matmul(m,w)+b
        f=tf.nn.relu(f)
    return f
    
def Network_h(f, class_num):
    with tf.variable_scope('p', reuse=reuse):
        w,b=Variable([L_1,class_num])
        p=tf.matmul(f,w)+b
        p=tf.nn.softmax(p) 
    return p        
    
def Network_ini(theta):
    update=[]
    for var in tf.trainable_variables():
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
    return update
    
def Loss(y,y_):
    cost=tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
#    cost=tf.reduce_mean(tf.square(y_ - y))
#    cost=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    return cost
    
def Optimizer(cost):
    with tf.variable_scope('opt', reuse=reuse):
        train_op=tf.train.AdamOptimizer().minimize(cost)
    return train_op  
    
def Accuracy(y,y_):
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype))
    return accuracy
    
def Train_batch(sess, signal_data, snr_data, label_data):
    train_values = []
    train_acc = []
    signal_data,snr_data,label_data = shuffle(signal_data,snr_data,label_data)
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_snr = snr_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        _, loss_val, acc_val = sess.run([optimizer,cost,acc], feed_dict={x: batch_x, snr: batch_snr, p_: batch_p_})
        train_values.append(loss_val)
        train_acc.append(acc_val)
    return np.mean(train_values), np.mean(train_acc)
    
def Valid_batch(sess, signal_data, snr_data, label_data):
    val_values = []
    valid_acc = []
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_snr = snr_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        loss_val, acc_val = sess.run([cost,acc], feed_dict={x: batch_x, snr: batch_snr, p_: batch_p_})
        val_values.append(loss_val)
        valid_acc.append(acc_val)
    return np.mean(val_values), np.mean(valid_acc)
    
def Test_batch(sess, signal_data,snr_data,label_data):
    test_values = []
    test_acc = []
    for offset in range(0, signal_data.shape[0]-batch_size+1, batch_size):
        batch_x = signal_data[offset:offset + batch_size][:]
        batch_snr = snr_data[offset:offset + batch_size][:]
        batch_p_ = label_data[offset:offset + batch_size][:]
        loss_test, acc_test = sess.run([cost,acc], feed_dict={x: batch_x, snr: batch_snr, p_: batch_p_})
        test_values.append(loss_test)
        test_acc.append(acc_test)
    return np.mean(test_values), np.mean(test_acc)

def Train(step):
    if step==1:
        best_weights_filepath=best_weights_filepath_step1
        base_train_filepath=base_train_filepath_step1
    else:
        best_weights_filepath=best_weights_filepath_step2   
        base_train_filepath=base_train_filepath_step2
    signal_train,snr_train,label_train,signal_valid,valid_snr,label_valid=Read_train_data(base_train_filepath)   
    print('\ntraining process step',str(step),'...')     
    with tf.Session() as sess:        
        tf.global_variables_initializer().run()    
        if step!=1:
            sess.run(update)
        best_val_loss, val_acc = Valid_batch(sess, signal_valid,valid_snr,label_valid)
        Save(best_weights_filepath)
        print("Initial session model saved in file: %s, Validation Loss: %.3f, Validation Acc: %.3f" %(best_weights_filepath,best_val_loss,val_acc))     
        for i in range(epochs):        
            start_time = time.time()          
            train_loss, train_acc = Train_batch(sess, signal_train,snr_train,label_train)
            valid_loss, valid_acc = Valid_batch(sess, signal_valid,valid_snr,label_valid)
            time_taken = time.time() - start_time
            print("Epoch %d Training Loss: %.3f, Training Acc: %.3f, Validation Loss: %.3f, Validation Acc: %.3f, Time Cost: %.2f s"
                  % (i+1,train_loss,train_acc,valid_loss,valid_acc,time_taken))
            if(valid_loss < best_val_loss):
                best_val_loss = valid_loss
                Save(best_weights_filepath)
        print("\nTraining",str(step),"is finished.") 

def Test():
    signal_data,snr_data,label_data=Read_test_data(base_test_filepath)
    with tf.Session() as sess: 
        tf.global_variables_initializer().run()       
        sess.run(update)
        start_time = time.time()
        loss_test, acc_test = Test_batch(sess, signal_data,snr_data,label_data)
        time_taken = time.time() - start_time
        print("Testing Loss: %.3f, Testing Acc: %.3f, Time Cost: %.2f s"% (loss_test, acc_test,time_taken))

def Save(filepath):
    dict_name={}
    for var in tf.trainable_variables():  
        dict_name[var.name]=var.eval()
    sc.savemat(filepath, dict_name) 
        
def Read():
    for var in tf.trainable_variables():
        print(var.name,var.shape)
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    
if __name__ == '__main__': 
    #step 1
    class_num=8
    epochs=20  
    step=1
    tf.reset_default_graph()
    graph = tf.Graph()       
    with graph.as_default() as g:  
        with tf.Session(graph=g): 
            x=tf.placeholder(dtype,shape=[None,signal_size,2])
            snr=tf.placeholder(dtype,shape=[None,1])
            p_=tf.placeholder(dtype,shape=[None,class_num])
            f=Network_l(x,snr)
            p=Network_h(f, class_num)
            cost=Loss(p,p_)
            optimizer=Optimizer(cost)
            acc=Accuracy(p,p_)             
            Train(step)
           
#    #step 2    
    class_num=7
    epochs=240   
    step=2
    graph = tf.Graph()       
    with graph.as_default() as g:  
        with tf.Session(graph=g):            
            x=tf.placeholder(dtype,shape=[None,signal_size,2])
            snr=tf.placeholder(dtype,shape=[None,1])
            p_=tf.placeholder(dtype,shape=[None,class_num])
            f=Network_l(x,snr)    
            update=Network_ini(sc.loadmat(best_weights_filepath_step1))
            p=Network_h(f, class_num)            
            cost=Loss(p,p_)
            optimizer=Optimizer(cost)
            acc=Accuracy(p,p_)           
            Train(step)
#            Read()
            
    #test
    class_num=7
    graph = tf.Graph()       
    with graph.as_default() as g:  
        with tf.Session(graph=g):
#            with tf.device("/gpu:0"):
                
            x=tf.placeholder(dtype,shape=[None,signal_size,2])
            snr=tf.placeholder(dtype,shape=[None,1])
            p_=tf.placeholder(dtype,shape=[None,class_num])
            f=Network_l(x,snr)
            p=Network_h(f, class_num)     
            update=Network_ini(sc.loadmat(best_weights_filepath_step2))
            cost=Loss(p,p_)
            acc=Accuracy(p,p_)           
            Test()
               
            