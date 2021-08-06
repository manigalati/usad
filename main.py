#!/usr/bin/env python
# coding: utf-8

######################### parameter

batch_size =  919
#BATCH_SIZE = 100 
n_epochs=  50
hidden_size = 40
window_size=12
# normal_data_path="input/SWaT_Dataset_Normal_v1.csv"
# attack_data_path="input/SWaT_Dataset_Attack_v0.csv"
normal_data_path="../dataset/WADI.A1_9 Oct 2017/WADI_normal_pre_2.csv"
attack_data_path="../dataset/WADI.A1_9 Oct 2017/WADI_Attack_pre.csv"
######################### 

# In[3]:


#from usad.utils import dataPreprocessing, handleData, seq2Window
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
from model import *
import torch.utils.data as data_utils


# In[4]:

# get_ipython().system('nvidia-smi -L')
device = get_default_device()

class execution:
    def __init__(self):
        pass

    def train(self,modelName):

        
        # train_loader 每次iter 取出來的東西是一個windows 攤平成一維的數據
        train_loader,val_loader,test_loader,windows_normal,labels= handleData(normal_data_path,attack_data_path,window_size,hidden_size,batch_size)

        input_feature_size = windows_normal.shape[2]
        windows_size = windows_normal.shape[1]
        # w_size = 一整個window 的input 變成一維
        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        # w_size = 一整個window 的latent 變成一維
        z_size=windows_normal.shape[1]*hidden_size


        if modelName == "USAD":
            model = UsadModel(w_size, z_size)
        elif modelName == "autoencoder":
            model = AutoencoderModel(w_size, z_size,input_feature_size)
        elif modelName == "LSTM_USAD":
            model = LSTM_UsadModel(w_size, z_size,input_feature_size,windows_size)
        elif modelName == "LSTM_VAE":
            model = LSTM_VAE(input_feature_size-5,hidden_size,input_feature_size,windows_size)
        elif modelName == "CNN_LSTM":
            model = CNN_LSTM(hidden_size,input_feature_size,windows_size)
        else:
            print("model name not found")
            exit()

        model = to_device(model,device)
        history = model.training_all(n_epochs,train_loader,val_loader)
        print("model",model)
        plot_history(history)

        model.saveModel()

    
    def test(self,modelName):
        _,_,test_loader,windows_normal,labels= handleData(normal_data_path,attack_data_path,window_size,hidden_size,batch_size)
        
        input_feature_size = windows_normal.shape[2]
        windows_size = windows_normal.shape[1]
        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        z_size=windows_normal.shape[1]*hidden_size


        if modelName == "USAD":
            model = UsadModel(w_size, z_size)
        elif modelName == "autoencoder":
            model = AutoencoderModel(w_size, z_size,input_feature_size,)
        elif modelName == "LSTM_USAD":
            model = LSTM_UsadModel(w_size, z_size,input_feature_size,windows_size)
        elif modelName == "LSTM_VAE":
            model = LSTM_VAE(input_feature_size-5,hidden_size,input_feature_size,windows_size)
        elif modelName == "CNN_LSTM":
            model = CNN_LSTM(hidden_size,input_feature_size,windows_size)
        else:
            print("model name not found")
            
        model = to_device(model,device)
        model.loadModel()

        # ## Testing
        ## 這個result 是越高越可能是anomaly
        results=model.testing_all(test_loader)


    ## 將seq label 變成windows_labels 
        windows_labels=[]
        for i in range(len(labels)-window_size):
            windows_labels.append(list(np.int_(labels[i:i+window_size])))


        ## 這邊就是windows裡面只要有一個是anomaly整段windows都標記成anomaly
        y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]


        y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                    results[-1].flatten().detach().cpu().numpy()])


        # print("y_test len ,y_pre len",len(y_test),len(y_pred))
        threshold=ROC(y_test,y_pred)
        printResult(y_test,y_pred,threshold)
        print("threshold",threshold)





if __name__ == "__main__":
    import sys
    executionObj = execution()
    if len(sys.argv) <3:
        print("usage : ipython main.py {train|test} {USAD|autoencoder|LSTM_USAD}")
    else:
        if sys.argv[1] == "test":
            executionObj.test(sys.argv[2])
        else:
            executionObj.train(sys.argv[2])
