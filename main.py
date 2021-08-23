#!/usr/bin/env python
# coding: utf-8

######################### parameter

batch_size =  7919
#BATCH_SIZE = 100 
n_epochs=  50
hidden_size = 40
window_size=12
# normal_data_path="input/SWaT_Dataset_Normal_v1.csv"
# attack_data_path="input/SWaT_Dataset_Attack_v0.csv"
normal_data_path="/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Normal_v1.csv"
attack_data_path="/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Attack_v0.csv"
# attack_data_path="/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Attack_v0_test.csv"
# normal_data_path="/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017/WADI_normal_pre_2.csv"
# attack_data_path="/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017/WADI_Attack_pre.csv"
                #  "/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017"
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
from dataPreprocessing import *
import torch.utils.data as data_utils
import argparse


# In[4]:

# get_ipython().system('nvidia-smi -L')
device = get_default_device()

class execution:
    def __init__(self):
        self.dataPreprocessingObj  = DataProcessing()

    def preProcess(self,modelName):

        # train_loader 每次iter 取出來的東西是一個windows 攤平成一維的數據
        self.train_loader,self.val_loader,self.test_loader,self.windows_dataset,self.labels,self.attack= self.dataPreprocessingObj.handleData(normal_data_path,attack_data_path,window_size,hidden_size,batch_size)

        input_feature_size = self.windows_dataset.shape[2]
        windows_size = self.windows_dataset.shape[1]
        # w_size = 一整個window 的input 變成一維
        w_size=self.windows_dataset.shape[1]*self.windows_dataset.shape[2]
        # w_size = 一整個window 的latent 變成一維
        z_size=self.windows_dataset.shape[1]*hidden_size


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

        self.model = to_device(model,device)

    def train(self,modelName):
        self.preProcess(modelName)
        history = self.model.training_all(n_epochs,self.train_loader,self.val_loader)
        print("model",self.model)
        plot_history(history,modelName)

        self.model.saveModel()

    

    def test(self,modelName):


        self.preProcess(modelName)
        ####### Testing
        ## 這個result 是越高越可能是anomaly
        results=self.model.testing_all(self.test_loader)

        # # print("results.shape",results.size())
        # print("torch.stack(results[:-1])",torch.stack(results[:-1]).cpu().size())
        # print("result[-1].shape",(results[-1].cpu()).size())


        ### windows 的 label
        y_True = self.seqLabel_2_WindowsLabels(self.labels)

        # 傳回來的result dim沒調好 需要flatten
        ### windows 的 pred
        y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                    results[-1].flatten().detach().cpu().numpy()])

        # print("y_pred.shape",y_pred.shape)
        # print("y_True.shape",np.array(y_True).shape)

        threshold=ROC(y_True,y_pred,modelName)
        plotAnomalyScore(window_size,self.attack,y_pred,threshold,y_True,modelName)
        printResult(y_True,y_pred,threshold,modelName)
        print("threshold",threshold)

    def seqLabel_2_WindowsLabels(self,labels):
        ## 將seq label 變成windows_labels 
        windows_labels=[]
        for i in range(len(labels)-window_size):
            windows_labels.append(list(np.int_(labels[i:i+window_size])))
        ## 這邊就是windows裡面只要有一個是anomaly整段windows都標記成anomaly
        y_True= [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

        return y_True




if __name__ == "__main__":
    import sys
    executionObj = execution()
    argparserObj = argparse.ArgumentParser()
    argparserObj.add_argument("--model",type=str,help="{USAD|autoencoder|LSTM_USAD|LSTM_VAE|CNN_LSTM}")
    argparserObj.add_argument("--action",type=str,help="{test|train}")
    args  = argparserObj.parse_args()
    if args.action == "test":
        executionObj.test(args.model)
    else:
        executionObj.train(args.model)
