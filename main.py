#!/usr/bin/env python
# coding: utf-8

# parameter

import argparse
import torch.utils.data as data_utils
from dataPreprocessing import *
from model import *
from utils import *
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
n_epochs = 10
hidden_size = 40
window_size = 12
# normal_data_path="input/SWaT_Dataset_Normal_v1.csv"
# attack_data_path="input/SWaT_Dataset_Attack_v0.csv"
normal_data_path = "/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Normal_v1.csv"
attackFeatureInfo_csv_path= "/workspace/lab/anomaly_detecton/dataset/SWAT/List_of_attacks_Final.csv"

TEST_testing=False
if TEST_testing == True:
    batch_size= 50
    attack_data_path="/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Attack_v0_test.csv"
else:
    batch_size=7919
    attack_data_path = "/workspace/lab/anomaly_detecton/dataset/SWAT/SWaT_Dataset_Attack_v0.csv"


# batch_size = 7919
# normal_data_path="/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017/WADI_normal_pre_2.csv"
# attack_data_path="/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017/WADI_Attack_pre.csv"
#  "/workspace/lab/anomaly_detecton/dataset/WADI.A1_9_Oct_2017"
#########################

# In[3]:


#from usad.utils import dataPreprocessing, handleData, seq2Window


# In[4]:

# get_ipython().system('nvidia-smi -L')

class execution:
    def __init__(self):
        self.dataPreprocessingObj = DataProcessing()
        self.device = get_default_device()

    def getModel(self, modelName,input_feature_size):


        # w_size = 一整個window 的input 變成一維
        w_size = input_feature_size * window_size
        # w_size = 一整個window 的latent 變成一維
        z_size = window_size*hidden_size

        if modelName == "USAD":
            model = UsadModel(w_size, z_size)
        elif modelName == "autoencoder":
            model = AutoencoderModel(w_size, z_size, input_feature_size)
        elif modelName == "LSTM_USAD":
            model = LSTM_UsadModel(
                w_size, z_size, input_feature_size, window_size)
        elif modelName == "LSTM_VAE":
            model = LSTM_VAE(input_feature_size-5, hidden_size,
                             input_feature_size, window_size)
        elif modelName == "CNN_LSTM":
            model = CNN_LSTM(hidden_size, input_feature_size, window_size)
        else:
            print("model name not found")
            exit()

        self.model = to_device(model, self.device)

    def train(self, modelName):
        # train_loader 每次iter 取出來的東西是一個windows 攤平成一維的數據
        train_loader, val_loader, input_feature_size = self.dataPreprocessingObj.HandleNormalData(
            normal_data_path, window_size, hidden_size, batch_size)
        self.getModel(modelName,input_feature_size)
        history = self.model.training_all(
            n_epochs, train_loader, val_loader)
        print("model", self.model)
        plot_history(history, modelName)

        self.model.saveModel()

    def test(self, modelName):
        test_loader,self.labels,self.origin_attack,self.attack,input_feature_size = self.dataPreprocessingObj.HandleAnomalyData(
            attack_data_path, window_size, hidden_size, batch_size)
        self.getModel(modelName,input_feature_size)

        plotFeatureObj=plotFeature(attackFeatureInfo_csv_path,self.origin_attack,modelName,window_size)
        # Testing
        # 這個result 是越高越可能是anomaly
        self.model.loadModel()
        results = self.model.testing_all(test_loader)

        print("torch.stack(results[:-1])",
              torch.stack(results[:-1]).cpu().size())
        print("result[-1].shape", (results[-1].cpu()).size())

        # windows 的 label
        y_True = seqLabel_2_WindowsLabels(window_size,self.labels)

        # 傳回來的result dim沒調好(batch 的形式) ,用torch.stack 是為了把list變成tensor 使用flatten
        # windows 的 anomaly_score
        # y_anomaly_score=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
        #                             results[-1].flatten().detach().cpu().numpy()])
        y_anomaly_score = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                          results[-1].flatten().detach().cpu().numpy()])


        threshold = ROC(y_True, y_anomaly_score, modelName)
        plotAnomalyScore(window_size, self.origin_attack,
                         y_anomaly_score, threshold, y_True, modelName)


        input_output_window_result = self.model.get_intput_output_window_result()
        print("input_output_window_result[input]",input_output_window_result["input"][:,-1,0])
        print("self.attack",self.attack[window_size-1:])
        print("self.original_attack",self.origin_attack[window_size-1:])
        print("input_output_window_result[output]",input_output_window_result["output"][:,-1,0])
        # plotFeatureObj.plot_anomalyFeature(
        #          input_output_window_result["input"][:,-1,:],
        #          input_output_window_result["output"][:,-1,:],
        #          input_output_window_result["loss"],
        #          )

        printResult(y_True, y_anomaly_score, threshold, modelName)
        print("threshold", threshold)



if __name__ == "__main__":
    import sys
    executionObj = execution()
    argparserObj = argparse.ArgumentParser()
    argparserObj.add_argument(
        "--model", type=str, help="{USAD|autoencoder|LSTM_USAD|LSTM_VAE|CNN_LSTM}")
    argparserObj.add_argument("--action", type=str, help="{test|train}")
    args = argparserObj.parse_args()
    if args.action == "test":
        executionObj.test(args.model)
    else:
        executionObj.train(args.model)
