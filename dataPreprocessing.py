
from ast import parse
import numpy as np
import torch.utils.data as data_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve,roc_auc_score

import sys

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.transformer import PcaProjection
from utils import *
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
from adtk.detector import PcaAD

class DataProcessing:
    def useADTKLibraryPreprocessing(self,dataset):
        print("useLibraryPreprocessing data.shape",dataset.shape)
        dataset = validate_series(dataset)
        dataset = PcaProjection(k=30).fit_transform(dataset)
        print("after useLibraryPreprocessing data.shape",dataset.shape)

        
        return dataset
    
    def dataPreprocessing(self,dataset):
        
        # Transform all columns into float64 
        
        dataset = dataset.astype(float)
        # print("dataset dtype",dataset.dtypes)

        dataset = self.useADTKLibraryPreprocessing(dataset)
        # 不知道為啥read 進來的float .會變成,
        # #### Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        print("dataset.head(2)",dataset.head(2))
        dataset = pd.DataFrame(min_max_scaler.fit_transform(dataset.values))
        print("dataset.head(2)",dataset.head(2))
        return dataset

    def seq2Window(self,dataset,window_size):
        # print("np.arange",np.arange(window_size)[None, :])
        # print("normal.shape[0]",np.arange(dataset.shape[0]-window_size)[:, None])
        # print("3",np.arange(window_size)[None, :] + np.arange(dataset.shape[0]-window_size+1)[:, None])
        windows_normal=dataset.values[np.arange(window_size)[None, :] + np.arange(dataset.shape[0]-window_size+1)[:, None]]
        return windows_normal

    def SWAT_loadNormalData(self,data_path):
        ############################### Normal 
        normal = pd.read_csv(data_path)#, nrows=1000)
        # normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        normal = normal.drop(["Normal/Attack" ] , axis = 1)
        normal["Timestamp"] = normal["Timestamp"].str.strip()
        normal["Timestamp"] = pd.to_datetime(normal["Timestamp"],format="%d/%m/%Y %I:%M:%S %p")
        normal.set_index("Timestamp",inplace=True)
        print("normal data.shape",normal.shape)
        for i in list(normal): 
            normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal = normal.astype(float)

        return normal

    def SWAT_loadAnomalyData(self,data_path):
        attack = pd.read_csv(data_path)#, nrows=1000)
        # attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        printDataInfo(attack)
        labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
        attack = attack.drop(["Normal/Attack" ] , axis = 1)
        attack["Timestamp"] = attack["Timestamp"].str.strip()
        attack["Timestamp"] = pd.to_datetime(attack["Timestamp"],format="%d/%m/%Y %I:%M:%S %p")
        attack.set_index("Timestamp",inplace=True)
        for i in list(attack): 
            attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack = attack.astype(float)

        return attack,labels
    
    
    

    def WADI_loadData(self,normal_data_path,attack_data_path):
        #### Normal 
        normal = pd.read_csv(normal_data_path)#, nrows=1000)
        # print("normal.head(2)",normal.head(2))
        normal = normal.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)
        # normal= normal.dropna(axis=1)
        normal= normal.fillna(0)
        # print("normal.head(2)",normal.head(2))
        # drop column with at leat one NAN

        #### Attack

        # attack= pd.read_csv(attack_data_path,sep=";")
        attack= pd.read_csv(attack_data_path)
        # print("attack.head(2)",attack.head(2))
        labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
        attack = attack.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)
        attack=attack.fillna(0)
        # attack= attack.dropna(axis=1)
        # print("attack.head(2)",attack.head(2))


        
        return normal,attack,labels
    
    def HandleNormalData(self,normal_data_path,window_size,hidden_size,BATCH_SIZE):
        # normal,attack,labels = WADI_loadData(normal_data_path,attack_data_path)
        origin_normal= self.SWAT_loadNormalData(normal_data_path)
        normal=self.dataPreprocessing(origin_normal)
        # pca_ad = PcaAD(k=20)
        # anomalies = pca_ad.fit_detect(normal)
        # exit()

        windows_normal=self.seq2Window(normal,window_size)
        print("windows_normal.shape",windows_normal.shape)

        input_feature_size = windows_normal.shape[2]
        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        z_size=windows_normal.shape[1]*hidden_size

        print("w_size",w_size)
        print("z_size",z_size)

        ## divide training dataset and validation dataset
        windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
        windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        ## build loader
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        return train_loader,val_loader,input_feature_size


        
    def HandleAnomalyData(self,attack_data_path,window_size,hidden_size,BATCH_SIZE):
        # normal,attack,labels = WADI_loadData(normal_data_path,attack_data_path)
        origin_attack,labels = self.SWAT_loadAnomalyData(attack_data_path)
        
        attack=self.dataPreprocessing(origin_attack)


        windows_attack=self.seq2Window(attack,window_size)
        print("window_attack.shape",windows_attack.shape)

        input_feature_size = windows_attack.shape[2]
        w_size=windows_attack.shape[1]*windows_attack.shape[2]
        z_size=windows_attack.shape[1]*hidden_size

        print("w_size",w_size)
        print("z_size",z_size)

        ## build loader

        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        return test_loader,labels,origin_attack,attack,input_feature_size

