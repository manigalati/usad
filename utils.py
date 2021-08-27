from ast import parse
from operator import indexOf
from typing import OrderedDict
import numpy as np
from numpy.lib.function_base import rot90
from pandas.io.parsers import read_csv
import torch.utils.data as data_utils
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
import seaborn as sns
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve, roc_auc_score

import sys

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD

    
    


def seqLabel_2_WindowsLabels(window_size, labels):
    # 將seq label 變成windows_labels
    windows_labels = []
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int_(labels[i:i+window_size])))
    # 這邊就是windows裡面只要有一個是anomaly整段windows都標記成anomaly
    y_True = [1.0 if (np.sum(window) > 0)
                else 0 for window in windows_labels]
    return y_True

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history, modelName):
    losses1 = [x['val_loss1'] for x in history]
    if modelName == "USAD":
        losses2 = [x['val_loss2'] for x in history]
        plt.plot(losses2, '-x', label="loss2")
    plt.plot(losses1, '-x', label="loss1")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig("result/"+modelName+"/history")


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist([y_pred[y_test == 0],
              y_pred[y_test == 1]],
             bins=20,
             color=['#82E0AA', '#EC7063'], stacked=True)
    plt.title("Results", size=20)
    plt.grid()
    plt.savefig("histogram")


def ROC(y_test, y_pred, modelName):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.plot(fpr, 1-fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    plt.savefig("result/"+modelName+"/ROC")
    plt.clf()
    return tr[idx]


def printDataInfo(dataset):
    abnormalCount = 0
    normalCount = 0
    for label in dataset["Normal/Attack"]:
        if label == "Normal":
            normalCount += 1
        else:
            abnormalCount += 1
    print("#####data info########")
    print("number of anomaly :", abnormalCount)
    print("number of normal :", normalCount)
    print("################")


def evaluateResult(y_True, y_pred, threshold, modelName):
    y_pred_anomaly = [1 if(x >= threshold) else 0 for x in y_pred]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, item in enumerate(y_pred_anomaly):
        if y_pred_anomaly[index] == 1 and y_True[index] == 1:
            TP += 1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 0:
            TN += 1
        elif y_pred_anomaly[index] == 1 and y_True[index] == 0:
            FP += 1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 1:
            FN += 1

    recall = float(TP/(TP+FN))
    precision = float(TP/(TP+FP))
    with open("result/"+modelName+"/result.txt", 'a') as resultFile:
        print("-------------------", file=resultFile)
        print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN, file=resultFile)
        print("precision:", precision, file=resultFile)
        print("recall:", recall, file=resultFile)
        print("F1 score", 2*precision*recall /
              (precision+recall), file=resultFile)
        print("TPR", TP/(TP+FN), file=resultFile)
        print("FPR", FP/(TN+FP), file=resultFile)
        print("-------------------", file=resultFile)


def printResult(y_True, y_pred, threshold, modelName):
    y_pred_anomaly = [1 if(x >= threshold) else 0 for x in y_pred]

    precision, recall, fscore, support = score(y_True, y_pred_anomaly)
    # caculate recall
    print("============== result ==================")
    evaluateResult(y_True, y_pred, threshold, modelName)

    print('precision: {}'.format(precision[0]))
    print('recall: {}'.format(recall[0]))
    print('f1score: {}'.format(fscore[0]))
    print("============== result ==================")


def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=[
                                   'Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix),
                    annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.savefig("confusion_matrix")




def anomalyScore2anomaly(anomalyScore, threshold):
    anomaly = [x >= threshold for x in anomalyScore]
    return anomaly


def plotAnomalyScore(window_size, dataset, windows_anomalyScore, threshold, label, modelName):
    anomaly = anomalyScore2anomaly(windows_anomalyScore, threshold)
    plt.figure(figsize=(100, 10))
    index = dataset[window_size:].index
    # print("index", index)
    anomaly_df = pd.DataFrame(data = anomaly,index = index,columns=["label"])
    anomaly_df = anomaly_df[anomaly_df["label"] == True]
    print("anomaly_df",anomaly_df)

    plt.plot(index, windows_anomalyScore, 'b', label="pred")
    plt.plot(index, label, 'r', label="ground_true")
    # plt.plot(index, anomaly, 'y', label="pred_anomaly")
    plt.axhline(threshold)
    # [ plt.axvline(index,color='y') for index,row in anomaly_df.iterrows()]
    # [ plt.axvline(index,color='y') for index,row in anomaly_df.iterrows()]

    # plt.axvline()
    plt.ylabel("anomaly_score")
    plt.xlabel("time")
    # plt.xticks(dataset.index,rot=90)
    # plt.legend(["predict","label"])
    plt.legend()
    # plt.yticks([threshold,1])
    # plt.show()
    # anomaly = dataset["Normal/Attack"]
    # anomaly.index = dataset.index
    # anomaly["anomalyScore"] = anomalyScore
    plt.savefig("result/"+modelName+"/anomalyScore")
    plt.close()


class plotFeature:
    def __init__(self,anomaly_info_file_path,origin_dataset,modelName,windows_size):
        self.feature_info_dict={}
        self.anomaly_info=self.getAttackFeatureInfo(anomaly_info_file_path)
        self.featureNameList =  self.getFeatureNameList()
        self.origin_dataset=origin_dataset
        self.modelName = modelName
        self.window_size = windows_size
        self.plotOriginalData()
        
    def getFeatureNameList(self):
        featureNameList =  [x for x in self.feature_info_dict.keys()]
        return featureNameList[:5]


    def dataFrameTime2Index(self,dataFrameTime):
        df = self.origin_dataset.reset_index()
        return df[df["Timestamp"] == dataFrameTime].index
    def seqIndex2WindowIndex(self,index):
        return index - self.window_size

    def getAnomalyScoreSortByTime(self,dataFrameTimeList,loss):
        with open("result/"+self.modelName+"/loss_cause.txt","a") as f:
            for dataFrameTime in dataFrameTimeList:
                index = self.seqIndex2WindowIndex(self.dataFrameTime2Index(dataFrameTime))
                # print("dataFrameTime:",dataFrameTime,"index:",index)
                print("time:",dataFrameTime,"loss:",loss[index].argsort(),file=f)
                # print("time:",dataFrameTime,"loss:",loss[index].argsort())
        
    def plotOriginalData(self):
        plt.figure(figsize=(150, 15))
        # fig,ax  = plt.subplots(3,1)
        plt.title("original_features")

        for index, featureName in enumerate(self.featureNameList):
            ax = plt.subplot(len(self.featureNameList), 1, index+1)
            plt.plot(self.origin_dataset.index, self.origin_dataset[featureName],label="origin_data",color='b')
            plt.gca().set_title(featureName)
            # plt.xticks(self.get_anomaly_time_by_FeatureName(featureName),rotation=90)

        plt.savefig("result/"+self.modelName+"/original_feature")
        plt.close()

    def plotData(self,featureNameList, dataset, input_Feature_list, output_Feature_list, loss, window_size,feature_info_dict={}):
        # print("dataset columns",dataset.columns)
        # ,{"1_FFT_001":,"2_LIT_002","1_AIT_001",\
        # "2_MCV_101","2_MCV_201","2_MCV_301","2_MCV_401","2_MCV_501",\
        # "2_MCV_601","2_MCV_101","2_MCV_201","1_AIT_002","2_MV_003",\
        # "2_MCV_007","1_P_006"]
        # FinalfeatureNameList=[]
        # for columnName in dataset.columns:
        #     print("columnName",columnName)
            # for featureName in featureNameList:
            #     if columnName.find(featureName) != -1:
            #         FinalfeatureNameList.append(columnName)
        # print("finalFeatureNameList",FinalfeatureNameList)
        # plotFeature= dataset[FinalfeatureNameList]
        plt.figure(figsize=(150, 15))
        plt.title("features_anomaly")
        print("featureNameList",featureNameList)
        for index,featureName in enumerate(featureNameList):
            self.plot_origin_predict_anomalyScore_singleFeature(
                 featureNameList,index,dataset, input_Feature_list, output_Feature_list, loss,window_size,feature_info_dict)
        plt.legend()
        plt.savefig("result/"+self.modelName+"/feature_anomaly")
        plt.close()

    def get_anomaly_time_by_FeatureName(self,featureName):
        result=[]
        [ result.extend(x) for x in self.feature_info_dict[featureName]]
        return result

    def featureName2index(self,featureName):
        count=0
        for column in self.origin_dataset.columns:
            if column == featureName:
                return count
            else:
                count+=1
        
    def plot_origin_predict_anomalyScore_singleFeature(self,featureNameList,indexOfFeatureNameList, dataset,inputFeature_list, outputFeature_list, 
        loss,window_size,feature_info_dict={}):

        featureName = featureNameList[indexOfFeatureNameList]
        index = self.featureName2index(featureName) 
        
        print("index",index)
        print("featureName ",featureName)
        print("origin_without_preProcessing_data",dataset[featureName][window_size:])
        print("inputFeature_list",inputFeature_list[:,index])
        print("outputFeature_list",outputFeature_list[:,index])
        print("loss",loss[:,index])

        ax = plt.subplot(len(featureNameList), 1, indexOfFeatureNameList+1)
        # plt.plot(dataset.index, dataset[featureName],label="origin_data",color='b')
        plt.plot(dataset.index[window_size:], inputFeature_list[:, index],label="input",color='c')
        plt.plot(dataset.index[window_size:], outputFeature_list[:, index],label="output",color='k')
        plt.plot(dataset.index[window_size:], loss[:, index],label="loss",color='r')
        plt.xticks(self.get_anomaly_time_by_FeatureName(featureName),rotation=90)
        plt.gca().set_title(featureName)

    def getAttackFeatureInfo(self,attackFeatureInfo_csv_path):
        df = pd.read_csv(attackFeatureInfo_csv_path)
        df= df.dropna(axis=0,subset=["End Time"])
        date_list = [x.split()[0]+" " for x in df["Start Time"].astype(str)]
        date_list = pd.Series(date_list)
        df["End Time"] = pd.to_datetime((date_list + df["End Time"] ).str.strip(),format="%d/%m/%Y %H:%M:%S")
        df["Start Time"] = pd.to_datetime(df["Start Time"].str.strip(),format="%d/%m/%Y %H:%M:%S")
        df.drop(df.tail(5).index,inplace=True)
        # print("4 df\n",df[["Start Time","End Time"]])
        df = self.preProcessAttackFeatureInfo(df)
        return df

    def preProcessAttackFeatureInfo(self,attack_info):
        for index,row in attack_info.iterrows():
            featureNameArray = row["Attack Point"]
            for featureName  in featureNameArray.split(","):
                featureName = featureName.strip().replace("-","")
                self.feature_info_dict.setdefault(featureName,[]).append((row["Start Time"],row["End Time"]))

        # print("feature_info_dict\n",self.feature_info_dict)
        return attack_info
            

    def plot_anomalyFeature(self,origin_attack,input_window,output_window,loss,window_size):
        self.plotData(self.featureNameList, origin_attack, 
                 input_window,output_window,loss,
                 window_size,
                 self.feature_info_dict)
        dataFrameTimeList = []
        for key,value in self.feature_info_dict.items():
            for tupleTime in value:
                dataFrameTimeList.extend(tupleTime)
        # print("dataFrameTimeList",dataFrameTimeList)
        self.getAnomalyScoreSortByTime(dataFrameTimeList,loss)


        