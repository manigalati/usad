from ast import parse
import numpy as np
from numpy.lib.function_base import rot90
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


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
def plot_history(history,modelName):
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
    plt.savefig("result/history_"+modelName)
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.savefig("histogram")
    
    
def ROC(y_test,y_pred,modelName):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    plt.savefig("result/ROC_"+modelName)
    plt.clf()
    return tr[idx]
def printDataInfo(dataset):
    abnormalCount=0
    normalCount=0
    for label in dataset["Normal/Attack"]:
        if label == "Normal":
            normalCount+=1
        else:
            abnormalCount+=1
    print("#####data info########")
    print("number of anomaly :",abnormalCount)
    print("number of normal :",normalCount)
    print("################")
def evaluateResult(y_True,y_pred,threshold,modelName):
    y_pred_anomaly=[ 1 if(x>=threshold) else 0 for x in y_pred]
    TP=0;TN=0;FP=0;FN=0;
    for index,item in enumerate(y_pred_anomaly):
        if y_pred_anomaly[index] == 1 and y_True[index] == 1:
            TP+=1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 0:
            TN+=1
        elif y_pred_anomaly[index] == 1 and y_True[index] == 0:
            FP+=1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 1:
            FN+=1


    recall=float(TP/(TP+FN))
    precision= float(TP/(TP+FP))
    with open("result/"+modelName+"_result.txt",'a') as resultFile:
        print("-------------------",file=resultFile)
        print("TP:",TP,"TN:",TN,"FP:",FP,"FN:",FN,file=resultFile)
        print("precision:",precision,file=resultFile)
        print("recall:",recall,file=resultFile)
        print("F1 score",2*precision*recall/(precision+recall),file=resultFile)
        print("TPR",TP/(TP+FN),file=resultFile)
        print("FPR",FP/(TN+FP),file=resultFile)
        print("-------------------",file=resultFile)

def printResult(y_True,y_pred,threshold,modelName):
        y_pred_anomaly=[ 1 if(x>=threshold) else 0 for x in y_pred]

        precision, recall, fscore, support = score(y_True, y_pred_anomaly)
        ### caculate recall
        print("============== result ==================")
        evaluateResult(y_True,y_pred,threshold,modelName)

        print('precision: {}'.format(precision[0]))
        print('recall: {}'.format(recall[0]))
        print('f1score: {}'.format(fscore[0]))
        print("============== result ==================")
    

def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    
    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.savefig("confusion_matrix")



def plotData(featureNameList,dataset):
    print("dataset columns",dataset.columns)
    # featureNameList = ["1_MV_001","1_FFT_001","2_LIT_002","1_AIT_001",\
    #     "2_MCV_101","2_MCV_201","2_MCV_301","2_MCV_401","2_MCV_501",\
    #     "2_MCV_601","2_MCV_101","2_MCV_201","1_AIT_002","2_MV_003",\
    #     "2_MCV_007","1_P_006"]
    # FinalfeatureNameList=[]
    # for columnName in dataset.columns:
    #     for featureName in featureNameList:
    #         if columnName.find(featureName) != -1:
    #             FinalfeatureNameList.append(columnName)
    # print("finalFeatureNameList",FinalfeatureNameList)


    # plotFeature= dataset[FinalfeatureNameList] 
    plt.figure(figsize=(100,10))
    # fig,ax  = plt.subplots(3,1)
    plt.title("features")
    numberOfFeatures = 3
    count=0
    
    for column in dataset:
        count+=1
        if count > numberOfFeatures:
            break
        ax = plt.subplot(310+count)
        plt.plot(dataset.index,dataset[column])
        print("column",column)
        # ax.title.set_text(column)
        plt.gca().set_title(column)
        # ax[count-1,0].plot(dataset.index,dataset[column])
        # ax[count-1,0].set_title(column)
    # plt.show()
    plt.savefig("result/feature")
    plt.close()
    # plotFeature.plot()


def plotAnomalyScore(window_size,dataset,windows_anomalyScore,threshold,label,modelName):
    plt.figure(figsize=(100,10))
    index = dataset[window_size:].index
    print("index",index)
    plt.plot(index,windows_anomalyScore,'b',label="pred")
    plt.plot(index,label,'r',label="ground_true")
    plt.axhline(threshold)

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
    plt.savefig("result/anomalyScore_"+modelName)
    plt.close()

    