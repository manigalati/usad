import numpy as np
import torch.utils.data as data_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve,roc_auc_score

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
    
def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    # losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    # plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig("history")
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.savefig("histogram")
    
def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.savefig("ROC")
    return tr[idx]
def printResult(y_test,y_pred,threshold):
    print("============== result ==================")
    y_pred=[ 1 if(x>=threshold) else 0 for x in y_pred]

    precision, recall, fscore, support = score(y_test, y_pred)

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



    
def dataPreprocessing(dataset):
    # 不知道為啥read 進來的float .會變成,
    for i in list(dataset): 
        dataset[i]=dataset[i].apply(lambda x: str(x).replace("," , "."))
    # Transform all columns into float64 
    
    dataset = dataset.astype(float)

    # #### Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x = dataset.values
    x_scaled = min_max_scaler.fit_transform(x)
    dataset = pd.DataFrame(x_scaled)
    # In[ ]:
    print("dataset.head(2)",dataset.head(2))
    return dataset

def seq2Window(dataset,window_size):
    # print("np.arange",np.arange(window_size)[None, :])
    # print("normal.shape[0]",np.arange(normal.shape[0]-window_size)[:, None])
    # print("3",np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None])
    windows_normal=dataset.values[np.arange(window_size)[None, :] + np.arange(dataset.shape[0]-window_size)[:, None]]
    return windows_normal

def SWAT_loadData(normal_data_path,attack_data_path):
    #### Normal 
    normal = pd.read_csv(normal_data_path)#, nrows=1000)
    normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
    print("normal data.shape",normal.shape)

    #### Attack

    # attack= pd.read_csv(attack_data_path,sep=";")
    attack= pd.read_csv(attack_data_path)
    labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
    return normal,attack,labels

def WADI_loadData(normal_data_path,attack_data_path):
    #### Normal 
    normal = pd.read_csv(normal_data_path)#, nrows=1000)
    print("normal.head(2)",normal.head(2))
    normal = normal.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)
    # normal= normal.dropna(axis=1)
    normal= normal.fillna(0)
    print("normal.head(2)",normal.head(2))
    # drop column with at leat one NAN

    #### Attack

    # attack= pd.read_csv(attack_data_path,sep=";")
    attack= pd.read_csv(attack_data_path)
    print("attack.head(2)",attack.head(2))
    labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
    attack = attack.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)
    attack=attack.fillna(0)
    # attack= attack.dropna(axis=1)
    print("attack.head(2)",attack.head(2))


    
    return normal,attack,labels
    
def handleData(normal_data_path,attack_data_path,window_size,hidden_size,BATCH_SIZE):
    normal,attack,labels = WADI_loadData(normal_data_path,attack_data_path)
    #normal,attack,labels = SWAT_loadData(normal_data_path,attack_data_path)

    normal=dataPreprocessing(normal)
    attack=dataPreprocessing(attack)

    windows_normal=seq2Window(normal,window_size)
    print("windows_normal.shape",windows_normal.shape)

    windows_attack=seq2Window(attack,window_size)
    print("window_attack.shape",windows_attack.shape)


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

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader,val_loader,test_loader,windows_normal,labels
