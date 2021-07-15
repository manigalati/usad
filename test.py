#!/usr/bin/env python
# coding: utf-8

# # USAD

# ## Environment

# In[3]:

AE_only=True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
from usad import *


# In[4]:


get_ipython().system('nvidia-smi -L')

device = get_default_device()


# ## EDA - Data Pre-Processing

# ### Download dataset

# In[5]:


# get_ipython().system('mkdir input')
# #normal period
# get_ipython().system('python gdrivedl.py https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw input/')
# #anomalies
# get_ipython().system('python gdrivedl.py https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7 input/')


# ### Normal period

# In[6]:


#Read data
normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
print("normal data.shape",normal.shape)


# In[ ]:


# Transform all columns into float64 
# because origin data have weird format float number
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)


# #### Normalization

# In[ ]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)


# In[ ]:


print("normal.head(2)",normal.head(2))


# ### Attack

# In[ ]:


#Read data
attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
print("attack.shape",attack.shape)


# In[ ]:


# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)


# #### Normalization

# In[ ]:


from sklearn import preprocessing

x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)


# In[ ]:


print("attack.head(2)",attack.head(2))


# ### Windows

# In[ ]:


window_size=12


# In[ ]:

print("np.arange",np.arange(window_size)[None, :])
print("normal.shape[0]",np.arange(normal.shape[0]-window_size)[:, None])
print("3",np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None])

windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
print("windows_normal.shape",windows_normal.shape)


# In[ ]:


windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
print("window_attack.shape",windows_attack.shape)


# ## Training

# In[ ]:


import torch.utils.data as data_utils

BATCH_SIZE =  7919
#BATCH_SIZE = 100 
N_EPOCHS = 100
hidden_size = 40

w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

print("w_size",w_size)
print("z_size",z_size)

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
model = to_device(model,device)


# In[ ]:


# ## Testing

# In[ ]:


checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])


# In[ ]:


results=testing(model,test_loader,AE_only)

print("results",results)

# In[ ]:


windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))

print("windows_labels.shape",np.array(windows_labels).shape)

# In[ ]:


y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]


print("y_test",y_test)
# In[ ]:


y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])

print("y_pred",y_pred)
print("y_pred",y_pred.shape)

# In[ ]:


threshold=ROC(y_test,y_pred)

print("threshold",threshold)

# In[ ]:




