import pandas as pd


import numpy as np



filePath = "WADI_attackdata.csv"
attack = pd.read_csv(filePath)


attack["Normal/Attack"] = pd.Series(np.random.randn(attack.shape[0]),index=attack.index,dtype=str)


for index,row in attack.iterrows():
    attack.at[index,"Normal/Attack"] = "Normal"

def labelOneEvent(Date,Time,span):
    count=0
    for index,row in attack.iterrows():
        if count > 0:
            count -=1
            attack.at[index,"Normal/Attack"] = "Attack"
        elif row["Date"] == Date and  row["Time"] == Time:
            print("match",row)
            count=25*60+16
            attack.at[index,"Normal/Attack"] = "Attack"
            
# print("test",attack[["Date","Time","1_P_006"]])
        

labelOneEvent("10/9/2017","7:25:00.000 PM",25*60+16)
labelOneEvent("10/10/2017","10:24:10.000 AM",9*60+50)
labelOneEvent("10/10/2017","10:55:00.000 AM",29*60)
labelOneEvent("10/10/2017","11:30:40.000 AM",14*60+10)
labelOneEvent("10/10/2017","1:39:30.000 PM",11*60+10)
labelOneEvent("10/10/2017","2:48:17.000 PM",11*60+38)
labelOneEvent("10/10/2017","5:40:00.000 PM",9*60+40)
labelOneEvent("10/11/2017","10:55:00.000 AM",1*60+27)
labelOneEvent("10/11/2017","11:17:54.000 AM",14*60+26)
labelOneEvent("10/11/2017","11:36:31.000 AM",11*60+29)
labelOneEvent("10/11/2017","11:59:00.000 AM",6*60)
labelOneEvent("10/11/2017","12:07:30.000 PM",3*60+22)
labelOneEvent("10/11/2017","12:16:00.000 PM",9*60+36)
labelOneEvent("10/11/2017","3:26:30.000 PM",11*60+30)
attack.drop(['Row'],axis=1)

attack.to_csv("WADI_Attack_pre.csv",index_label=False)

