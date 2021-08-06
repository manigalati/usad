import pandas as pd

normal = pd.read_csv("WADI_normal.csv")
    
normal["Normal/Attack"] = "Normal"

normal.to_csv("WADI_normal_2.csv")