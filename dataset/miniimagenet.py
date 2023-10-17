'''
Every dataset should make df_dataset where 

df_dataset = {
    "train": <data frame for train>
    "val": <data frame for val>
    "test": <data frame for test>
}
'''
import os
import json
import numpy as np
import pandas as pd

def setup_df(dataset_root = "./data/miniimagenet/"):
    df_dataset = {}
    img_dir_path = os.path.join(dataset_root,"images/")
    for split in ["train","val","test"]:
        df =  pd.read_csv(os.path.join(dataset_root,"%s.csv"%split))
        df["path"] = img_dir_path+df["filename"]
        del df["filename"]
        df_dataset[split] = df
    return df_dataset