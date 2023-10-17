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
import glob

def setup_df(dataset_root="./data/car/"):
    table = []
    for path in glob.glob(os.path.join(dataset_root, "images/*/*.jpg")):
        print(path)
        label = path.split("/")[-2]
        table.append({"label": label, "path": path, "img_name": path.split("/")[-1]})
    df_all = pd.DataFrame(table)
    df_all = df_all.sort_values(["path"])
    labels = df_all["label"].unique().tolist()
    labels.sort()
    label2split = {}

    for i, label in enumerate(labels):
        if i % 2 == 0:
            label2split[label] = "train"
        elif i % 4 == 3:
            label2split[label] = "val"
        elif i % 4 == 1:
            label2split[label] = "test"
    df_all["split"] = df_all['label'].map(label2split)

    df_dataset = {}
    for split in ["train", "val", "test"]:
        df = df_all.query("split=='%s'" % split)
        df = df.sort_values(by=['path'])
        df_dataset[split] = df

    return df_dataset
