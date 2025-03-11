from dataset import ProcessDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import os
from pathlib import Path
import glob
import csv

def read_pH_temp_csv(file_path):
    data_dict = {}
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            pdb_id = row[0]
            ph = float(row[1])
            temp = float(row[2])
            data_dict[pdb_id] = [ph, temp]
    return data_dict


def get_ds():
    temp_pH_vals = read_pH_temp_csv("model/pH_temp.csv")
    full_ds = ProcessDataset(temp_pH_vals)
    train_ds_size = int(0.8 * len(full_ds))
    
    train_ds = full_ds[:train_ds_size]
    val_ds = full_ds[train_ds_size:]

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    print(len(val_dataloader))
    return full_ds
    
get_ds()

    
