import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import random
from scipy import stats

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms

import os


class jenlife_glaucoma():
    def __init__(self, info_file,Filter=False, transform=None, target_transform=None, mode='train',feature=''):
        self.info = pd.read_csv(info_file)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.filter= Filter
        self.feature= feature

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        path =self.info.iloc[idx,]['path']
        Type=self.info.iloc[idx,]['type']
        all_data=torch.tensor(np.array(data[1]))
        if self.feature=='all':
            data=all_data
        elif self.feature=='opt':
            data_opt1=all_data[171:287]
            data_opt2=all_data[1085:1316]
            data_opt3=all_data[1869].reshape(1)
            data=torch.cat([torch.cat([data_opt1,data_opt2]),data_opt3])
        elif self.feature=='opt_wrong1':
            data_opt1=all_data[:171]
            data_opt2=all_data[287:464]
            data=torch.cat([data_opt1,data_opt2])
        elif self.feature=='opt_wrong2':
            data=all_data[675:1023]
        elif self.feature=='opt_wrong3':
            data_opt1=all_data[1652:1869]
            data_opt2=all_data[1870:]
            data=torch.cat([data_opt1,data_opt2])

        else: raise ValueError(f'unvalid feature: {self.feature}')

        answer=torch.tensor(0 if Type=='Control' else 1)
        
        return all_data, data, answer 
