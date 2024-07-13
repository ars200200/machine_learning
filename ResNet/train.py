#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import cv2
from arch import *

import tqdm 
import os
from torch.cuda.amp import autocast, GradScaler
import yaml
import optimizers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, required=True)
args = parser.parse_args()
option_path = args.option
with open(option_path, 'r') as file_options:
    option = yaml.safe_load(file_options)

import zipfile as zf
files = zf.ZipFile("archive.zip",'r')
files.extractall()
files.close()

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()



class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()
        
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2
        
        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))
    
    def __len__(self):
        return len(self.dir2_list) + len(self.dir1_list)
    
    def __getitem__(self, idx :int):
        
        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        
        img = cv2.resize(img, (64, 64),  interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)
        
        return {'img': t_img, 'label': t_class_id}



train_dogs_path = option['dataset']['train_dogs_path']
train_cats_path = option['dataset']['train_cats_path']
test_dogs_path = option['dataset']['test_dogs_path']
test_cats_path = option['dataset']['test_cats_path']

train_ds_catsdogs = Dataset2class(
    train_dogs_path,
    train_cats_path
)

test_ds_catsdogs = Dataset2class(
    test_dogs_path,
    test_cats_path
)



batch_size = option['dataset']['batch_size']

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=True
    
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0
    
    
)


option_network = option['network']
model = get_network(option_network)

option_optim = option['optimizer']
loss_fn = nn.CrossEntropyLoss()
optimizer = optimizers.get_optimizers(model.parameters(), option_optim)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
loss_fn = loss_fn.to(device)


use_amp = True 
scaler = torch.cuda.amp.GradScaler()


epochs = 10
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm.tqdm(train_loader)):
        img, label = sample['img'], sample['label']
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        label = F.one_hot(label, 2).float()
            
        
        pred = model(img)

        loss = loss_fn(pred, label)

        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item
        
        optimizer.step()
        acc_current = accuracy(pred.cpu().float(), label.cpu().float())
        acc_val += acc_current

    pbar.set_description(f'loss:{loss_item:.4e}\taccuracy: {acc_current:.3f}')
    print(loss_val/len(train_loader))
    print(acc_val/len(train_loader))



