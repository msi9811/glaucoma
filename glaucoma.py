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

import argparse
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, required=True, 
                        help='path of training dataset metadata (e.g. folder1/folder2/train_meta.csv)')
    parser.add_argument('--test_path', type=str, required=True, 
                        help='path of test dataset metadata (e.g. folder1/folder2/test_meta.csv)')
    parser.add_argument('--md_type', type=str, default='feature', 
                        help='basic model: basic, feature model: feature')
    

    args = parser.parse_args()

if __name__ == "__main__":
    main()

    train_dir=args.train_path
    test_dir=args.test_path
    feature=args.md_type
    
    batch_size=128
    dr_rate=0.1 #dropout
    k=5
    num_epochs=60
    repeat=1
    lr=0.00001 #learning rate
    wd=0.00001 #weight decay
    
    # Dataset Preprocessing---------------------------------------------------------------------------------------------------
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
            if self.feature=='basic':
                data=all_data
            elif self.feature=='feature':
                data_opt1=all_data[171:287]
                data_opt2=all_data[1085:1316]
                data_opt3=all_data[1869].reshape(1)
                data=torch.cat([torch.cat([data_opt1,data_opt2]),data_opt3])
    
            else: raise ValueError(f'unvalid feature: {self.feature}')
    
            answer=torch.tensor(0 if Type=='Control' else 1)
            
            return all_data, data, answer 
    
    data_train=jenlife_glaucoma(train_dir,feature=feature)
    data_test=jenlife_glaucoma(test_dir,feature=feature)
    test_loader=DataLoader(data_test,batch_size=batch_size,shuffle=False)
    
    # DL model ---------------------------------------------------------------------------------------------------
    class multiuse:
        def __init__(self,feature='all'):
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
            self.feature=feature
        
                
    class jenlife():
        def __init__():pass 
        class jenlife_highLayer(nn.Module, multiuse):
            def __init__(self,dr_rate=0.1,num_in=2001):
                super().__init__() #부모 클래스 input값 넣는 곳
                self.dr_rate=dr_rate
                self.num_in=num_in
                self.dataset='single'
                
                self.first_layer=nn.Sequential(
                nn.Linear(self.num_in,self.num_in*2),
                nn.BatchNorm1d(self.num_in*2),
                nn.PReLU(),
                nn.Linear(self.num_in*2,self.num_in*4),
                nn.BatchNorm1d(self.num_in*4),
                nn.PReLU(),
                nn.Linear(self.num_in*4,self.num_in*8),
                nn.BatchNorm1d(self.num_in*8),
                nn.Linear(self.num_in*8,1024),
                nn.Dropout(self.dr_rate),
                nn.Linear(1024,256),
                nn.Dropout(self.dr_rate),
                nn.Linear(256,2)
                )
    
    
            def forward(self, x):
                x=self.first_layer(x)
            
                return x
    
        class jenlife_opthyper_mlp(nn.Module, multiuse):
            def __init__(self,dr_rate=0.1):
                super().__init__() #부모 클래스 input값 넣는 곳
                self.dr_rate=dr_rate
                self.dataset='multie'
                self.num_in=2349
                
                self.first_layer=nn.Sequential(
                nn.Linear(2001,4000),
                nn.BatchNorm1d(4000),
                nn.PReLU(),
                nn.Linear(4000,2000),
                nn.BatchNorm1d(2000),
                nn.PReLU(),
                nn.Linear(2000,1000),
                nn.BatchNorm1d(1000),
                nn.PReLU()
                )
                
                self.second_layer=nn.Sequential(
                    nn.Linear(348,512),
                    nn.BatchNorm1d(512),
                    nn.PReLU(),
                    nn.Linear(512,1000),
                    nn.BatchNorm1d(1000),
                    nn.PReLU(),
                    nn.Linear(1000,1000),
                    nn.BatchNorm1d(1000),
                    nn.PReLU()
                )
            
                self.feature_layer=nn.Sequential(
                    nn.Linear(2000,200),
                    nn.Dropout(self.dr_rate),
                    nn.Linear(200, 2)
                )
    
    
            def forward(self, x, y):
                if len(x[0])==348: 
                    x=self.second_layer(x)
                    y=self.first_layer(y)
                    z=torch.cat((y,x),1)
                elif len(x[0])==2001:
                    x=self.first_layer(x)
                    y=self.second_layer(y)
                    z=torch.cat((x,y),1)
                z=torch.flatten(z,1)
                z=self.feature_layer(z)
            
                return z
    
    if feature=='basic': model=jenlife.jenlife_highLayer()
    elif feature=='feature': model=model_jenlife.jenlife_opthyper_mlp()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # model training -----------------------------------------------------------------------------------------------------
    def epoch(model,dataloader,loss_fn=None,optimizer=None,TRAIN=True,norm=False): # this function operate as one epoch training
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=model.to(device)
        Abs=[]
        num_corr=0
        len_num_corr=0
        if TRAIN:
            if not loss_fn: raise ValueError('criterion must be contained!')
            if not optimizer: raise ValueError('optimizer must be contained!')
            model.train()
            for all_datas, datas, labels in dataloader:
                all_data, data, label= all_datas.to(device), datas.to(device),labels.to(device)
                data=data.type(torch.float32)
                all_data=all_data.type(torch.float32)
                label=label.type(torch.float32)
                optimizer.zero_grad()
                if model.dataset=='multie':
                    output=model(data,all_data)
                elif model.dataset=='single':
                    output=model(data)
                else:
                    raise ValueError('not available model dataset type')
                pred_probab = nn.Softmax(dim=1)(output)
                y_pred = pred_probab.argmax(1)
                op=output[:,1]
                loss = loss_fn(op,label.to(device))
                loss.backward()
                optimizer.step()
        
                Abs.append(loss.item())
            
                Class=output.data.max(1)[1]
                corr=y_pred.eq(label.to(device).data).sum().item() 
                num_corr+=corr
                len_num_corr+=len(y_pred)
            abs_loss=np.array(Abs).mean()
    
            return model, abs_loss, num_corr/len_num_corr
        else:
            model.eval()
            with torch.no_grad():
                for all_datas, datas, labels in dataloader:
                    all_data, data, label= all_datas.to(device), datas.to(device),labels.to(device)
                    data=data.type(torch.float32)
                    all_data=all_data.type(torch.float32)
                    label=label.type(torch.float32)
                    optimizer.zero_grad()
                    if model.dataset=='multie':
                        output=model(data,all_data)
                    elif model.dataset=='single':
                        output=model(data)
                    else:
                        raise ValueError('not available model dataset type')
                    pred_probab = nn.Softmax(dim=1)(output)
                    y_pred = pred_probab.argmax(1)
                    op=output[:,1]
                    loss = loss_fn(op,label.to(device))
                    
                    Abs.append(loss.item())
    
                    corr=y_pred.eq(label.to(device).data).sum().item() 
                    num_corr+=corr
                    len_num_corr+=len(y_pred)
                abs_loss=np.array(Abs).mean()
    
            return model, abs_loss, num_corr/len_num_corr
    
    history = {'test_abs':[],'test_acc':[]}
    test_loss=100
    test_acc=0
    
    for kk in range(repeat):
        if Use: break
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_train)))):
    
            print('Fold {}-{}'.format(kk+1,fold + 1))
    
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(data_train, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(data_train, batch_size=batch_size, sampler=val_sampler)
    
            tof_train=False
            tof_val=False
            tof_test=False
    
            for epoch in range(num_epochs):
                model,loss, acc=epoch(model, train_loader,criterion,optimizer)
                history['train_abs'].append(loss.item())
                history['train_acc'].append(acc)
                
                model,loss, acc=epoch(model, val_loader,criterion,optimizer,TRAIN=False)   
                history['val_abs'].append(loss.item())
                history['val_acc'].append(acc)
                
                model, loss, acc=epoch(model, test_loader,criterion,optimizer,TRAIN=False)
                try: 
                    min(history['test_abs'])
                    if min(history['test_abs'])>loss: #test loss
                        torch.save(model.state_dict(),"model_testmin.md")
                        test_loss=round(loss,4)
                        test_acc=round(acc,3)
                        print('--------------------------------test model saved-----------------------------------')
                except:
                    torch.save(model.state_dict(),f"model_testmin_testmin.md")
                history['test_abs'].append(loss.item())
                history['test_acc'].append(acc)
    
                buf=len(history['train_abs'])-1
                print("Epoch:{}/{} Train ABS Loss:{:.4f} Val ABS Loss:{:.4f} / Test ABS Loss:{:.4f}"
                  .format(epoch + 1,num_epochs,history['train_abs'][buf],history['val_abs'][buf],history['test_abs'][buf]))
                print('                  Train accuracy:{:.3f} Val accuracy:{:.3f} / Test accuracy:{:.3f}'
                      .format(history['train_acc'][buf],history['val_acc'][buf],history['test_acc'][buf]))
