import numpy as np
import os
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset


def read_data():
    labeled_anomalies=pd.read_csv('labeled_anomalies.csv',index_col=0)
    path='./data/test'
    files=os.listdir(path)
    train_lenth=200
    """
    x_train1=[]
    x_train0=[]
    y_train1=[]
    y_train0=[]
    """
    x_train=[]
    y_train=[]
    for file in files:
        name=file.split('.')[0]
        if name in labeled_anomalies.index:
            if labeled_anomalies.loc[name].spacecraft=='SMAP':
                data=np.load(os.path.join(path,file))
                anomalies=eval(labeled_anomalies.loc[name].anomaly_sequences)
                for i in range(0,len(data)-train_lenth,50):
                    for anomaly in anomalies:
                        if i<anomaly[1] and i+train_lenth>anomaly[0]:
                            y_train.append([1])
                            x_train.append(data[(i):(i + train_lenth), :])
                        else:
                            y_train.append([0])
                            x_train.append(data[(i):(i + train_lenth), :])


    return np.array(x_train), np.array(y_train)


class SeriesDataset(Dataset):
    def __init__(self):
        self.x,self.y=read_data()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x=torch.from_numpy(self.x[idx])
        y=torch.from_numpy(self.y[idx])
        mask=torch.from_numpy(np.ones((len(x),1))).type(torch.uint8)
        sample={'data':x,'mask':mask,'label':y}
        return sample


def make_weights_for_balanced_classes(y):
    weight=[0.]*len(y)
    y_1=sum(y)
    for i in range(len(y)):
        if y[i]==0:
            weight[i]=0.5*float(y_1)/float(len(y))
        else:
            weight[i]=0.5
    return weight


def load_data():
    x,y=read_data()
    series_dataset=SeriesDataset()
    weights = make_weights_for_balanced_classes(y)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    dataloader = DataLoader(series_dataset, batch_size=32, num_workers=10,sampler=sampler)
    return dataloader

load_data()

