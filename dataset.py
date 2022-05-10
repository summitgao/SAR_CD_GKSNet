import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import scipy.io


class MyDataset_source(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset_source, self).__init__()
       

        if mode == 'train':
            self.data = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Training_Data_Florence.mat'))['Training_Data']
            self.labels = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Training_Label_Florence.mat'))['Training_Label']
            self.data = self.data.transpose((0,3,1,2))

        elif mode == 'test':
            self.data = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Testing_Data_Florence.mat'))['Testing_Data']
            self.labels = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Testing_Label_Florence.mat'))['Testing_Label']
            self.data = self.data.transpose((0,3,1,2))
            
        
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]

class MyDataset_target(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset_target, self).__init__()

        if mode == 'train':
            self.data = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Training_Data_ottawa.mat'))['Training_Data']
            self.labels = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Training_Label_ottawa.mat'))['Training_Label']
            self.data = self.data.transpose((0,3,1,2))

        elif mode == 'test':
            self.data = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Testing_Data_ottawa.mat'))['Testing_Data']
            self.labels = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Testing_Label_ottawa.mat'))['Testing_Label']
            self.data = self.data.transpose((0,3,1,2))

        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


