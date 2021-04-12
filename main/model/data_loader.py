import random
import os

import pandas as pd
import numpy as np
import pickle

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model.deepst import MinMaxNormalization

def get_depends(params):
    T = 24
    TrendInterval = 7
    PeriodInterval = 1
    offset_frame = pd.DateOffset(minutes=24 * 60 // T)
    depends = list(range(1, params.len_close+1)) + [PeriodInterval * T * j for j in range(1, params.len_period+1)] + [TrendInterval * T * j for j in range(1, params.len_trend+1)]
    return np.array(depends)

class BikeTrianDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_file, params):
        self.params = params
        self.depends = get_depends(params)
        f = h5py.File(data_file, 'r')
        test_pos = params.test_timesteps + self.depends.max()
        self.data = f['data'][()][:-test_pos]
        self.timestamps = f['date'][()][:-test_pos]

    def __len__(self):
        return len(self.data)-self.depends.max()

    def __getitem__(self, index):
        y_pos = self.depends.max()+index
        y = self.data[y_pos]
        x = self.data[y_pos-self.depends]
        return x, y

class BikeTestDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_file, params):
        self.params = params
        self.depends = get_depends(params)
        f = h5py.File(data_file, 'r')
        test_pos = params.test_timesteps + self.depends.max()
        self.data = f['data'][()]
        self.timestamps = f['date'][()]
        
    def __len__(self):
        return self.params.test_timesteps

    def __getitem__(self, index):
        y_pos = len(self.data) - self.params.test_timesteps + index
        y = self.data[y_pos]
        x = self.data[y_pos-self.depends]
        return x, y
    
def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    dataloaders = {}
    
    # get the train_dataset
    path = os.path.join(data_dir, params.file_name)
    train_dataset = BikeTrianDataset(data_file=path, params=params)

    # print('train length:', len(train_dataset)) #train length: 4140

    # Define the indices
    indices = list(range(len(train_dataset))) # start with all the indices in training set
    split = int(0.1*len(train_dataset)) # define the split size
    # print('split', split)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, sampler=validation_sampler)

    dataloaders['train'] = train_loader
    dataloaders['val']   = validation_loader

    if 'test' in types:
        test_dataset = BikeTestDataset(data_file=path, params=params)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size)
        dataloaders['test'] = test_loader

    return dataloaders