from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np


class AdBench_Dataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    urls = {
        'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1',
        'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
        'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
        'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1',
        'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
        'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1'
    }

    def __init__(self, root: str, dataset_name: str, train=True, feature_range = (0,1), random_state=None, download=False,train_option=None):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name + '.npz'
        self.data_file = self.root / self.file_name


        mat = np.load(self.data_file, allow_pickle=True)

        X = mat['X']
        y = mat['y'].ravel()
        if train_option == 'glow':
            num, dim = X.shape
            sqrt_dim = np.sqrt(dim)
            # if sqrt_dim < 6:
            #     sqrt_dim = 6
            if sqrt_dim < 4:
                sqrt_dim = 4
            else:
                if sqrt_dim != int(sqrt_dim):
                    sqrt_dim = int(sqrt_dim) + 1
                if sqrt_dim % 2 == 1:
                    sqrt_dim = sqrt_dim + 1
            adjust_dim = sqrt_dim**2
            if adjust_dim > dim:
                padding = np.zeros(shape=(num,adjust_dim-dim),dtype=float)
                X_concat = np.concatenate([X, padding], axis = 1)
                X = X_concat

        idx_total = np.array(range(X.shape[0]))
        idx_norm = y == 0
        idx_out = y == 1

        # 60% data for training and 40% for testing; keep outlier ratio
        # all data is train data.
        X_train_norm = X[idx_norm]
        X_train_out = X[idx_out]
        
        y_train_norm = y[idx_norm]
        y_train_out = y[idx_out]
        
        idx_train_norm = idx_total[idx_norm]
        idx_train_out = idx_total[idx_out]
        
        
        X_train = np.concatenate((X_train_norm, X_train_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        idx_train = np.concatenate((idx_train_norm, idx_train_out))


        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler(feature_range = feature_range).fit(X_train)
        X_train_scaled = minmax_scaler.transform(X_train)
        if train_option == 'glow':
            X_train_scaled = np.reshape(X_train_scaled,newshape=(num,1,int(sqrt_dim), int(sqrt_dim)))
        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
            self.indices = torch.tensor(idx_train, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target,indices = self.data[index], int(self.targets[index]), int(self.semi_targets[index]),int(self.indices[index])

        #return sample, target, semi_target, index
        #return sample, target, index,indices
        return sample, target, indices  ####deepSVDD때문에 2번째것에서 변경

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        # download file
        download_url(self.urls[self.dataset_name], self.root, self.file_name)

        print('Done!')
