from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from os.path import join
from scipy.stats import multivariate_normal

from datasets.base import BaseDataset


class GAUSSIAN(BaseDataset):
    """
    synthetic Gaussian dataset
    """
    def __init__(self, dsname):
        super(GAUSSIAN, self).__init__()
        self.name = dsname
        if 'gaussianm' in dsname and 'c' in dsname and '*' in dsname:
            # Example: dsname = 'gaussianm4c5*5'
            mcd = dsname.split('m')[1].split('c')
            mean = float(mcd[0])
            cd = mcd[1].split('*')
            var = float(cd[0])
            dim = int(cd[1])
            self.mean = [mean] * dim
            self.cov = np.identity(dim) * var
        else:
            if 'gaussianm' in dsname and '*' in dsname:
                # E.g. gaussianm1*2
                md = dsname.split('m')[1].split('*')
                mean = float(md[0])
                dim = int(md[1])
                self.mean = [mean]*dim
                self.cov = np.identity(dim)
            else:
                # E.g. gaussianm0_0c1_0.5_0.5_1
                mc = dsname.split('m')[1].split('c')
                mean = mc[0].split('_')
                if len(mc) == 1:
                    dim = len(mean)
                    cov = np.identity(dim)
                else:
                    cov = mc[1].split('_')
                    cov = [float(i) for i in cov]
                    if len(cov) == 4:
                        cov = [[cov[0], cov[1]], [cov[2], cov[3]]]
                mean = [float(i) for i in mean]
                self.mean = mean
                self.cov = cov

        # Get train and test split
        self.train_split = torch.from_numpy(np.random.multivariate_normal(self.mean, self.cov, (9000, 1, 1))).float()
        self.val_split = torch.from_numpy(np.random.multivariate_normal(self.mean, self.cov, (1000, 1, 1))).float()
        self.test_split = torch.from_numpy(np.random.multivariate_normal(self.mean, self.cov, (2000, 1, 1))).float()

        self.c = self.train_split.shape[1]
        self.h = self.train_split.shape[2]
        self.w = self.train_split.shape[3]

        # Other utilities
        self.mode = None
        self.length = None

    def val(self, InD):
        self.mode = 'val'
        self.length = self.val_split.shape[0]

    def train(self, InD):
        self.mode = 'train'
        self.length = self.train_split.shape[0]
        print(f"Training Set prepared, Num:{self.length}")

    def test(self, InD):
        self.mode = 'test'
        self.length = self.test_split.shape[0]
        print(f"Test Set prepared, Num:{self.length}")


    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.length


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]
        """
        Provides the i-th example.
        """

        # Load the i-th example
        if self.mode == 'test':
            log_density = np.log(multivariate_normal.pdf(self.test_split[i], mean=self.mean, cov=self.cov))
            sample = self.test_split[i], log_density

        elif self.mode == 'val':
            log_density = np.log(multivariate_normal.pdf(self.val_split[i], mean=self.mean, cov=self.cov))
            sample = self.val_split[i], log_density

        elif self.mode == 'train':
            log_density = np.log(multivariate_normal.pdf(self.train_split[i], mean=self.mean, cov=self.cov))
            sample = self.train_split[i], log_density
        else:
            raise ValueError

        # sample = i-th data, ground truth of log density of i-th data
        return sample


    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return self.c, self.h, self.w

    def __repr__(self):
        return ("OOD detection on Gaussian (InD = {} )").format(self.InD)