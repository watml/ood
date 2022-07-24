from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import BaseDataset


class FAKE(BaseDataset):
    def __init__(self, ds_name, InD):

        super(FAKE, self).__init__()

        if InD in ['cifar10', 'svhn']:
            self.shapei = (3, 32, 32)
        elif InD in ['mnist', 'fmnist']:
            self.shapei = (1, 28, 28)

        (c, h, w) = self.shapei
        self.length = 10000 # 10000 test fake images

        self.name = ds_name
        if ds_name == 'random':
            self.test_split = torch.randn(self.length, c, h, w)
            self.test_idxs = np.arange(self.length)
        elif ds_name == 'const':
            self.test_split = torch.ones(self.length, c, h, w) * torch.randn(self.length, 1, 1, 1)
            # self.test_split = torch.ones(self.length, c, h, w) * 0.5
            self.test_idxs = np.arange(self.length)

    def val(self, InD):
        self.mode = 'val'

    def train(self, InD):
        self.mode = 'train'

    def test(self, InD):
        self.mode = 'test'
        print(f"Test Set prepared, Num:{self.length}")

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.length


    def __getitem__(self, i):
        """
        Provides the i-th example.
        """

        # Load the i-th example
        if self.mode == 'test':
            x = self.test_split[self.test_idxs[i]]
            sample = x, -1
        else:
            raise ValueError
        return sample



    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return self.shapei

    def __repr__(self):
        return ("OOD detection on FAKE (InD =  {} )").format(self.InD)