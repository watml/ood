from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import BaseDataset


class SVHN(BaseDataset):
    def __init__(self, path, n_class=10, InD=None):
        super(SVHN, self).__init__()

        self.path = path
        self.n_class = n_class
        self.InD = InD

        self.name = 'svhn'

        # Get train and test split
        self.train_split = datasets.SVHN(self.path, split='train', download=True, transform=transforms.ToTensor())
        self.test_split = datasets.SVHN(self.path, split='test', download=True, transform=transforms.ToTensor())

        # Shuffle training indexes to build a validation set (see val())
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx

        self.mode = None
        self.length = None

        self.val_idxs = None
        self.train_idxs = None
        self.test_idxs = None

    def val(self, InD):

        self.mode = 'val'

        if len(InD) > 1:
            self.InD = InD
            self.val_idxs = [idx for idx in self.shuffled_train_idx]
            self.val_idxs = self.val_idxs[int(0.9 * len(self.val_idxs)):]
            self.length = len(self.val_idxs)
        else:
            self.InD = int(InD)
            self.val_idxs = [idx for idx in self.shuffled_train_idx if self.train_split[idx][1] == self.InD]
            self.val_idxs = self.val_idxs[int(0.9 * len(self.val_idxs)):]
            self.length = len(self.val_idxs)

    def train(self, InD):

        self.mode = 'train'

        if len(InD) > 1:
            self.InD = InD
            # training examples are all normal
            self.train_idxs = [idx for idx in self.shuffled_train_idx]
            self.train_idxs = self.train_idxs[0:int(0.9 * len(self.train_idxs))]
        else:
            self.InD = int(InD)
            # training examples are all normal
            self.train_idxs = [idx for idx in self.shuffled_train_idx if self.train_split[idx][1] == self.InD]
            self.train_idxs = self.train_idxs[0:int(0.9 * len(self.train_idxs))]

        # fix the size of training set, noise from other datasets
        self.length = len(self.train_idxs)
        # shuffle the training set manually
        np.random.shuffle(self.train_idxs)
        print(f"Training Set prepared, Num:{self.length}")

    def test(self, InD):

        self.mode = 'test'

        test_idx = np.arange(len(self.test_split))

        self.test_idxs = test_idx
        self.length = len(self.test_idxs)

        print(f"Test Set prepared, Num:{self.length}")

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.length

    def __getitem__(self, i):
        #
        """
        Provides the i-th example.
        """

        # Load the i-th example
        if self.mode == 'test':
            x, y = self.test_split[self.test_idxs[i]]
            sample = x, torch.tensor(y).float()

        elif self.mode == 'val':
            x, y = self.train_split[self.val_idxs[i]]
            sample = x, torch.tensor(y).float()

        elif self.mode == 'train':
            x, y = self.train_split[self.train_idxs[i]]
            sample = x, torch.tensor(y).float()
        else:
            raise ValueError

        return sample

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return 3, 32, 32

    def __repr__(self):
        return ("OOD detection on SVHN (InD = {} )").format(self.InD)