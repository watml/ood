from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import BaseDataset


class CELEBA(BaseDataset):
    def __init__(self, path, n_class=200, InD=None):
        super(CELEBA, self).__init__()

        self.path = path
        self.n_class = n_class
        self.InD = InD

        self.name = 'celeba'

        ##### Update: The following code works as of Aug. 2021, but it doesn't work now. Please prepare celeba by yourself
        # # FOR ONCE ONLY:
        # # manual download since the torch download will trigger error
        # # URL for the CelebA dataset
        # url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
        # # Path to download the dataset to
        # download_path = f'{self.path}/celeba/img_align_celeba.zip'
        #
        # # Download the dataset from google drive
        # gdown.download(url, download_path, quiet=False)

        # Get train and test split
        self.train_split = datasets.CelebA(self.path, split='train', download=True,
                                           transform=transforms.Compose(
                                               [transforms.CenterCrop(178),
                                                transforms.Resize(32),
                                                transforms.ToTensor()])
                                           )
        self.test_split = datasets.CelebA(self.path, split='test', download=True,
                                          transform=transforms.Compose(
                                              [transforms.CenterCrop(178),
                                               transforms.Resize(32),
                                               transforms.ToTensor()])
                                          )

        # Shuffle training indexes to build a validation set (see val())
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx[0:50000]

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
        return ("OOD detection on CELEBA (InD = {} )").format(self.InD)