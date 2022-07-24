from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def test(self, *args):
        """
        Sets the dataset in test mode.
        """
        pass

    @abstractmethod
    def val(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass

    @abstractmethod
    def train(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Returns the shape of examples.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass

