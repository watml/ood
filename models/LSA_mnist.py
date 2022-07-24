from functools import reduce
from operator import mul
from typing import Tuple
import torch
import torch.nn as nn

# LSA
from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import UpsampleBlock 

# flows
from models.transform_realnvp import TinvREALNVP
import torch.nn.functional as F


class Encoder(BaseModule):
    """
    MNIST model encoder.
    same as LSA
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape
        
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            # PrintLayer(), # Add Print layer for debug
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
        )

        self.deepest_shape = (64, h // 4, w // 4)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """

        h = x  
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o

class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.
        same as LSA

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )


    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h
        return o


class LSA_MNIST(BaseModule):
    """
    LSA model for MNIST.
    """
 

    def __init__(self,  input_shape, code_length, num_blocks = 1, hidden_size = 2048, est_name = None):
        """
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors of LSA.
        :param num_blocks: number of blocks in realnvp
        :param hidden_size: size of hidden units
        :param est_name: density estimator {"REALNVP"}

        """
        super(LSA_MNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name

        self.coder_name = 'LSA'

        if est_name == None:  
            self.name = 'LSA'
        else:
            self.name = f'LSA_{est_name}'
        
        print(f'{self.name} Model Initialization')


        # Build encoder
        self.encoder = Encoder(input_shape=input_shape, code_length=code_length)

        # Build decoder
        self.decoder = Decoder(code_length=code_length, deepest_shape=self.encoder.deepest_shape, output_shape=input_shape)

        # Build estimator
        if est_name == "REALNVP":
            self.estimator = TinvREALNVP(num_blocks, code_length, hidden_size)
        # No estimator
        elif est_name == None:
            self.estimator = None


    def forward(self, x):
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """

        # Produce representations
        z = self.encoder(x)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)


        if self.est_name in ['REALNVP']:
            # z: latent vector of input x, density estimation is based on reduced-dimensional space. s=T_inverse(z)
            s, log_jacob_T_inverse = self.estimator(z)
            return x_r, z, s, log_jacob_T_inverse