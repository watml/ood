"""based on MAF."""

import torch.nn as nn
import torch
from models.base import BaseModule

import models.flow_maf_models as fnn
import numpy as np


class TinvREALNVP(BaseModule):
    """
    Adapted from https://github.com/ikostrikov/pytorch-flows/blob/master/main.py

    Model: T-inverse , T-inverse(z) = s, where T-inverse is built by real nvp

    """

    def __init__(self, num_blocks, input_size, hidden_size):
        num_cond_inputs = None
        self.name = 'REALNVP'
        self.input_size = input_size
        device = torch.device('cuda')

        super(TinvREALNVP, self).__init__()

        modules = []
        mask = torch.arange(0, input_size) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                fnn.CouplingLayer(
                    input_size, hidden_size, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(input_size)
            ]
            mask = 1 - mask


        model = fnn.FlowSequential(*modules)

        # intialize
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        self.T_inverse = model


    def forward(self, z, mode='direct'):
        """
        Forward propagation.

        Model: T-inverse , T-inverse(z) = s, T-inverse is built by real nvp
        Input: latent vector z
        Output: s, -log_jacob of T (i.e., logjab of T-inverse)
        """
        h = z.view(-1, self.input_size)

        s, log_jacob_T_inv = self.T_inverse(h, mode=mode)

        return s, log_jacob_T_inv