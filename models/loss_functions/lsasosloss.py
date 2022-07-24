import torch
import torch.nn as nn

from models.loss_functions.reconstruction_loss import ReconstructionLoss
from models.loss_functions.flow_loss import FlowLoss

class LSASOSLoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, lam=1):
        super(LSASOSLoss, self).__init__()

        self.lam = lam

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = FlowLoss()

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None

        self.total_loss = None

    def forward(self, x, x_r, s, nagtive_log_jacob, average = True):
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r, average)
        arg_loss, nlog_probs, nlog_jacob_d = self.autoregression_loss_fn(s, nagtive_log_jacob, average)

        tot_loss = rec_loss + self.lam * arg_loss

        # Store numerical
        self.reconstruction_loss = rec_loss
        self.autoregression_loss = arg_loss

        
        self.total_loss = tot_loss

        return tot_loss