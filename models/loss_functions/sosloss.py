import torch
import torch.nn as nn

from models.loss_functions.flow_loss import FlowLoss

class SOSLoss(nn.Module):
    """
    loss of flow, simply the negative log likelihood of input variable
    """
    def __init__(self):
        super(SOSLoss, self).__init__()
        # Set up loss modules
        self.autoregression_loss_fn = FlowLoss()

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None
        
        self.total_loss = None

    def forward(self, s, nagtive_log_jacob, average = True):
        # Compute pytorch loss

        arg_loss, _ , logdet = self.autoregression_loss_fn(s,nagtive_log_jacob, average)

        tot_loss = arg_loss  

        # Store numerical
        self.autoregression_loss = arg_loss
        self.logdet = logdet
        self.total_loss = tot_loss

        return tot_loss