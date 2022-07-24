import torch
import torch.nn as nn

class GLOWLoss(nn.Module):
    """
    loss of glow, simply the negative log likelihood of input variable
    """
    def __init__(self):
        super(GLOWLoss, self).__init__()
        # Set up loss modules
        self.autoregression_loss = None
        self.total_loss = None


    def forward(self, nll, average = True):
        # Compute pytorch loss
        if average:
            tot_loss = torch.mean(nll)
        else:
            tot_loss = nll

        # Store numerical
        self.autoregression_loss = tot_loss
        self.total_loss = tot_loss

        return tot_loss