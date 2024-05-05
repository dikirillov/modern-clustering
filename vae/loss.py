import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, reconstruction, mus, log_vars):
        kl_div = -0.5 * torch.sum(1 + log_vars - mus ** 2 - torch.exp(log_vars))
        reconstruction_loss = nn.MSELoss(reduction="sum")(x, reconstruction)
        return (self.alpha * reconstruction_loss + (1 - self.alpha) * kl_div) / x.shape[0]

