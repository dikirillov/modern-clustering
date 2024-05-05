import torch
from torch import nn


class VAEEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2 * output_dim, kernel_size=4),
        )

    def forward(self, x):
        z = self.encoder(x)
        mu, log_var = z[:, :self.output_dim], z[:, self.output_dim:]
        return mu, log_var
