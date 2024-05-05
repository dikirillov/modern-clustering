import torch
from torch import nn


class VAEDecoder(nn.Module):
    def __init__(self, output_channels, input_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 64, kernel_size=4),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        return self.decoder(x)
