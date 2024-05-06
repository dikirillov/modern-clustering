from torch import nn

class SAEBasicBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=None):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, stride=1, padding=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
        ) 

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
