from torch import nn

class SAEBasicBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_features + out_features) // 2

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
