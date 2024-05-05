import torch
from torch import nn
from tqdm.notebook import tqdm

from vae_encoder import VAEEncoder
from vae_decoder import VAEDecoder
from loss import VAELoss


class VAE(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(input_channels, hidden_dim)
        self.decoder = VAEDecoder(input_channels, hidden_dim)

    def reparametrisation_trick(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(log_var / 2)

    def forward(self, x):
        mus, log_vars = self.encoder(x)
        hidden = self.reparametrisation_trick(mus, log_vars)
        decoded = self.decoder(hidden)
        return decoded, mus, log_vars

    def training_epoch(self, data_loader, optimizer, alpha, device):
        self.train()
        total_loss = 0

        for x, y in tqdm(data_loader):
            x = x.to(device)

            optimizer.zero_grad()
            loss = self.criterion(x, *self(x))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)


    @torch.inference_mode()
    def evaluate(self, data_loader, alpha, device):
        self.eval()
        total_loss = 0

        for x, y in tqdm(data_loader):
            x = x.to(device)
            loss = self.criterion(x, *self(x))
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def fit(self, optimizer, train_loader, valid_loader, n_epochs, device, alpha=0.5):
        train_loss_history, valid_loss_history = [], []
        epoch = 0
        self.criterion = VAELoss(alpha)
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, optimizer, alpha, device)
            valid_loss = self.evaluate(valid_loader, alpha, device)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
