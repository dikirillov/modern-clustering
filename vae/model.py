import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_layer_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.encoder_layer = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_layer_dim),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.hidden_layer_dim),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.hidden_layer_dim),
        )

        self.mus = nn.Linear(in_features=self.hidden_layer_dim, out_features=self.embedding_dim)
        self.log_var = nn.Linear(in_features=self.hidden_layer_dim, out_features=self.embedding_dim)

        self.decoder_layer = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_layer_dim, out_features=self.input_dim)
        )

    def encoder(self, x):
        encoded_x = self.encoder_layer(x)
        mean, log_var = self.mus(encoded_x), self.log_var(encoded_x)
        gauss_rand = torch.randn_like(mean)
        encoded = mean + gauss_rand * torch.exp(log_var / 2)

        return encoded, mean, log_var

    def forward(self, x):
        encoded, mean, log_var = self.encoder(x)
        decoded = self.decoder_layer(encoded)

        return encoded, decoded, mean, log_var

    def calc_loss(self, x, alpha):
        encoded, decoded, mean, log_var = self.forward(x)
        kl_div = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var))
        reconstruction_loss = nn.MSELoss(reduction="sum")(x, decoded)

        return alpha * reconstruction_loss + (1 - alpha) * kl_div

    def __train(self, data_loader, optimizer, alpha, device):
        self.train()
        total_loss = 0

        for x, y in data_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss = self.calc_loss(x, alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)


    @torch.inference_mode()
    def __evaluate(self, data_loader, alpha, device):
        self.eval()
        total_loss = 0

        for x, y in data_loader:
            x = x.to(device)
            total_loss += self.calc_loss(x, alpha)
        total_loss = total_loss.item()

        return total_loss / len(data_loader)

    def fit(self, optimizer, train_loader, valid_loader, n_epochs, device, alpha=0.5, *optimizer_args, **optimizer_kwargs):
        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.__train(train_loader, optimizer, alpha, device)
            valid_loss = self.__evaluate(valid_loader, alpha, device)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
