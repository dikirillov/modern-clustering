import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=28*28):
        super().__init__()
        self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.decoder(self.encoder(x))
        return x

    def fit(self, train_loader, valid_loader, optimizer, num_epochs, device):
        train_loss_history, valid_loss_history = [], []

        for epoch in tqdm(range(num_epochs)):
            train_loss = 0
            for x, y in tqdm(train_loader):
                x = x.to(device)
                x = x.reshape((x.shape[0], -1))
                reconstruction = self(x)
                loss = self.criterion(x, reconstruction)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / len(train_loader)

            valid_loss = 0
            for x, y in tqdm(valid_loader):
                x = x.to(device)
                x = x.reshape((x.shape[0], -1))
                reconstruction = self(x)
                valid_loss += self.criterion(x, reconstruction).item() / len(valid_loader)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

            clear_output()
            plt.plot(train_loss_history, label='Train loss')
            plt.plot(valid_loss_history, label='Valid loss')
            plt.legend()
            plt.show()

        return train_loss_history, valid_loss_history
