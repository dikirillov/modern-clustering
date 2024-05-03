import torch
from torch import nn
from tqdm.notebook import tqdm

from sae_basic_block import SAEBasicBlock

class SAE(nn.Module):
    def __init__(self, dimentions):
        super().__init__()
        self.blocks = [SAEBasicBlock(dimentions[i], dimentions[i + 1]) for i in range(len(dimentions) - 1)]

    def encode(self, x, num_layers=None):
        if num_layers is None:
            num_layers = len(self.blocks)
        for block in self.blocks[:num_layers]:
            x = block.encode(x)
        return x

    def decode(self, x, num_layers=None):
        if num_layers is None:
            num_layers = len(self.blocks)
        for block in self.blocks[::-1][-num_layers:]:
            x = block.decode(x)
        return x

    def training_epoch(self, data_loader, optimizer, criterion, device, dataloader_mode="unsupervised", *optimizer_args, **optimizer_kwargs):
        self.train()
        total_loss = 0

        for i in range(len(self.blocks)):
            cur_optimizer = optimizer(self.blocks[i].parameters(), *optimizer_args, **optimizer_kwargs)
            for x in tqdm(data_loader):
                if dataloader_mode != "unsupervised":
                    x = x[0]
                x = x.to(device)

                cur_optimizer.zero_grad()
                encoded = self.encode(x, i + 1)
                decoded = self.decode(encoded, i + 1)
                loss = criterion(x, decoded)
                loss.backward()
                cur_optimizer.step()

                total_loss += loss.item()

        return total_loss / len(data_loader)

    @torch.inference_mode()
    def evaluate(self, data_loader, criterion, device, dataloader_mode="unsupervised"):
        self.eval()
        total_loss = 0

        for x in data_loader:
            if dataloader_mode != "unsupervised":
                x = x[0]
            x = x.to(device)

            encoded = self.encode(x)
            decoded = self.decode(encoded)
            loss = criterion(x, decoded)
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def fit(self, train_loader, val_loader, optimizer, criterion, n_epochs, device, dataloader_mode="unsupervised", *optimizer_args, **optimizer_kwargs):
        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, optimizer, criterion, device, dataloader_mode, *optimizer_args, **optimizer_kwargs)
            valid_loss = self.evaluate(valid_loader, criterion, device, dataloader_mode)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
