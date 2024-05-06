import torch
from torch import nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans

from sae import SAE


class DEC(nn.Module):
    def __init__(self, representation_module=None):
        super().__init__()
        if representation_module is not None:
            self.representation_module = representation_module
            self.need_sae = False
        else:
            self.need_sae = True

    def dec_loss(self, z, mu, alpha=1):
        dist = ((z.unsqueeze(1) - mu) ** 2).sum(dim=2)

        numerator = (1 + dist / alpha)
        denominator = numerator.sum(dim=1) ** (-(alpha + 1) // 2)
        numerator = numerator ** (-(alpha + 1) // 2)
        q = numerator / denominator.unsqueeze(1)
        f = q.sum(dim=1)
        p = (q ** 2 / f.unsqueeze(1)) / (q ** 2 / f.unsqueeze(1)).sum(dim=1).unsqueeze(1)
        loss = nn.functional.kl_div(p, q)
        return loss

    def training_epoch(self, data_loader, optimizer, device, dataloader_mode="unsupervised"):
        self.train()
        total_loss = 0
        for x in tqdm(data_loader):
            if dataloader_mode != "unsupervised":
                x = x[0]
            x = x.to(device)

            optimizer.zero_grad()
            encoded = self.representation_module.encode(x).flatten(start_dim=1)
            loss = self.dec_loss(encoded, self.cluster_centers)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    @torch.inference_mode()
    def evaluate(self, data_loader, device, dataloader_mode="unsupervised"):
        self.eval()
        total_loss = 0

        for x in tqdm(data_loader):
            if dataloader_mode != "unsupervised":
                x = x[0]
            x = x.to(device)

            encoded = self.representation_module.encode(x).flatten(start_dim=1)
            loss = self.dec_loss(encoded, self.cluster_centers)
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def fit(self, 
            n_clusters, in_channels, 
            train_loader, val_loader, optimizer, 
            n_epochs, device, dataloader_mode="unsupervised", 
            *optimizer_args, **optimizer_kwargs
        ):
        if self.need_sae:
            self.representation_module = SAE(in_channels).fit(
                train_loader, val_loader, 
                torch.optim.Adam, nn.MSELoss(reduction="sum"), 
                5, device, dataloader_mode, 
                lr=3e-4
            )

        pretrain_kmeans = []
        for x in train_loader:
            if dataloader_mode != "unsupervised":
                x = x[0]
            x = x.to(device)
            pretrain_kmeans += self.representation_module.encode(x).flatten(start_dim=1).tolist()
            break

        pretrain_kmeans = np.asarray(pretrain_kmeans)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(pretrain_kmeans)
        self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True)

        cur_optimizer = optimizer(list(self.representation_module.parameters()) + [self.cluster_centers], *optimizer_args, **optimizer_kwargs)
        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, cur_optimizer, device, dataloader_mode)
            valid_loss = self.evaluate(valid_loader, device, dataloader_mode)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
