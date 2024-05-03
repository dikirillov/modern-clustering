import torch
from torch import nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans

from sae import SAE


class DEC(nn.Module):
    def __init__(self):
        super().__init__()

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
            encoded = self.sae.encode(x)
            loss = self.dec_loss(encoded, self.cluster_centers)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    @torch.inference_mode()
    def evaluate(self, data_loader, device, dataloader_mode="unsupervised"):
        self.eval()
        total_loss = 0

        for x in data_loader:
            if dataloader_mode != "unsupervised":
                x = x[0]
            x = x.to(device)

            encoded = self.sae.encode(x)
            loss = self.dec_loss(encoded, self.cluster_centers)
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def fit(self, n_clusters, dimentions, train_loader, val_loader, optimizer, n_epochs, device, dataloader_mode="unsupervised", *optimizer_args, **optimizer_kwargs):
        target_size = train_loader.dataset[0][0].shape[0]
        self.sae = SAE(dimentions).fit(train_loader, val_loader, torch.optim.Adam, nn.MSELoss(reduction="sum"), 5, device, dataloader_mode)
        # self.sae = SAE(dimentions)

        pretrain_kmeans = train_loader.dataset.data.view(-1, target_size).type(torch.FloatTensor)
        pretrain_kmeans = self.sae.encode(pretrain_kmeans)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(pretrain_kmeans.detach().numpy())
        self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True)

        cur_opimizer = optimizer(list(self.sae.parameters()) + [self.cluster_centers], lr=0.1)

        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, cur_opimizer, device, dataloader_mode)
            valid_loss = self.evaluate(valid_loader, device, dataloader_mode)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
