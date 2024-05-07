import torch
from torch import nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans

from sae import SAE
from metrics.evaluate import evaluate


class DEC(nn.Module):
    def __init__(self, representation_module=None):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        if representation_module is not None:
            self.representation_module = representation_module
            self.need_sae = False
        else:
            self.need_sae = True

    def dec_loss(self, z, mu, alpha=1):
        dist = torch.pow(z.unsqueeze(1) - mu, 2).sum(dim=2)

        q_numerator = torch.pow(1 + dist / alpha, -(alpha + 1) / 2)
        q = q_numerator / torch.sum(q_numerator, dim=1, keepdim=True)
        f = q.sum(0, keepdim=True)
        p_unnorm = torch.pow(q, 3) / f
        p = p_unnorm / p_unnorm.sum(1, keepdim=True)

        p_numerator = q ** 2 / torch.sum(q, 0)
        p = (p_numerator.T / torch.sum(p_numerator, 1)).T
        loss = self.criterion(q.log(), p)
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

    def eval_kmeans(self, dataloader, device):
        y_true, hidden_vectors = [], []
        for a, b in dataloader:
            a = a.to(device)
            hidden_vectors += self.representation_module.encode(a).flatten(start_dim=1).tolist()
            y_true += b.tolist()
        hidden_vectors = np.asarray(hidden_vectors)
        y_true = np.asarray(y_true)

        labels = []
        for elem in hidden_vectors:
            labels.append(((self.cluster_centers.detach().numpy() - elem) ** 2).sum(axis=1).argmin())
        labels = np.asarray(labels)
        return evaluate(y_true, labels, 10)

    def fit(self,
            n_clusters, in_channels,
            train_loader, val_loader, optimizer,
            n_epochs, device, dataloader_mode="unsupervised",
            centroids=None,
            *optimizer_args, **optimizer_kwargs):
        target_size = train_loader.dataset[0][0].shape[0]
        if self.need_sae:
            self.representation_module = SAE(in_channels).fit(
                train_loader, val_loader,
                torch.optim.Adam, nn.MSELoss(reduction="sum"),
                5, device, dataloader_mode,
                lr=3e-4
            )

        if centroids is None:
            pretrain_kmeans = []
            print("kmeans preparing")
            for x in tqdm(train_loader):
                if dataloader_mode != "unsupervised":
                    x = x[0]
                x = x.to(device)
                pretrain_kmeans += self.representation_module.encode(x).flatten(start_dim=1).tolist()

            pretrain_kmeans = np.asarray(pretrain_kmeans)
            kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(pretrain_kmeans)
            print("kmeans done")
            self.cluster_centers = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True)
            self.initialed_clusters = kmeans.cluster_centers_
        else:
            self.cluster_centers = Variable(torch.from_numpy(centroids), requires_grad=True)
        acc, nmi, ari, fmi = self.eval_kmeans(valid_loader, device)

        cur_optimizer = optimizer([self.cluster_centers], *optimizer_args, **optimizer_kwargs)
        accs, nmis, aris, fmis = [], [], [], []
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        fmis.append(fmi)

        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, cur_optimizer, device, dataloader_mode)
            acc, nmi, ari, fmi = self.eval_kmeans(valid_loader, device)
            valid_loss = self.evaluate(valid_loader, device, dataloader_mode)
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            fmis.append(fmi)

        return train_loss_history, valid_loss_history, accs, nmis, aris, fmis
