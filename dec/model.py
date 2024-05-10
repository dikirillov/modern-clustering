import torch
from torch import nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from autoencoder import AutoEncoder
from metrics.evaluate import evaluate


class DEC(nn.Module):
    def __init__(self, num_clusters=10, autoencoder=None, cluster_centers=None, alpha=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.autoencoder = autoencoder
        self.alpha = alpha
        self.cluster_centers = None
        self.criterion = nn.KLDivLoss(size_average=False)

    def p_distribution(self, q):
        p_numerator = q ** 2 / torch.sum(q, 0)
        p = (p_numerator.t() / torch.sum(p_numerator, 1)).t()
        return p.detach()

    def forward(self, x):
        z = self.autoencoder.encode(x) 
        dist = torch.pow(z.unsqueeze(1) - self.cluster_centers, 2).sum(dim=2)
        q_numerator = 1.0 / (1.0 + dist / self.alpha)
        q_numerator = torch.pow(q_numerator, (self.alpha + 1) / 2)
        q = (q_numerator.t() / torch.sum(q_numerator, 1)).t()
        return q

    @torch.inference_mode()
    def eval_kmeans(self, dataloader, device):
        y_true, hidden_vectors = [], []
        for a, b in tqdm(dataloader):
            a = a.to(device).reshape((a.shape[0], -1))
            hidden_vectors += self.autoencoder.encode(a).flatten(start_dim=1).tolist()
            y_true += b.tolist()
        hidden_vectors = np.asarray(hidden_vectors)
        y_true = np.asarray(y_true)

        labels = []
        for elem in hidden_vectors:
            labels.append(((self.cluster_centers.cpu().detach().numpy() - elem) ** 2).sum(axis=1).argmin())
        labels = np.asarray(labels)
        output = evaluate(y_true, labels, self.num_clusters)
        return output

    def fit(self, origin_dataset, train_indices, train_loader, valid_loader, num_epochs, device, path="/kaggle/working"):
        pretrain_kmeans = []

        for x, y in train_loader:
            x = x.float()
            x = x.to(device).reshape((x.shape[0], -1))
            pretrain_kmeans += self.autoencoder.encode(x).detach().cpu().tolist()

        pretrain_kmeans = np.asarray(pretrain_kmeans)
        kmeans = KMeans(n_clusters=10).fit(pretrain_kmeans)

        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).cuda()
        self.cluster_centers = torch.nn.Parameter(cluster_centers)

        acc, nmi, ari, fmi, bc = self.eval_kmeans(valid_loader, device)
        accs, nmis, aris, fmis, bcs = [], [], [], [], []
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        fmis.append(fmi)
        bcs.append(bc)

        loss_function = nn.KLDivLoss(size_average=False)
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1, momentum=0.9)
        tmp_loader = DataLoader(origin_dataset.data[train_indices], batch_size=len(train_indices), num_workers=4, pin_memory=True, shuffle=True)
        train_loss_history = []
        best_acc = acc
        for epoch in range(num_epochs):
            for x in tmp_loader:
                x = x.float() / 255.
                x = x.to(device).reshape((x.shape[0], -1))
                output = self(x)
                target = self.p_distribution(output).detach()

                loss = loss_function(output.log(), target) / output.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_history.append(loss.item())
            
            acc, nmi, ari, fmi, bc = self.eval_kmeans(valid_loader, device)
            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            fmis.append(fmi)
            bcs.append(bc)

        return train_loss_history, accs, nmis, aris, fmis, bcs
