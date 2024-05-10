import torch
from torch import nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from tqdm.notebook import tqdm

from transforms import CCTransforms
from loss import CCLoss
from metrics.evaluate import evaluate


class ContrastiveClustering(nn.Module):
    def __init__(self, n_clusters, hidden_dim, in_channels=1):
        super().__init__()
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.in_channels = in_channels

        self.instances_projector = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, hidden_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, n_clusters),
            nn.Softmax(dim=1),
        )

        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x_for_resnet = x
        if self.in_channels == 1:
            x_for_resnet = torch.cat((x, x, x), dim=1).to(self.device)

        resnet_features = self.resnet(x_for_resnet)
        hidden_features = self.instances_projector(resnet_features)
        hidden_features = torch.nn.functional.normalize(hidden_features, dim=1)
        cluster_assignment = self.cluster_projector(resnet_features)
        return hidden_features, cluster_assignment

    def get_cluster(self, x):
        x_for_resnet = x
        if self.in_channels == 1:
            x_for_resnet = torch.cat((x, x, x), dim=1).to(self.device)

        resnet_features = self.resnet(x_for_resnet)
        cluster_assignment = self.cluster_projector(resnet_features)
        return cluster_assignment.argmax(dim=1)

    def training_epoch(self, data_loader, optimizer, device, dataloader_mode="unsupervised"):
        self.train()
        total_loss = 0

        for (x1, x2), _ in tqdm(data_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            h1, c1 = self(x1)
            h2, c2 = self(x2)

            optimizer.zero_grad()
            loss = self.loss_fn(h1, c1, h2, c2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    @torch.inference_mode()
    def evaluate(self, data_loader, device, dataloader_mode="unsupervised"):
        self.eval()
        total_loss = 0

        for (x1, x2), _ in tqdm(data_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            h1, c1 = self(x1)
            h2, c2 = self(x2)

            loss = self.loss_fn(h1, c1, h2, c2)
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def get_metrics(self, dataloader, device):
        y_true, predictions = [], []
        for (x1, x2), labels  in tqdm(dataloader):
            x1 = x1.to(device)
            predictions += self.get_cluster(x1).tolist()
            y_true += labels.tolist()
        predictions = np.asarray(predictions)
        y_true = np.asarray(y_true)
        output = evaluate(y_true, predictions, 10)
        return output

    def fit(self, train_loader, val_loader, optimizer, n_epochs, device, dataloader_mode="unsupervised", path="/kaggle/working/cc"):
        self.loss_fn = CCLoss(temperature=1, batch_size=train_loader.batch_size, num_clusters=self.n_clusters, device=device, alpha=0.5)
        self.device = device

        train_loss_history, valid_loss_history = [], []
        epoch = 0
        accs, nmis, aris, fmis, bcs = [], [], [], [], []
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, optimizer, device, dataloader_mode)
            valid_loss = self.evaluate(val_loader, device, dataloader_mode)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            acc, nmi, ari, fmi, bc = self.get_metrics(val_loader, device)
            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            fmis.append(fmi)
            bcs.append(bc)

        return train_loss_history, valid_loss_history, accs, nmis, aris, fmis, bcs
