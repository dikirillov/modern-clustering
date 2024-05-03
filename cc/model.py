import torch
from torch import nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from tqdm.notebook import tqdm

from transforms import CCTransforms
from loss import CCLoss

class CCLoss(nn.Module):
    def __init__(self, temperature, batch_size, num_clusters, device, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.device = device

        self.instance_loss_mask = self.create_mask(batch_size)
        self.cluster_loss_mask = self.create_mask(num_clusters)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def create_mask(self, n):
        mask = torch.ones((n, n))
        mask = mask.fill_diagonal_(0)
        mask = torch.cat((mask, mask))
        return torch.cat((mask, mask), dim=1).bool().to(self.device)

    def instance_loss(self, similarities, mask):
        N = mask.shape[0]
        positive_1 = torch.diag(similarities, int(N // 2))
        positive_2 = torch.diag(similarities, -int(N // 2))
        positives = torch.cat((positive_1, positive_2), dim=0).to(self.device)
        positives = positives.reshape(N, 1)
        negatives = similarities[mask].reshape(N, -1)
        # loss = (torch.exp(positives) / similarities.sum()).sum()
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positives, negatives), dim=1).to(self.device)
        loss = self.criterion(logits, labels) / N

        return loss

    def calc_log_loss(self, c):
        p = c.sum(0).view(-1).to(self.device)
        p /= p.sum()
        return (p * torch.log(p)).sum()

    def cluster_loss(self, c1, c2):
        similarity_f = nn.CosineSimilarity(dim=2)
        c = torch.cat((c1.t(), c2.t()), dim=0)
        similarities = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        loss = self.instance_loss(similarities, self.cluster_loss_mask)
        h_loss = self.calc_log_loss(c1) + self.calc_log_loss(c2)
        return loss + h_loss

    def forward(self, x1, c1, x2, c2):
        cur_batch = torch.cat((x1, x2))
        similarities = (cur_batch @ cur_batch.T / self.temperature).to(self.device)

        return self.alpha * self.instance_loss(similarities, self.instance_loss_mask) + (1 - self.alpha) * self.cluster_loss(c1, c2)


class ContrastiveClustering(nn.Module):
    def __init__(self, n_clusters, hidden_dim):
        super().__init__()
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.instances_projector = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, hidden_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_clusters),
            nn.Softmax(dim=1),
        )

        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x_for_resnet = torch.cat((x, x, x), dim=1).to(self.device)
        x_for_resnet = x
        resnet_features = self.resnet(x_for_resnet)
        hidden_features = self.instances_projector(resnet_features)
        hidden_features = torch.nn.functional.normalize(hidden_features, dim=1)
        cluster_assignment = self.cluster_projector(resnet_features)
        return hidden_features, cluster_assignment

    def get_cluster(self, x):
        # x_for_resnet = torch.cat((x, x, x), dim=1).to(self.device)
        x_for_resnet = x
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

    def fit(self, train_loader, val_loader, optimizer, n_epochs, device, dataloader_mode="unsupervised"):
        self.loss_fn = CCLoss(temperature=1, batch_size=train_loader.batch_size, num_clusters=self.n_clusters, device=device, alpha=0.5)
        self.device = device

        train_loss_history, valid_loss_history = [], []
        epoch = 0
        for epoch in tqdm(range(n_epochs), desc=f'Training {epoch}/{n_epochs}'):
            train_loss = self.training_epoch(train_loader, optimizer, device, dataloader_mode)
            valid_loss = self.evaluate(valid_loader, device, dataloader_mode)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history
