import torch
from torch import nn


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
