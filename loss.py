import torch
import torch.nn as nn
import math

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.0):
        super(ContrastiveLoss, self).__init__()
        # self.margin = margin
        self.margin = math.cos(math.pi / 5)

    def forward(self, x0, x1, y):
        # cosine similarity
        cos_func = torch.nn.CosineSimilarity(dim=1)
        cosine = cos_func(x0, x1)

        loss_similarity = 1 - cosine
        loss_dissimlarity = torch.clamp(cosine - self.margin, min=0.0)

        loss = (1 - y)*loss_similarity + y*loss_dissimlarity
        loss = torch.sum(loss) / x0.size()[0]

        return loss