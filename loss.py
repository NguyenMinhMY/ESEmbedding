import torch
import torch.nn as nn
import math

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=math.cos(math.pi/5)):
        super(ContrastiveLoss, self).__init__()

        # self.margin = margin
        self.margin = math.cos(math.pi/5)

        self.w = torch.nn.Parameter(torch.abs(torch.randn(1,)))
        self.b = torch.nn.Parameter(torch.randn(1,))


    def forward(self, anchors, pos_outs, neg_outs):
        # cosine similarity

        # cosine = cos_func(x0, x1)

        # loss_similarity = 1 - cosine
        # loss_dissimlarity = torch.clamp(cosine - self.margin, min=0.0)

        # loss = (1 - y)*loss_similarity + y*loss_dissimlarity
        # loss = torch.sum(loss) / x0.size()[0]

        # return loss

        # sigmoid = torch.nn.Sigmoid()

        pos_centroids = torch.stack([self.get_centroids(embs) for embs in pos_outs], dim=0) # (B,D)
        neg_outs = torch.stack(neg_outs, dim=0) # (B,4,D)

        S_pos = self.cacl_similarity(anchors, pos_centroids, dim=1) # (B,1)
        # S_pos = sigmoid(S_pos)
        loss_sim = 1 - S_pos

        S_negs = self.cacl_similarity(anchors.unsqueeze(1), neg_outs, dim=2) # (B,4,1)
        # S_negs = sigmoid(S_negs)
        margin = self.w * self.margin + self.b
        loss_dissims = torch.clamp(S_negs - margin, min=0.0)

        loss = loss_sim + torch.max(loss_dissims, dim=1).values
        return torch.sum(loss)/anchors.size()[0]

    def get_centroids(self, embeddings):
        return torch.mean(embeddings, dim=0)
    
    def cacl_similarity(self, x0 ,x1, dim):
        try:
            cosine_func = torch.nn.CosineSimilarity(dim=dim)
            sim = self.w * cosine_func(x0, x1) + self.b
            return sim
        except:
            raise f"Inputs not match dims: x0's size is {x0.size()} - x1's size is {x1.size()}"
