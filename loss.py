import torch
import torch.nn as nn
import math

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.0):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin


    def forward(self, anchors, pos_outs, neg_outs):
        # cosine similarity

        # cosine = cos_func(x0, x1)

        # loss_similarity = 1 - cosine
        # loss_dissimlarity = torch.clamp(cosine - self.margin, min=0.0)

        # loss = (1 - y)*loss_similarity + y*loss_dissimlarity
        # loss = torch.sum(loss) / x0.size()[0]

        # return loss
        sigmoid = torch.nn.Sigmoid()

        pos_centroids = torch.stack([self.get_centroids(embs) for embs in pos_outs], dim=0) # (B,D)
        neg_outs = torch.stack(neg_outs, dim=0) # (B,4,D)

        S_pos = self.cacl_similarity(anchors, pos_centroids, dim=1) # (B,1)
        S_pos = sigmoid(S_pos)

        S_negs = self.cacl_similarity(anchors, neg_outs, dim=2) # (B,4,1)
        S_negs = sigmoid(S_negs)

        loss = 1 - S_pos + torch.max(S_negs, dim=1).values
        return torch.sum(loss)/anchors.size()[0]

    def get_centroids(self, embeddings):
        return torch.mean(embeddings, dim=0)
    
    def cacl_similarity(self, x0 ,x1, dim):
        try:
            cosine_func = torch.nn.CosineSimilarity(dim=dim)
            sim = cosine_func(x0, x1)
            return sim
        except:
            raise "Inputs not match dims"
