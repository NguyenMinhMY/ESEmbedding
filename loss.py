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


    def forward(self, signal_list, batch_size):
        '''
        1-Pair Loss - Cosine

        cos_func = torch.nn.CosineSimilarity(dim=1)
        cosine = cos_func(x0, x1)

        loss_similarity = 1 - cosine
        loss_dissimlarity = torch.clamp(cosine - self.margin, min=0.0)

        loss = (1 - y)*loss_similarity + y*loss_dissimlarity
        loss = torch.sum(loss) / x0.size()[0]

        return loss
        '''

        '''
        N-pair loss - Cosine + margin

        pos_centroids = torch.stack([self.get_centroids(embs) for embs in pos_outs], dim=0) # (B,D)
        neg_outs = torch.stack(neg_outs, dim=0) # (B,4,D)

        S_pos = self.cacl_similarity(anchors, pos_centroids, dim=1) # (B,1)
        loss_sim = 1 - S_pos

        S_negs = self.cacl_similarity(anchors.unsqueeze(1), neg_outs, dim=2) # (B,4,1)
        margin = self.w * self.margin + self.b
        loss_dissims = torch.clamp(S_negs - margin, min=0.0)

        loss = loss_sim + torch.max(loss_dissims, dim=1).values
        return torch.sum(loss)/anchors.size()[0]
        '''

        # N-pair loss - Cosine + logarit
        loss = 0.0
        for idx in range(0, batch_size):
            anchor_idx = 9 * idx
            anchor = signal_list[anchor_idx]

            pos_range = (anchor_idx + 1, anchor_idx + 5)
            pos_samples = signal_list[pos_range[0] : pos_range[1]]
            pos_centroid = torch.mean(pos_samples, dim=0)

            neg_range = (pos_range[1], pos_range[1] + 4)
            neg_samples = signal_list[neg_range[0] : neg_range[1]]

            

            S_pos = self.cacl_similarity(anchor, pos_centroid, dim=-1)
            S_pos = torch.exp(S_pos)

            S_negs = self.cacl_similarity(anchor, neg_samples, dim=1)
            S_neg = torch.sum(torch.exp(S_negs))

            loss += - torch.log(S_pos / (S_pos + S_neg))

        return loss / batch_size
    
    def cacl_similarity(self, x0 ,x1, dim):
        try:
            cosine_func = torch.nn.CosineSimilarity(dim=dim)
            sim = self.w * cosine_func(x0, x1) + self.b
            return sim
        except:
            raise f"Inputs not match dims: x0's size is {x0.size()} - x1's size is {x1.size()}"
