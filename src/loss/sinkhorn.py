import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class SinkhornIVLoss(BaseLoss):
    def __init__(self, epsilon=0.01, max_iters=100, reduction='none'):
        """
        Sinkhorn Value Iteration Loss based on:
        "Bisimulation Metrics are Optimal Transport Distances, and Can be Computed Efficiently"
        
        Args:
            epsilon: Entropic regularization parameter
            max_iters: Number of Sinkhorn iterations
        """
        super(SinkhornIVLoss).__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.reduction = reduction

    def __str__(self):
        return LossType.SINKHORN.value
    
    def forward(self, cur_vector, src_vector):
        """
        Compute Sinkhorn distance between two batches of vectors.
        
        Args:
            cur_vector: (batch_size, n)
            src_vector: (batch_size, n)
            
        Returns:
            loss: (batch_size, n) tensor with broadcasted Sinkhorn distance
        """
        # Normalize vectors to probability distributions
        a = F.softmax(cur_vector, dim=-1)
        b = F.softmax(src_vector, dim=-1)
        
        # Compute pairwise cost matrix (Euclidean distance between vectors)
        # Using the same approach as in the paper's "ground metric"
        C = torch.cdist(a.unsqueeze(1), b.unsqueeze(1)).squeeze(1)
        C = C / C.max()  # Normalize for stability
        
        # Sinkhorn algorithm
        K = torch.exp(-C / self.epsilon)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        for _ in range(self.max_iters):
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = b / (torch.bmm(K.transpose(1,2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        # Compute transport plan and distance
        T = u.unsqueeze(-1) * K * v.unsqueeze(1)
        sinkhorn_dist = torch.sum(T * C, dim=(1,2))
        
        # Broadcast to original input shape
        return sinkhorn_dist.unsqueeze(-1).expand_as(cur_vector)
