import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class KLDivLoss(BaseLoss):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def __str__(self):
        return LossType.KL_DIV.value
    
    def forward(self, cur_vector, src_vector):
        # Add epsilon for numerical stability
        eps = 1e-8
        cur_vector = cur_vector + eps
        src_vector = src_vector + eps
        
        # Normalize to probability distributions
        cur_prob = cur_vector / cur_vector.sum(dim=-1, keepdim=True)
        src_prob = src_vector / src_vector.sum(dim=-1, keepdim=True)
        
        # Compute element-wise KL terms: p * (log(p) - log(q))
        return cur_prob * (torch.log(cur_prob) - torch.log(src_prob))
