import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class CosineLoss(BaseLoss):
    def __init__(self):
        super(CosineLoss, self).__init__()
    
    def __str__(self):
        return LossType.COS.value
    
    def forward(self, cur_vector, src_vector):
        cosine_sim = F.cosine_similarity(cur_vector, src_vector, dim=-1)
        loss_val = 1 - cosine_sim
        
        # Broadcast to match input shape
        return loss_val.unsqueeze(-1).expand_as(cur_vector)
