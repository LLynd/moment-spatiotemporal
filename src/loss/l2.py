import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()
        
    def __str__(self):
        return LossType.L2.value
    
    def forward(self, cur_vector, src_vector):
        return (cur_vector - src_vector) ** 2
