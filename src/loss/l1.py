import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __str__(self):
        return LossType.L1.value
    
    def forward(self, cur_vector, src_vector):
        return torch.abs(cur_vector - src_vector)
