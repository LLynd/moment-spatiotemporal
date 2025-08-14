import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, LossType


class TimeSeriesMSELoss(BaseLoss):
    def __init__(self, reduce_time=True, reduce_features=True):
        """
        MSE loss for time series reconstructions
        
        Args:
            reduce_time: Whether to average over the time dimension
            reduce_features: Whether to average over feature dimensions
        """
        super(TimeSeriesMSELoss).__init__()
        self.reduce_time = reduce_time
        self.reduce_features = reduce_features
        
    def __str__(self) -> str:
        return LossType.TIME_SERES_MSE.value
    
    def forward(self, pred_series, target_series) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target time series
        
        Args:
            pred_series: (batch_size, seq_len, features)
            target_series: (batch_size, seq_len, features)
            
        Returns:
            loss: Unreduced loss tensor with same dimensions as input
        """
        # Compute element-wise squared error
        sq_err = (pred_series - target_series) ** 2
        
        # Apply reduction while keeping dimensions
        if self.reduce_time:
            sq_err = sq_err.mean(dim=1, keepdim=True)
        if self.reduce_features:
            sq_err = sq_err.mean(dim=2, keepdim=True)
            
        # Broadcast back to original shape
        return sq_err.expand_as(pred_series)
