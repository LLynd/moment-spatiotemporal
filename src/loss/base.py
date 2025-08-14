import enum
import abc
import torch


class LossType(enum.Enum):
    L1 = "L1"
    L2 = "L2"
    KL_DIV = "KL_DIV"
    COS = "COS"
    SINKHORN = "SINKHORN"
    TIME_SERES_MSE = "TIME_SERIES_MSE"


class BaseLoss(abc.ABC, torch.nn.Module):
    def __init__(self):
        """
        Loss is always unreduced in order to allow for thresholding.
        We want to compute the loss for all inputs but only have non-zero gradient for those above treshold.
        That is why the treshold is not passed to the forward function.
        """
        super(BaseLoss, self).__init__()

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def forward(self, cur_vector, src_vector) -> torch.Tensor:
        pass
