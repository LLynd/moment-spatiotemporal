import abc
import torch
import torchvision.transforms.functional as TF
import logging

from utils.misc import min_max_scale

log = logging.getLogger(__name__)


class BaseMetric(abc.ABC, torch.nn.Module):
    """Base class for metrics."""

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.batch_input: torch.Tensor | None = None
        self.batch_output: torch.Tensor | None = None
        self.batch_misc: dict = dict()

    @abc.abstractmethod
    def compute_and_log(self, fabric, log_prefix=""):
        """Compute metric from intermediate states and log results."""
        pass

    def map_scale_and_shape(self, x):
        # get shapes
        B, C, H, W = x.shape

        # scale to [0, 1] range
        x = min_max_scale(x)

        # map to uint8 and then back to float32
        x = TF.convert_image_dtype(x, torch.uint8) / 255.0

        # repeat channels to have 3
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        return x

    def forward(
        self,
        batch_input: torch.Tensor,
        batch_output: torch.Tensor,
        batch_misc: dict,
    ):
        pass

    def precondition(self, conditions: dict):
        pass

    def step(
        self,
        batch_input: torch.Tensor = None,
        batch_output: torch.Tensor = None,
        batch_misc: dict = None,
    ):
        self.forward(
            batch_input if batch_input is not None else self.batch_input,
            batch_output if batch_output is not None else self.batch_output,
            batch_misc if batch_misc is not None else self.batch_misc,
        )

        self.batch_input = None
        self.batch_output = None
        self.batch_misc = dict()

    def register_batch_input(self, batch_input: torch.Tensor):
        self.batch_input = batch_input

    def register_batch_output(self, batch_output: torch.Tensor):
        self.batch_output = batch_output

    def register_batch_misc(self, batch_misc: dict):
        self.batch_misc.update(batch_misc)


class MetricsList(BaseMetric):
    """Class to store a list of metrics."""

    def __init__(self, metrics: list[BaseMetric]):
        super(MetricsList, self).__init__()
        self.metrics = metrics

    def compute_and_log(self, fabric, log_prefix=""):
        for metric in self.metrics:
            metric.compute_and_log(fabric, log_prefix=log_prefix)

    def precondition(self, conditions: dict):
        for metric in self.metrics:
            metric.precondition(conditions)

    def step(
        self,
        batch_input: torch.Tensor = None,
        batch_output: torch.Tensor = None,
        batch_misc: dict = None,
    ):
        for metric in self.metrics:
            metric.step(batch_input, batch_output, batch_misc)

    def register_batch_input(self, batch_input: torch.Tensor):
        for metric in self.metrics:
            metric.register_batch_input(batch_input)

    def register_batch_output(self, batch_output: torch.Tensor):
        for metric in self.metrics:
            metric.register_batch_output(batch_output)

    def register_batch_misc(self, batch_misc: dict):
        for metric in self.metrics:
            metric.register_batch_misc(batch_misc)

    def forward(
        self,
        batch_input: torch.Tensor,
        batch_output: torch.Tensor,
        batch_misc: dict,
    ):
        for metric in self.metrics:
            log.info(f"Accumulating {metric.module}")
            metric.forward(batch_input, batch_output, batch_misc)