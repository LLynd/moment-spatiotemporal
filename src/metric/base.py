import abc
import torch
import logging


log = logging.getLogger(__name__)

class BaseMetric(abc.ABC, torch.nn.Module):
    """Base class for time series metrics."""

    def __init__(self):
        super(BaseMetric, self).__init__()
        # Dummy parameter to ensure module has parameters
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.batch_input: torch.Tensor | None = None
        self.batch_output: torch.Tensor | None = None
        self.batch_misc: dict = dict()

    @abc.abstractmethod
    def compute_and_log(self, fabric, log_prefix=""):
        """Compute metric from intermediate states and log results."""
        pass

    def forward(
        self,
        batch_input: torch.Tensor,
        batch_output: torch.Tensor,
        batch_misc: dict,
    ):
        """Process batch data - implement in child classes if needed"""
        pass

    def precondition(self, conditions: dict):
        """Set preconditions for metric computation"""
        pass

    def step(
        self,
        batch_input: torch.Tensor = None,
        batch_output: torch.Tensor = None,
        batch_misc: dict = None,
    ):
        """Process a batch of data"""
        self.forward(
            batch_input if batch_input is not None else self.batch_input,
            batch_output if batch_output is not None else self.batch_output,
            batch_misc if batch_misc is not None else self.batch_misc,
        )

        # Clear batch data after processing
        self.batch_input = None
        self.batch_output = None
        self.batch_misc = dict()

    def register_batch_input(self, batch_input: torch.Tensor):
        """Register input time series batch"""
        # Shape: (batch_size, num_channels, sequence_length)
        self.batch_input = batch_input

    def register_batch_output(self, batch_output: torch.Tensor):
        """Register output/reconstructed time series batch"""
        # Shape: (batch_size, num_channels, sequence_length)
        self.batch_output = batch_output

    def register_batch_misc(self, batch_misc: dict):
        """Register additional batch information"""
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
            log.info(f"Accumulating {type(metric).__name__}")
            metric.forward(batch_input, batch_output, batch_misc)
