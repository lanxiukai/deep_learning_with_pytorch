"""Training evaluation and metrics serialization primitives."""

import csv
import math
import os
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TypeAlias

import torch
from torch import nn


NumericScalar: TypeAlias = int | float
MetricHistory: TypeAlias = Mapping[str, Sequence[NumericScalar]]


class WeightedMetricAccumulator:
    """Accumulate named scalar tensors with example-count weighting."""

    def __init__(self, names: Sequence[str], *, device: torch.device):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("metric names must be non-empty and unique.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("metric names must be non-empty strings.")
        self._totals = torch.zeros(len(self.names), device=device)
        self._total_weight = 0

    def update(
        self,
        values: Sequence[torch.Tensor],
        *,
        weight: int,
    ) -> None:
        """Add one ordered collection of scalar metric tensors."""
        if weight < 1:
            raise ValueError("weight must be positive.")
        if len(values) != len(self.names):
            raise ValueError("metric values do not match configured names.")
        if any(value.ndim != 0 for value in values):
            raise ValueError("metric values must be scalar tensors.")
        stacked = torch.stack(tuple(values)).detach().to(self._totals)
        self._totals += stacked * weight
        self._total_weight += weight

    def compute(self) -> dict[str, float]:
        """Return weighted means for all configured metrics."""
        if self._total_weight == 0:
            raise RuntimeError("cannot compute metrics before an update.")
        means = (self._totals / self._total_weight).tolist()
        return dict(zip(self.names, means, strict=True))


class Accumulator:
    """
    Accumulate sum over n variables (for multiple metrics).
    
    Args:
        n: the number of variables, initialize the data with n zeros
    """
    def __init__(self, n):
        self.data = [0.0] * n  # initialize the data with n zeros

    def add(self, *args):
        """Add the arguments to the data."""
        vals = []
        for b in args:
            if torch.is_tensor(b):
                b = b.detach()
                if b.dim() == 0:
                    vals.append(b.item())
                else:
                    vals.append(b.float().sum().item())
            else:
                vals.append(float(b))
        self.data = [a + v for a, v in zip(self.data, vals)]

    def reset(self):
        """Reset the data to n zeros."""
        self.data = [0.0] * len(self.data)  # reset the data to n zeros

    def __getitem__(self, idx):  # double underscores getitem: get the data at the index
        """Get the data at the index."""
        return self.data[idx]  # return the data at the index


def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.

    Args:
        y_hat: the predicted value (batch_size, num_outputs) or (batch_size,)
        y: the true value (batch_size,)
    Returns:
        the number of correct predictions
    """
    # len(y_hat.shape) > 1 and y_hat.shape[1] > 1: means y_hat is a matrix
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat_idx = y_hat.argmax(axis=1)  # y_hat_idx: (batch_size,), obtain the index of the maximum probability
        cmp = y_hat_idx.type(y.dtype) == y  # cmp: (batch_size,), True or False
        return float(cmp.type(y.dtype).sum())  # sum the True values, convert to float
    return 0.0  # return 0.0 if y_hat is not a matrix


def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.

    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    """
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            metric.add(accuracy(net(features), labels), labels.numel())  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    Evaluate the accuracy of the model on the given dataset using GPU.
    
    Args:
        net: the model
        data_iter: the data iterator
        device: the device to use (Default: None)
    Returns:
        The accuracy of the model on the given dataset using GPU
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            # Get the device of the first parameter of the net
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            if isinstance(features, list):
                # Required for BERT fine-tuning (to be introduced later)
                device_features = [feature.to(device) for feature in features]
            else:
                device_features = features.to(device)
            device_labels = labels.to(device)
            metric.add(accuracy(net(device_features), device_labels), device_labels.numel())
    return metric[0] / metric[1]  # Return the accuracy


def evaluate_loss(net, data_iter, loss):
    """
    Evaluate the model's loss on the given dataset.
    
    Args:
        net: the model
        data_iter: the data iterator
        loss: the loss function
    Returns:
        the average loss
    """
    metric = Accumulator(2)  # loss_sum, num_samples
    for features, labels in data_iter:
        out = net(features)
        labels = labels.reshape(out.shape)
        l = loss(out, labels)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def as_list(values: Sequence[NumericScalar] | None) -> list[NumericScalar]:
    """Return an owned numeric metric list, preserving sequence order."""
    return [] if values is None else list(values)


def has_any_finite(values: Sequence[NumericScalar]) -> bool:
    """Return True if the list contains at least one finite numeric value."""
    return any(math.isfinite(value) for value in values)


def align_metrics_for_csv(metrics: MetricHistory) -> dict[str, list[NumericScalar]]:
    """
    Align a metrics dict to a common length by padding shorter lists with NaN.
    This makes CSV saving robust even when some optional metrics are missing.
    """
    if not metrics:
        return {}
    lengths = [len(as_list(v)) for v in metrics.values()]
    n = max(lengths) if lengths else 0
    aligned: dict[str, list[NumericScalar]] = {}
    for k, v in metrics.items():
        lst = as_list(v)
        if len(lst) < n:
            lst = lst + [math.nan] * (n - len(lst))
        else:
            lst = lst[:n]
        aligned[k] = lst
    return aligned


def align_and_drop_all_nan_rows(
    metrics: MetricHistory,
    *,
    exclude_keys: set[str] | None = None,
) -> dict[str, list[NumericScalar]]:
    """
    Align a metrics dict to a common length, then DROP rows where all "value" columns
    are non-finite (NaN/inf).

    This is mainly for step-level metrics where we intentionally store NaN as a
    placeholder for steps that skip metric computation (e.g. `log_every_steps`).
    Keeping those NaN rows makes CSVs huge and confusing.

    Args:
        metrics: key -> list values.
        exclude_keys: keys that are treated as coordinates/metadata (not used to
            decide whether a row is "all-NaN"). Defaults to {"step","epoch","step_in_epoch"}.

    Returns:
        A NEW metrics dict with aligned and filtered rows.
    """
    if not metrics:
        return {}

    exclude = exclude_keys or {"step", "epoch", "step_in_epoch"}
    aligned = align_metrics_for_csv(metrics)
    if not aligned:
        return {}

    # Determine which keys are "value columns" we use to decide keep/drop.
    value_keys = [k for k in aligned.keys() if k not in exclude]
    if not value_keys:
        return aligned

    # All lists should now be the same length.
    first_key = next(iter(aligned.keys()))
    n = len(as_list(aligned[first_key]))
    if n <= 0:
        return aligned

    keep_mask: list[bool] = []
    for i in range(n):
        keep_mask.append(any(math.isfinite(aligned[k][i]) for k in value_keys))

    # Fast path: nothing to drop
    if all(keep_mask):
        return aligned

    filtered: dict[str, list[NumericScalar]] = {}
    for k, v in aligned.items():
        lst = as_list(v)[:n]
        filtered[k] = [lst[i] for i, keep in enumerate(keep_mask) if keep]
    return filtered


def save_metrics_csv(metrics: MetricHistory, path: str | PathLike[str]) -> None:
    """
    Save a metrics dict (key -> list of values) to CSV.

    All lists must have the same length. Values are written as-is (via str()).
    """
    if not metrics:
        raise ValueError("save_metrics_csv: metrics is empty.")

    keys = list(metrics.keys())
    n = len(metrics[keys[0]])
    for k in keys[1:]:
        if len(metrics[k]) != n:
            raise ValueError(
                f"save_metrics_csv: length mismatch for '{k}', expected {n} got {len(metrics[k])}."
            )

    path_str = os.fspath(path)
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    with open(path_str, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([metrics[k][i] for k in keys])
