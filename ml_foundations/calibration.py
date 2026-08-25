"""Measure whether model confidence matches observed correctness.

Accuracy asks how often argmax was right. Calibration asks whether predictions made
at a stated confidence succeed at roughly that rate. Temperature scaling changes logit
scale on a labelled calibration split and leaves class order unchanged. Expected
calibration error is bin-dependent and noisy on small datasets, so it belongs beside
reliability plots and uncertainty rather than acting as a universal score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .losses import cross_entropy_from_logits
from .probabilities import stable_softmax


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TemperatureFit:
    """The chosen temperature, every candidate loss, and whether the grid bound it.

    ``on_grid_boundary`` is true when the winner is the smallest or largest declared
    candidate. A grid search cannot see past its own endpoints, so a boundary winner
    means the reported temperature may be an artifact of where the grid stopped
    rather than a minimum the calibration data actually chose.
    """

    temperature: float
    candidates: tuple[tuple[float, float], ...]
    on_grid_boundary: bool


def _labelled_logits(
    logits: ArrayLike, labels: ArrayLike
) -> tuple[FloatArray, NDArray[np.int64]]:
    matrix = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("calibration needs at least two rows and two classes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("calibration logits must be finite")
    if targets.ndim != 1 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("labels must provide one class index per logit row")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("calibration labels must be integer class indices")
    integer_targets = targets.astype(np.int64, copy=False)
    if np.any(integer_targets < 0) or np.any(integer_targets >= matrix.shape[1]):
        raise ValueError("calibration labels must fall inside the class range")
    return matrix, integer_targets


def expected_calibration_error(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    bins: int,
    temperature: float = 1.0,
) -> float:
    """Return equal-width ECE for a finite labelled classification set.

    Bins must fall between one and the row count. Confidence exactly on an interior
    boundary enters the higher bin; confidence one enters the final bin. This scalar
    can hide class and cohort errors, and passing it does not establish calibration
    outside the measured distribution.
    """

    matrix, targets = _labelled_logits(logits, labels)
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise TypeError("bins must be an integer")
    if bins < 1 or bins > matrix.shape[0]:
        raise ValueError("bins must be between one and the number of rows")
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError("temperature must be greater than zero")
    probabilities = stable_softmax(matrix / temperature_value, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == targets
    bin_ids = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        count = int(np.sum(selected))
        if count:
            weight = count / matrix.shape[0]
            error += weight * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return error


def fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    temperatures: tuple[float, ...],
) -> TemperatureFit:
    """Choose the declared temperature with the lowest mean cross-entropy.

    The candidate grid must contain at least two unique, strictly increasing positive
    values. Exact loss ties choose the smaller temperature. The caller owns the split
    boundary and must not evaluate calibration on these same fitted rows.

    The result flags a winner that sits on either grid endpoint. That happens whenever
    loss is still falling at the edge, which a calibration split the model classifies
    perfectly guarantees. Treat a flagged fit as unresolved and widen the grid, or say
    plainly that the data could not choose a temperature.
    """

    matrix, targets = _labelled_logits(logits, labels)
    if len(temperatures) < 2:
        raise ValueError("temperature grid must contain at least two values")
    normalized = tuple(float(value) for value in temperatures)
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("temperature candidates must be finite and positive")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("temperature candidates must be unique and increasing")
    candidates: list[tuple[float, float]] = []
    for temperature in normalized:
        losses = [
            cross_entropy_from_logits(row / temperature, int(target))
            for row, target in zip(matrix, targets, strict=True)
        ]
        candidates.append((temperature, float(np.mean(losses))))
    winner = min(candidates, key=lambda item: (item[1], item[0]))
    on_boundary = winner[0] in (normalized[0], normalized[-1])
    return TemperatureFit(winner[0], tuple(candidates), on_boundary)
