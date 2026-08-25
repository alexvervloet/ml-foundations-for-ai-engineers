"""Measure how strongly logits support the required class.

Cross-entropy consumes logits and a trusted integer target. It combines log-softmax
and negative log likelihood without first rounding or sampling a prediction. The
target belongs to supervised training or an evaluator, never to inference input.
This implementation handles one row at a time so the arithmetic stays visible.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def cross_entropy_from_logits(logits: ArrayLike, target: int) -> float:
    """Return single-example cross-entropy after validating logits and target.

    The function subtracts the maximum before log-sum-exp for numerical stability.
    The target must be an integer token id between zero and the class count minus
    one. A lower loss means the target received more relative support. It does not
    say whether the model is calibrated or useful on representative data.
    """

    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1 or row.size == 0:
        raise ValueError("cross-entropy requires one nonempty 1-D logit row")
    if not np.all(np.isfinite(row)):
        raise ValueError("cross-entropy logits must be finite")
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)):
        raise TypeError("target must be an integer class index")
    target_index = int(target)
    if target_index < 0 or target_index >= row.size:
        raise ValueError("target must be between zero and the class count minus one")
    maximum = float(np.max(row))
    log_partition = maximum + math.log(float(np.sum(np.exp(row - maximum))))
    return log_partition - float(row[target_index])
