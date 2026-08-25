"""Turn logits into probabilities or token choices under an explicit policy.

Logits are unnormalized scores. Softmax converts one chosen axis into probabilities,
while temperature and top-k decide how generation samples those probabilities. The
implementation subtracts the row maximum before exponentiation and validates every
policy control before drawing. It teaches distribution mechanics, not the quality or
safety of any token choice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def stable_softmax(values: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Normalize finite logits along one existing axis without overflow.

    Subtracting the maximum preserves softmax in real arithmetic because softmax is
    shift invariant. Floating-point results may differ by roundoff. Empty, nonfinite,
    scalar, or invalid-axis inputs fail instead of producing a misleading
    distribution.
    """

    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim == 0 or logits.size == 0:
        raise ValueError("softmax requires a nonempty array with at least one axis")
    if not np.all(np.isfinite(logits)):
        raise ValueError("softmax logits must be finite")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError("axis must be an integer")
    if axis < -logits.ndim or axis >= logits.ndim:
        raise ValueError(f"axis {axis} does not exist for shape {logits.shape}")
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def _sampling_row(logits: ArrayLike) -> FloatArray:
    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1 or row.size == 0:
        raise ValueError("sampling requires one nonempty 1-D logit row")
    if not np.all(np.isfinite(row)):
        raise ValueError("sampling logits must be finite")
    return row


def greedy_token(logits: ArrayLike) -> int:
    """Return the lowest token id among logits tied for the maximum."""

    row = _sampling_row(logits)
    return int(np.flatnonzero(row == np.max(row))[0])


def sample_from_logits(
    logits: ArrayLike,
    *,
    temperature: float,
    rng: np.random.Generator,
    top_k: int | None = None,
) -> int:
    """Draw one token from finite 1-D logits under temperature and top-k.

    Validation order is logits, temperature, then top-k. Temperature must be
    positive. Greedy decoding has its own function so zero cannot quietly change
    policy. Top-k ties use lower token ids first. The injected generator makes a run
    reproducible, but a single seeded draw does not characterize the distribution.
    """

    row = _sampling_row(logits)
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature):
        raise ValueError("temperature must be a finite positive number")
    temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    allowed = np.arange(row.size)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None")
        if top_k < 1 or top_k > row.size:
            raise ValueError("top_k must be between 1 and the vocabulary size")
        order = np.lexsort((allowed, -row))
        allowed = order[:top_k]
    probabilities = stable_softmax(row[allowed] / temperature_value)
    return int(rng.choice(allowed, p=probabilities))
