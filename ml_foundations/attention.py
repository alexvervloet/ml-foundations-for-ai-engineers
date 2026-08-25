"""Expose scaled dot-product attention as shape checks plus weighted retrieval.

Queries score keys, a mask removes forbidden relationships, softmax normalizes the
allowed scores, and the weights mix values. Scaling by the square root of key width
keeps dot products from growing solely because vectors are wider. This NumPy version
is readable and slow. Production code should use a fused framework kernel.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def causal_mask(length: int) -> NDArray[np.bool_]:
    """Return a square mask where token i can use tokens zero through i."""

    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    return np.tril(np.ones((length, length), dtype=np.bool_))


def _matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def scaled_dot_product_attention(
    queries: ArrayLike,
    keys: ArrayLike,
    values: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return attention output and weights after validating every relationship.

    Validation order is query, key, and value shape, then mask shape and dtype, then
    permission coverage. Every query must retain at least one key. A successful result
    proves the stated numeric operation and mask. It does not show that learned
    projections put useful information in these vectors.
    """

    query_matrix = _matrix("queries", queries)
    key_matrix = _matrix("keys", keys)
    value_matrix = _matrix("values", values)
    if query_matrix.shape[1] != key_matrix.shape[1]:
        raise ValueError("query and key widths must match")
    if key_matrix.shape[0] != value_matrix.shape[0]:
        raise ValueError("keys and values must contain the same number of positions")
    allowed = np.ones(
        (query_matrix.shape[0], key_matrix.shape[0]), dtype=np.bool_
    )
    if mask is not None:
        raw_mask = np.asarray(mask)
        if raw_mask.dtype != np.bool_:
            raise TypeError("attention mask must contain booleans")
        if raw_mask.shape != allowed.shape:
            raise ValueError(
                f"mask shape must be {allowed.shape}, got {raw_mask.shape}"
            )
        allowed = raw_mask
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("every query must be allowed to attend to at least one key")
    scores = query_matrix @ key_matrix.T / math.sqrt(query_matrix.shape[1])
    masked_scores = np.where(allowed, scores, -np.inf)
    row_maximum = np.max(masked_scores, axis=1, keepdims=True)
    exponentials = np.where(allowed, np.exp(masked_scores - row_maximum), 0.0)
    weights = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return weights @ value_matrix, weights
