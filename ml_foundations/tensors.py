"""Make tensor shapes and axes explicit before model math begins.

Model code often fails because two arrays happen to broadcast instead of because a
formula is unfamiliar. This module validates rank, shape, finiteness, and zero norms
before it computes a similarity or affine projection. Successful arithmetic proves
that these numeric arrays fit the stated contract. It does not prove that their axes
carry the intended business meaning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_array(name: str, values: ArrayLike, *, ndim: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def cosine_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Return directional similarity for two finite, nonzero 1-D vectors.

    The function rejects shape mismatch before computing norms, then rejects either
    zero norm because the angle of a zero vector is undefined. A return between -1
    and 1 describes these two vectors only. It is not evidence that an embedding
    model's geometry matches a product notion of relevance.
    """

    left_array = _finite_array("left", left, ndim=1)
    right_array = _finite_array("right", right, ndim=1)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"vectors must share one shape, got {left_array.shape} and {right_array.shape}"
        )
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return float(np.clip(similarity, -1.0, 1.0))


def affine(inputs: ArrayLike, weights: ArrayLike, bias: ArrayLike) -> FloatArray:
    """Apply inputs @ weights + bias under a batch-by-feature shape contract.

    Inputs must have shape batch by input features. Weights must have shape input
    features by output features. Bias must contain one output-feature row. Bias
    broadcasting is deliberate and no other broadcasting is accepted.
    """

    input_array = _finite_array("inputs", inputs, ndim=2)
    weight_array = _finite_array("weights", weights, ndim=2)
    bias_array = _finite_array("bias", bias, ndim=1)
    if input_array.shape[1] != weight_array.shape[0]:
        raise ValueError("input feature width must match the first weight dimension")
    if weight_array.shape[1] != bias_array.shape[0]:
        raise ValueError("bias width must match the output feature width")
    return input_array @ weight_array + bias_array
