"""Reduce logical weight precision and measure the reconstruction error.

Symmetric quantization maps a floating-point range onto signed integer levels with
one scale. Fewer bits reduce the packed payload estimate and increase rounding noise
in a workload-dependent way. This module does not pack sub-byte values or benchmark a
kernel. A smaller byte count is not a latency or throughput result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class QuantizedTensor:
    """Logical signed integers, scale, shape, and packed-payload accounting.

    ``source_bytes`` measures the array the caller passed in, at the caller's own
    dtype. It is not measured after any internal conversion, because a float32
    tensor stored as float32 must not be reported as if it occupied float64 bytes.
    """

    values: NDArray[np.int8]
    scale: float
    bits: int
    original_shape: tuple[int, ...]
    source_bytes: int
    packed_payload_bytes: int

    def dequantize(self) -> NDArray[np.float64]:
        """Reconstruct float64 values; this does not restore discarded precision."""

        return self.values.astype(np.float64).reshape(self.original_shape) * self.scale


def quantize_symmetric(values: ArrayLike, *, bits: int) -> QuantizedTensor:
    """Quantize finite values into a signed two-to-eight-bit logical range.

    Empty and nonfinite arrays fail before bit validation. Bit widths below two have
    no positive signed level, while widths above eight do not fit the teaching int8
    container. An all-zero tensor uses scale one and reconstructs exactly. Packed
    bytes include only the logical payload and one float64 scale, not a real file or
    runtime layout.

    Byte accounting reads the caller's array before the float64 working copy is made.
    Reporting the converted copy would inflate every float32 or float16 source by two
    or four times and turn an honest payload comparison into a flattering one.
    """

    original = np.asarray(values)
    source = original.astype(np.float64, copy=False)
    if source.size == 0:
        raise ValueError("quantization requires at least one value")
    if not np.all(np.isfinite(source)):
        raise ValueError("quantization values must be finite")
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise TypeError("bits must be an integer")
    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")
    maximum_level = 2 ** (bits - 1) - 1
    absolute_maximum = float(np.max(np.abs(source)))
    scale = 1.0 if absolute_maximum == 0.0 else absolute_maximum / maximum_level
    logical = np.rint(source / scale)
    logical = np.clip(logical, -maximum_level, maximum_level).astype(np.int8)
    payload_bytes = math.ceil(source.size * bits / 8) + 8
    return QuantizedTensor(
        logical.reshape(-1),
        scale,
        bits,
        source.shape,
        original.nbytes,
        payload_bytes,
    )


def mean_absolute_error(
    source: ArrayLike, reconstructed: ArrayLike
) -> float:
    """Return mean absolute error for finite arrays with one exact shape."""

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(reconstructed, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("error inputs must share one nonempty shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("error inputs must be finite")
    return float(np.mean(np.abs(left - right)))
