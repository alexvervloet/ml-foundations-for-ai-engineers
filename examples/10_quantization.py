"""Measure the payload and reconstruction tradeoff at two logical bit widths.

Prediction: int4 has a smaller payload than int8 but more error on these weights.
Run with `python examples/10_quantization.py`.
"""

import numpy as np

from ml_foundations import mean_absolute_error, quantize_symmetric


def main() -> None:
    weights = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
    int8 = quantize_symmetric(weights, bits=8)
    int4 = quantize_symmetric(weights, bits=4)
    int8_error = mean_absolute_error(weights, int8.dequantize())
    int4_error = mean_absolute_error(weights, int4.dequantize())

    print("Logical symmetric quantization")
    print(f"  float64 source: {weights.nbytes} bytes")
    print(
        f"  int8 payload:   {int8.packed_payload_bytes} bytes, "
        f"mean error={int8_error:.6f}"
    )
    print(
        f"  int4 payload:   {int4.packed_payload_bytes} bytes, "
        f"mean error={int4_error:.6f}"
    )
    print(f"  smaller payload has more error: {int4_error > int8_error}")
    print("\nTakeaway: byte estimates are not kernel speed measurements.")


if __name__ == "__main__":
    main()
