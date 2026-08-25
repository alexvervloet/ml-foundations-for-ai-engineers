"""Inspect scaled dot-product attention and its causal mask.

Prediction: changing the final value cannot change earlier causal outputs. Run with
`python examples/06_attention.py`.
"""

import numpy as np

from ml_foundations import causal_mask, scaled_dot_product_attention


def main() -> None:
    query = np.eye(3)
    key = np.eye(3)
    value = np.array([[1.0, 0.0], [2.0, 1.0], [8.0, 4.0]])
    changed = value.copy()
    changed[-1] = [800.0, 400.0]
    mask = causal_mask(3)

    baseline, weights = scaled_dot_product_attention(query, key, value, mask=mask)
    counterfactual, _changed_weights = scaled_dot_product_attention(
        query, key, changed, mask=mask
    )
    earlier_unchanged = np.allclose(baseline[:-1], counterfactual[:-1])

    print("Causal scaled dot-product attention")
    print(f"  mask rows: {mask.astype(int).tolist()}")
    print(f"  first-row weights: {np.round(weights[0], 4).tolist()}")
    print(f"  last-row weights:  {np.round(weights[-1], 4).tolist()}")
    print(f"  earlier outputs unchanged: {earlier_unchanged}")
    print("\nTakeaway: the mask controls information flow before softmax.")


if __name__ == "__main__":
    main()
