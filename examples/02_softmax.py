"""Turn logits into probabilities without numeric overflow.

Prediction: adding the same constant to every logit does not change softmax.
Run with `python examples/02_softmax.py`.
"""

import numpy as np

from ml_foundations import stable_softmax


def main() -> None:
    logits = np.array([1000.0, 1001.0, 1004.0])
    shifted = logits - 10_000.0
    probabilities = stable_softmax(logits)
    shifted_probabilities = stable_softmax(shifted)

    print("Stable softmax")
    print(f"  logits: {logits.tolist()}")
    print(f"  probabilities: {np.round(probabilities, 6).tolist()}")
    print(f"  sum: {probabilities.sum():.6f}")
    print(
        "  unchanged after shared shift: "
        f"{np.allclose(probabilities, shifted_probabilities)}"
    )
    print("\nTakeaway: logits are relative scores, not probabilities or confidence.")


if __name__ == "__main__":
    main()
