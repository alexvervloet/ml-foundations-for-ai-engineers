"""Connect target probability, cross-entropy, and a PyTorch training loss.

Prediction: raising only the target logit lowers cross-entropy. Run with
`python examples/03_cross_entropy.py`.
"""

import numpy as np
import torch
from torch.nn import functional as F

from ml_foundations import cross_entropy_from_logits, stable_softmax


def main() -> None:
    logits = np.array([0.5, 1.0, -0.5])
    improved = logits.copy()
    improved[2] += 3.0
    target = 2

    before = cross_entropy_from_logits(logits, target)
    after = cross_entropy_from_logits(improved, target)
    torch_loss = float(
        F.cross_entropy(
            torch.from_numpy(improved[None, :]),
            torch.tensor([target]),
        )
    )

    print("Cross-entropy from logits")
    print(f"  target probability before: {stable_softmax(logits)[target]:.4f}")
    print(f"  target probability after:  {stable_softmax(improved)[target]:.4f}")
    print(f"  loss before: {before:.4f}")
    print(f"  loss after:  {after:.4f}")
    print(f"  NumPy result matches PyTorch: {np.isclose(after, torch_loss)}")
    print("\nTakeaway: cross-entropy rewards relative target score, not raw score.")


if __name__ == "__main__":
    main()
