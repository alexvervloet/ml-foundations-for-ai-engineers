"""Separate greedy selection, temperature, top-k, and random state.

Prediction: top-k sampling never returns an excluded token, and the same seeded
generator reproduces the sequence. Run with `python examples/08_sampling.py`.
"""

import numpy as np

from ml_foundations import greedy_token, sample_from_logits


def draw(seed: int) -> list[int]:
    logits = np.array([3.0, 2.5, 1.0, -4.0])
    rng = np.random.default_rng(seed)
    return [
        sample_from_logits(logits, temperature=0.8, top_k=2, rng=rng)
        for _ in range(12)
    ]


def main() -> None:
    logits = np.array([3.0, 2.5, 1.0, -4.0])
    first = draw(41)
    replay = draw(41)

    print("Sampling policy")
    print(f"  greedy token: {greedy_token(logits)}")
    print(f"  top-k draws: {first}")
    print(f"  excluded ids absent: {set(first).isdisjoint({2, 3})}")
    print(f"  seeded replay matches: {first == replay}")
    print("\nTakeaway: randomness is reproducible only when state is part of the input.")


if __name__ == "__main__":
    main()
