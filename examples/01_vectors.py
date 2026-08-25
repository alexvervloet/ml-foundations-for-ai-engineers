"""See vectors as model data with explicit shape contracts.

Prediction: two vectors that point in the same direction have cosine similarity one,
even when their magnitudes differ. Run with `python examples/01_vectors.py`.
"""

import numpy as np

from ml_foundations import affine, cosine_similarity


def main() -> None:
    query = np.array([1.0, 2.0, 3.0])
    scaled = query * 4.0
    unrelated = np.array([3.0, -2.0, 1.0])
    batch = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    weights = np.array([[2.0, -1.0], [0.5, 1.0], [1.0, 0.0]])
    bias = np.array([0.25, -0.25])

    same_direction = cosine_similarity(query, scaled)
    different_direction = cosine_similarity(query, unrelated)
    projected = affine(batch, weights, bias)

    print("Vector geometry and shape contracts")
    print(f"  same direction cosine: {same_direction:.3f}")
    print(f"  other direction cosine: {different_direction:.3f}")
    print(f"  affine shapes: {batch.shape} @ {weights.shape} -> {projected.shape}")
    print(f"  first projected row: {projected[0].tolist()}")
    print("\nTakeaway: shape compatibility and direction are separate facts.")


if __name__ == "__main__":
    main()
