"""Fit temperature on calibration rows and measure it on separate test rows.

Prediction: this overconfident calibration set chooses cooling. Whether cooling helps
is decided only on the held-out rows. Run with `python examples/09_calibration.py`.
"""

import numpy as np

from ml_foundations import expected_calibration_error, fit_temperature


def main() -> None:
    calibration_logits = np.array(
        [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]
    )
    calibration_labels = np.array([0, 0, 1, 1])
    test_logits = np.array(
        [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]
    )
    test_labels = np.array([0, 1, 1, 1])

    fitted = fit_temperature(
        calibration_logits,
        calibration_labels,
        temperatures=(0.5, 1.0, 2.0, 4.0),
    )
    before = expected_calibration_error(test_logits, test_labels, bins=2)
    after = expected_calibration_error(
        test_logits, test_labels, bins=2, temperature=fitted.temperature
    )

    print("Held-out temperature scaling")
    print(f"  fitted temperature: {fitted.temperature:.1f}")
    print(f"  test ECE before:    {before:.4f}")
    print(f"  test ECE after:     {after:.4f}")
    print(f"  held-out ECE improved: {after < before}")
    print("\nTakeaway: fitting and judging calibration require different rows.")


if __name__ == "__main__":
    main()
