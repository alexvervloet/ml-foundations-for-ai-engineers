"""See learning rate and step budget as parts of an optimizer contract.

Prediction: the same quadratic converges with a moderate learning rate, while a large
one moves away from the minimum until its step budget ends. Run with
`python examples/05_gradient_descent.py`.
"""

from ml_foundations import gradient_descent


def quadratic(parameter: float) -> tuple[float, float]:
    difference = parameter - 2.0
    return difference * difference, 2.0 * difference


def main() -> None:
    steady = gradient_descent(
        8.0,
        quadratic,
        learning_rate=0.25,
        max_steps=30,
        tolerance=1e-5,
    )
    unstable = gradient_descent(
        8.0,
        quadratic,
        learning_rate=1.1,
        max_steps=12,
        tolerance=1e-5,
    )

    print("Gradient descent states")
    for name, result in (("moderate rate", steady), ("large rate", unstable)):
        final = result.trace[-1]
        print(
            f"  {name}: status={result.status.value}, steps={final.step}, "
            f"parameter={final.parameter:.4f}, loss={final.loss:.4f}"
        )
    print("\nTakeaway: budget exhaustion and numeric divergence are not convergence.")


if __name__ == "__main__":
    main()
