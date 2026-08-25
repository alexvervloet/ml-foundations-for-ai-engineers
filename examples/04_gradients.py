"""Check one gradient three ways before trusting an optimizer.

Prediction: analytic, finite-difference, and autograd derivatives agree for this
smooth scalar loss. Run with `python examples/04_gradients.py`.
"""

import torch

from ml_foundations import finite_difference


def loss(value: float) -> float:
    return (value - 3.0) ** 2 + 1.0


def main() -> None:
    parameter = 1.25
    analytic = 2.0 * (parameter - 3.0)
    numeric = finite_difference(loss, parameter)
    tensor = torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
    torch_loss = (tensor - 3.0) ** 2 + 1.0
    torch_loss.backward()
    autograd = float(tensor.grad)

    print("Gradient checks")
    print(f"  analytic:          {analytic:.8f}")
    print(f"  finite difference: {numeric:.8f}")
    print(f"  autograd:          {autograd:.8f}")
    print(f"  largest gap:       {max(abs(analytic - numeric), abs(analytic - autograd)):.2e}")
    print("\nTakeaway: agreement checks the local derivative, not the whole training run.")


if __name__ == "__main__":
    main()
