"""Show how gradients, learning rate, and a step budget determine training.

Gradient descent subtracts a local slope. It does not know whether the objective is
globally well behaved. This module keeps convergence, budget exhaustion, and numeric
divergence as separate terminal states. The scalar implementation exposes the control
flow. Real training uses vector gradients, adaptive optimizers, schedules, and more
careful diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math


LossAndGradient = Callable[[float], tuple[float, float]]


class OptimizationStatus(str, Enum):
    """Name why the optimizer stopped."""

    CONVERGED = "converged"
    MAX_STEPS = "max_steps"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class OptimizationPoint:
    """One observed parameter, loss, and gradient before an update."""

    step: int
    parameter: float
    loss: float
    gradient: float


@dataclass(frozen=True)
class OptimizationResult:
    """The complete scalar trace and its explicit terminal status."""

    status: OptimizationStatus
    trace: tuple[OptimizationPoint, ...]


def finite_difference(
    loss: Callable[[float], float], parameter: float, *, epsilon: float = 1e-6
) -> float:
    """Estimate a scalar derivative with a centered float64 difference.

    Epsilon below 1e-8 is rejected because cancellation can dominate this teaching
    calculation. The bound is a lesson-specific float64 policy, not a universal
    optimum. PyTorch gradcheck or analytic error analysis is the better tool for
    production tensor code.
    """

    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    epsilon_value = float(epsilon)
    if epsilon_value < 1e-8:
        raise ValueError("epsilon must be at least 1e-8 for this float64 check")
    plus = float(loss(parameter + epsilon_value))
    minus = float(loss(parameter - epsilon_value))
    if not math.isfinite(plus) or not math.isfinite(minus):
        raise ValueError("loss must remain finite around the checked parameter")
    return (plus - minus) / (2.0 * epsilon_value)


def gradient_descent(
    initial: float,
    loss_and_gradient: LossAndGradient,
    *,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
) -> OptimizationResult:
    """Run scalar gradient descent with explicit stop and failure semantics.

    Controls are validated before the objective runs. Each trace point records the
    state before an update. A gradient whose magnitude equals tolerance counts as
    converged. Nonfinite loss, gradient, or updated parameter reports divergence.
    Exhausting the budget reports max_steps rather than pretending convergence.
    """

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    parameter = float(initial)
    trace: list[OptimizationPoint] = []
    for step in range(max_steps + 1):
        loss, gradient = loss_and_gradient(parameter)
        point = OptimizationPoint(step, parameter, float(loss), float(gradient))
        trace.append(point)
        if not all(
            math.isfinite(value)
            for value in (point.parameter, point.loss, point.gradient)
        ):
            return OptimizationResult(OptimizationStatus.DIVERGED, tuple(trace))
        if abs(point.gradient) <= tolerance:
            return OptimizationResult(OptimizationStatus.CONVERGED, tuple(trace))
        if step == max_steps:
            return OptimizationResult(OptimizationStatus.MAX_STEPS, tuple(trace))
        parameter -= learning_rate * point.gradient
        if not math.isfinite(parameter):
            trace.append(OptimizationPoint(step + 1, parameter, math.inf, math.inf))
            return OptimizationResult(OptimizationStatus.DIVERGED, tuple(trace))
    raise RuntimeError("gradient descent reached an impossible state")
