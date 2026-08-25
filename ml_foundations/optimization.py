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

import numpy as np


LossAndGradient = Callable[[float], tuple[float, float]]


def _real(name: str, value: object) -> float:
    """Return one finite control value, rejecting Booleans before any conversion.

    ``float(True)`` is 1.0, so an unguarded control reads a Boolean as a plausible
    number rather than refusing it. ``max_steps`` has always rejected Booleans and the
    floating controls now agree with it. ``numpy.bool_`` is not a Python ``bool``, so
    it is named explicitly. Anything else that converts cleanly to a finite float is
    still accepted, so NumPy and other real scalar types keep working.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a Boolean")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


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

    epsilon_value = _real("epsilon", epsilon)
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

    When the update itself overflows, the final trace point carries the nonfinite
    parameter with a loss and gradient of nan, because the objective was never called
    there. Filling those fields with infinity would put a number in the trace that
    nothing measured.
    """

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    rate = _real("learning_rate", learning_rate)
    if rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")
    stop_below = _real("tolerance", tolerance)
    if stop_below < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    parameter = _real("initial", initial)
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
        if abs(point.gradient) <= stop_below:
            return OptimizationResult(OptimizationStatus.CONVERGED, tuple(trace))
        if step == max_steps:
            return OptimizationResult(OptimizationStatus.MAX_STEPS, tuple(trace))
        parameter -= rate * point.gradient
        if not math.isfinite(parameter):
            trace.append(
                OptimizationPoint(step + 1, parameter, math.nan, math.nan)
            )
            return OptimizationResult(OptimizationStatus.DIVERGED, tuple(trace))
    raise RuntimeError("gradient descent reached an impossible state")
