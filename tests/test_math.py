"""Tests for tensor, probability, loss, and optimization contracts."""

import math
import unittest

import numpy as np
import torch

from ml_foundations import (
    OptimizationStatus,
    affine,
    cosine_similarity,
    cross_entropy_from_logits,
    finite_difference,
    gradient_descent,
    greedy_token,
    sample_from_logits,
    stable_softmax,
)


class TensorTests(unittest.TestCase):
    def test_cosine_similarity_has_geometric_boundaries(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [2, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 3]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-2, 0]), -1.0)

    def test_cosine_rejects_shape_and_zero_norm(self) -> None:
        with self.assertRaisesRegex(ValueError, "share one shape"):
            cosine_similarity([1, 2], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "zero vector"):
            cosine_similarity([0, 0], [1, 0])

    def test_affine_allows_only_bias_broadcasting(self) -> None:
        result = affine([[1, 2], [3, 4]], [[1, 0], [0, 2]], [10, -1])
        np.testing.assert_array_equal(result, [[11, 3], [13, 7]])
        with self.assertRaisesRegex(ValueError, "feature width"):
            affine([[1, 2]], [[1, 2]], [0, 0])


class ProbabilityAndLossTests(unittest.TestCase):
    def test_softmax_shift_invariance_and_large_logits(self) -> None:
        ordinary = stable_softmax([1.0, 2.0, 3.0])
        shifted = stable_softmax([10_001.0, 10_002.0, 10_003.0])
        np.testing.assert_allclose(ordinary, shifted, rtol=0.0, atol=1e-15)
        self.assertTrue(np.all(np.isfinite(shifted)))
        self.assertAlmostEqual(float(np.sum(shifted)), 1.0)

    def test_softmax_rejects_invalid_inputs(self) -> None:
        for values in ([], [0.0, math.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stable_softmax(values)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            stable_softmax([[1.0, 2.0]], axis=2)

    def test_target_logit_increase_lowers_loss_and_matches_torch(self) -> None:
        logits = np.array([0.2, -0.1, 0.4])
        original = cross_entropy_from_logits(logits, 1)
        improved = logits.copy()
        improved[1] += 1.0
        self.assertLess(cross_entropy_from_logits(improved, 1), original)
        expected = torch.nn.functional.cross_entropy(
            torch.tensor(logits, dtype=torch.float64).unsqueeze(0),
            torch.tensor([1]),
        )
        self.assertAlmostEqual(original, expected.item(), places=12)

    def test_cross_entropy_rejects_target_boundaries(self) -> None:
        for target in (-1, 3):
            with self.subTest(target=target), self.assertRaises(ValueError):
                cross_entropy_from_logits([1, 2, 3], target)

    def test_sampling_is_reproducible_with_injected_rng(self) -> None:
        first = np.random.default_rng(7)
        second = np.random.default_rng(7)
        draws_a = [
            sample_from_logits([0.0, 1.0, 2.0], temperature=0.8, rng=first)
            for _ in range(20)
        ]
        draws_b = [
            sample_from_logits([0.0, 1.0, 2.0], temperature=0.8, rng=second)
            for _ in range(20)
        ]
        self.assertEqual(draws_a, draws_b)

    def test_top_k_never_emits_an_excluded_id(self) -> None:
        rng = np.random.default_rng(11)
        draws = {
            sample_from_logits(
                [8.0, 7.0, 6.0, 5.0], temperature=1.0, top_k=2, rng=rng
            )
            for _ in range(200)
        }
        self.assertTrue(draws <= {0, 1})
        self.assertEqual(greedy_token([3.0, 3.0, 1.0]), 0)

    def test_sampling_rejects_invalid_policy(self) -> None:
        rng = np.random.default_rng(1)
        with self.assertRaisesRegex(ValueError, "1-D"):
            sample_from_logits([[1.0]], temperature=1.0, rng=rng)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            sample_from_logits([1.0], temperature=0.0, rng=rng)
        for top_k in (0, 3):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                sample_from_logits([1.0, 0.0], temperature=1.0, top_k=top_k, rng=rng)
        with self.assertRaisesRegex(TypeError, "not a Boolean"):
            sample_from_logits([1.0, 0.0], temperature=True, rng=rng)


class OptimizationTests(unittest.TestCase):
    @staticmethod
    def objective(parameter: float) -> tuple[float, float]:
        return (parameter - 3.0) ** 2, 2.0 * (parameter - 3.0)

    def test_three_gradient_methods_agree(self) -> None:
        parameter = 1.5
        analytic = self.objective(parameter)[1]
        numeric = finite_difference(lambda value: (value - 3.0) ** 2, parameter)
        torch_parameter = torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
        torch_loss = (torch_parameter - 3.0) ** 2
        torch_loss.backward()
        self.assertAlmostEqual(analytic, numeric, places=7)
        self.assertAlmostEqual(analytic, torch_parameter.grad.item(), places=12)

    def test_descent_converges_and_tolerance_equality_counts(self) -> None:
        converged = gradient_descent(
            0.0,
            self.objective,
            learning_rate=0.2,
            max_steps=100,
            tolerance=1e-6,
        )
        self.assertIs(converged.status, OptimizationStatus.CONVERGED)
        boundary = gradient_descent(
            0.0,
            lambda value: (value, 0.25),
            learning_rate=0.1,
            max_steps=2,
            tolerance=0.25,
        )
        self.assertIs(boundary.status, OptimizationStatus.CONVERGED)
        self.assertEqual(len(boundary.trace), 1)

    def test_step_budget_and_nonfinite_failure_are_distinct(self) -> None:
        budget = gradient_descent(
            0.0,
            self.objective,
            learning_rate=0.01,
            max_steps=1,
            tolerance=0.0,
        )
        self.assertIs(budget.status, OptimizationStatus.MAX_STEPS)
        divergence = gradient_descent(
            0.0,
            lambda _value: (math.inf, math.inf),
            learning_rate=0.1,
            max_steps=2,
            tolerance=0.0,
        )
        self.assertIs(divergence.status, OptimizationStatus.DIVERGED)

    def test_optimization_controls_and_epsilon_boundary(self) -> None:
        for learning_rate, steps, tolerance in ((0.0, 1, 0.0), (0.1, 0, 0.0), (0.1, 1, -1.0)):
            with self.subTest(
                learning_rate=learning_rate, steps=steps, tolerance=tolerance
            ), self.assertRaises(ValueError):
                gradient_descent(
                    0.0,
                    self.objective,
                    learning_rate=learning_rate,
                    max_steps=steps,
                    tolerance=tolerance,
                )
        with self.assertRaisesRegex(ValueError, "at least"):
            finite_difference(lambda value: value**2, 1.0, epsilon=1e-9)
        self.assertAlmostEqual(
            finite_difference(lambda value: value**2, 1.0, epsilon=1e-8),
            2.0,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
