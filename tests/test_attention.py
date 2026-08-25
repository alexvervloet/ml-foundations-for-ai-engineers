"""Tests for attention shape, masking, and scaling contracts."""

import unittest

import numpy as np

from ml_foundations import causal_mask, scaled_dot_product_attention


class AttentionTests(unittest.TestCase):
    def test_causal_attention_cannot_see_future_values(self) -> None:
        queries = np.eye(3)
        keys = np.eye(3)
        original_values = np.array([[1.0], [2.0], [3.0]])
        changed_future = np.array([[1.0], [2.0], [999.0]])
        original, weights = scaled_dot_product_attention(
            queries, keys, original_values, mask=causal_mask(3)
        )
        changed, _ = scaled_dot_product_attention(
            queries, keys, changed_future, mask=causal_mask(3)
        )
        np.testing.assert_allclose(original[:2], changed[:2])
        self.assertEqual(weights[0, 1], 0.0)
        self.assertEqual(weights[0, 2], 0.0)
        np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(3))

    def test_weights_use_the_scaled_score_not_the_raw_dot_product(self) -> None:
        query = np.ones((1, 4))
        keys = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        values = np.eye(2)
        _, weights = scaled_dot_product_attention(query, keys, values)
        scaled_expected = np.exp(2.0) / (np.exp(2.0) + 1.0)
        unscaled_wrong = np.exp(4.0) / (np.exp(4.0) + 1.0)
        self.assertAlmostEqual(weights[0, 0], scaled_expected)
        self.assertNotAlmostEqual(weights[0, 0], unscaled_wrong)

    def test_attention_rejects_incompatible_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "widths"):
            scaled_dot_product_attention([[1, 2]], [[1]], [[1]])
        with self.assertRaisesRegex(ValueError, "same number"):
            scaled_dot_product_attention([[1]], [[1], [2]], [[1]])

    def test_attention_rejects_invalid_and_fully_masked_rows(self) -> None:
        with self.assertRaisesRegex(TypeError, "booleans"):
            scaled_dot_product_attention([[1]], [[1]], [[1]], mask=[[1]])
        with self.assertRaisesRegex(ValueError, "mask shape"):
            scaled_dot_product_attention([[1]], [[1]], [[1]], mask=[[True, False]])
        with self.assertRaisesRegex(ValueError, "at least one key"):
            scaled_dot_product_attention([[1]], [[1]], [[1]], mask=[[False]])
        output, weights = scaled_dot_product_attention(
            [[1]], [[1]], [[4]], mask=[[True]]
        )
        np.testing.assert_array_equal(output, [[4]])
        np.testing.assert_array_equal(weights, [[1]])


if __name__ == "__main__":
    unittest.main()
