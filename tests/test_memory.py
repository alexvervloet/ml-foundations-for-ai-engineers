"""Tests for explicit training and inference byte accounting."""

import unittest

from ml_foundations import estimate_inference_memory, estimate_training_memory


class MemoryTests(unittest.TestCase):
    def test_training_names_every_default_component(self) -> None:
        estimate = estimate_training_memory(100, 50)
        components = {item.name: item.bytes for item in estimate.components}
        self.assertEqual(
            components,
            {
                "weights": 200,
                "gradients": 200,
                "master_weights": 400,
                "optimizer_states": 800,
                "saved_activations": 100,
            },
        )
        self.assertEqual(estimate.total_bytes, 1_700)

    def test_kv_memory_doubles_with_tokens(self) -> None:
        first = estimate_inference_memory(100, 40, 10)
        second = estimate_inference_memory(100, 80, 10)
        first_components = {item.name: item.bytes for item in first.components}
        second_components = {item.name: item.bytes for item in second.components}
        self.assertEqual(first_components["kv_cache"], 80)
        self.assertEqual(second_components["kv_cache"], 160)
        self.assertEqual(
            second_components["kv_cache"], 2 * first_components["kv_cache"]
        )
        self.assertEqual(second_components["weights"], first_components["weights"])

    def test_zero_counts_are_valid_but_negative_counts_fail(self) -> None:
        self.assertEqual(
            estimate_inference_memory(0, 0, 0).total_bytes,
            0,
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            estimate_training_memory(-1, 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            estimate_inference_memory(1, 1, 1, kv_bytes=0)


if __name__ == "__main__":
    unittest.main()
