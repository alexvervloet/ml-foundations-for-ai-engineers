"""Tests for calibration split mechanics and quantization accounting."""

import unittest

import numpy as np

from ml_foundations import (
    expected_calibration_error,
    fit_temperature,
    mean_absolute_error,
    quantize_symmetric,
    stable_softmax,
)


class CalibrationTests(unittest.TestCase):
    def test_overconfident_logits_choose_cooling_and_lower_test_ece(self) -> None:
        calibration_logits = np.array(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]
        )
        calibration_labels = np.array([0, 0, 1, 1])
        fitted = fit_temperature(
            calibration_logits,
            calibration_labels,
            temperatures=(0.5, 1.0, 2.0, 4.0, 8.0),
        )
        self.assertGreater(fitted.temperature, 1.0)
        self.assertFalse(fitted.on_grid_boundary)
        test_logits = np.array([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
        test_labels = np.array([0, 1, 1, 1])
        before = expected_calibration_error(test_logits, test_labels, bins=2)
        after = expected_calibration_error(
            test_logits, test_labels, bins=2, temperature=fitted.temperature
        )
        self.assertLess(after, before)

    def test_a_separable_split_pins_the_fit_to_the_grid_floor(self) -> None:
        separable_logits = np.array([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0]])
        separable_labels = np.array([0, 1, 0])
        narrow = fit_temperature(
            separable_logits, separable_labels, temperatures=(0.5, 1.0, 2.0)
        )
        wider = fit_temperature(
            separable_logits, separable_labels, temperatures=(0.05, 0.5, 1.0, 2.0)
        )

        self.assertEqual(narrow.temperature, 0.5)
        self.assertTrue(narrow.on_grid_boundary)
        self.assertEqual(wider.temperature, 0.05)
        self.assertTrue(wider.on_grid_boundary)
        self.assertLess(
            dict(wider.candidates)[0.05], dict(narrow.candidates)[0.5]
        )

    def test_a_numpy_float_grid_is_still_accepted(self) -> None:
        fitted = fit_temperature(
            [[1.0, 0.0], [0.0, 1.0]],
            [0, 1],
            temperatures=tuple(np.linspace(0.5, 2.0, 4)),
        )

        self.assertAlmostEqual(fitted.temperature, 0.5)

    def test_temperature_scaling_preserves_argmax(self) -> None:
        logits = np.array([[3.0, 1.0, -2.0], [0.0, 2.0, 1.0]])
        cold = np.argmax(stable_softmax(logits / 0.5, axis=1), axis=1)
        hot = np.argmax(stable_softmax(logits / 4.0, axis=1), axis=1)
        np.testing.assert_array_equal(cold, hot)

    def test_temperature_tie_chooses_smallest_value(self) -> None:
        fitted = fit_temperature(
            np.zeros((2, 2)),
            np.array([0, 1]),
            temperatures=(0.5, 1.0, 2.0),
        )
        self.assertEqual(fitted.temperature, 0.5)

    def test_calibration_rejects_invalid_data_grid_and_bins(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            fit_temperature([[1.0, 0.0]], [0], temperatures=(1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            fit_temperature(
                [[1.0, 0.0], [0.0, 1.0]],
                [0, 1],
                temperatures=(1.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "between one"):
            expected_calibration_error(
                [[1.0, 0.0], [0.0, 1.0]], [0, 1], bins=3
            )
        for temperature in (True, np.True_):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                TypeError, "not a Boolean"
            ):
                expected_calibration_error(
                    [[1.0, 0.0], [0.0, 1.0]],
                    [0, 1],
                    bins=2,
                    temperature=temperature,
                )
        for grid in ((True, 2.0), (0.5, np.True_)):
            with self.subTest(grid=grid), self.assertRaisesRegex(
                TypeError, "not Booleans"
            ):
                fit_temperature([[1.0, 0.0], [0.0, 1.0]], [0, 1], temperatures=grid)
        self.assertGreaterEqual(
            expected_calibration_error(
                [[1.0, 0.0], [0.0, 1.0]], [0, 1], bins=2
            ),
            0.0,
        )


class QuantizationTests(unittest.TestCase):
    def test_quantization_respects_range_shape_and_byte_accounting(self) -> None:
        source = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
        quantized = quantize_symmetric(source, bits=4)
        self.assertGreaterEqual(int(np.min(quantized.values)), -7)
        self.assertLessEqual(int(np.max(quantized.values)), 7)
        self.assertEqual(quantized.dequantize().shape, source.shape)
        self.assertEqual(quantized.source_bytes, 128)
        self.assertEqual(quantized.packed_payload_bytes, 16)

    def test_source_bytes_follow_the_caller_dtype(self) -> None:
        values = np.linspace(-1.0, 1.0, 16)
        as_float64 = quantize_symmetric(values, bits=8)
        as_float32 = quantize_symmetric(values.astype(np.float32), bits=8)
        self.assertEqual(as_float64.source_bytes, 128)
        self.assertEqual(as_float32.source_bytes, 64)
        self.assertEqual(
            as_float32.packed_payload_bytes, as_float64.packed_payload_bytes
        )

    def test_zero_tensor_has_exact_reconstruction(self) -> None:
        source = np.zeros((2, 3))
        quantized = quantize_symmetric(source, bits=8)
        self.assertEqual(quantized.scale, 1.0)
        np.testing.assert_array_equal(quantized.dequantize(), source)

    def test_more_levels_reduce_error_on_the_same_values(self) -> None:
        source = np.array([-1.0, -0.3, 0.2, 0.9])
        int4 = quantize_symmetric(source, bits=4)
        int8 = quantize_symmetric(source, bits=8)
        self.assertLess(
            mean_absolute_error(source, int8.dequantize()),
            mean_absolute_error(source, int4.dequantize()),
        )

    def test_quantization_rejects_invalid_values_and_bits(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            quantize_symmetric([], bits=8)
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_symmetric([np.nan], bits=8)
        for bits in (1, 9):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                quantize_symmetric([1.0], bits=bits)


if __name__ == "__main__":
    unittest.main()
