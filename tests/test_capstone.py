"""Exercise capstone evidence, boundaries, lineage, and reproducibility."""

from dataclasses import replace
import math
import unittest

from hands_on.train_tiny_transformer import (
    DEFAULT_REQUIREMENTS,
    ExperimentVerdict,
    run_experiment,
    verdict_for,
)


class CapstoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_experiment()

    def test_default_experiment_meets_independent_requirements(self) -> None:
        report = self.report
        observed = report.observations

        self.assertEqual(report.verdict, ExperimentVerdict.READY_FOR_LAB_USE)
        self.assertLess(observed.final_loss, observed.initial_loss)
        self.assertGreaterEqual(
            observed.test_accuracy, DEFAULT_REQUIREMENTS.minimum_test_accuracy
        )
        self.assertLessEqual(
            observed.test_ece_after, DEFAULT_REQUIREMENTS.maximum_test_ece
        )
        self.assertLessEqual(
            observed.mean_logit_drift,
            DEFAULT_REQUIREMENTS.maximum_mean_logit_drift,
        )
        self.assertLess(
            observed.quantized_payload_bytes, observed.source_weight_bytes
        )

    def test_experiment_is_reproducible(self) -> None:
        self.assertEqual(run_experiment(), self.report)

    def test_equal_boundaries_pass(self) -> None:
        observed = replace(
            self.report.observations,
            final_loss=DEFAULT_REQUIREMENTS.maximum_final_loss,
            test_accuracy=DEFAULT_REQUIREMENTS.minimum_test_accuracy,
            test_ece_after=DEFAULT_REQUIREMENTS.maximum_test_ece,
            mean_logit_drift=DEFAULT_REQUIREMENTS.maximum_mean_logit_drift,
        )

        verdict, _reason = verdict_for(DEFAULT_REQUIREMENTS, observed)

        self.assertEqual(verdict, ExperimentVerdict.READY_FOR_LAB_USE)

    def test_each_requirement_can_change_the_verdict(self) -> None:
        observed = self.report.observations
        cases = (
            (
                replace(
                    observed,
                    final_loss=DEFAULT_REQUIREMENTS.maximum_final_loss + 0.01,
                ),
                ExperimentVerdict.TRAINING_FAILED,
            ),
            (
                replace(
                    observed,
                    test_accuracy=DEFAULT_REQUIREMENTS.minimum_test_accuracy - 0.01,
                ),
                ExperimentVerdict.TRAINING_FAILED,
            ),
            (
                replace(
                    observed,
                    test_ece_after=DEFAULT_REQUIREMENTS.maximum_test_ece + 0.01,
                ),
                ExperimentVerdict.CALIBRATION_FAILED,
            ),
            (
                replace(
                    observed,
                    mean_logit_drift=(
                        DEFAULT_REQUIREMENTS.maximum_mean_logit_drift + 0.01
                    ),
                ),
                ExperimentVerdict.QUANTIZATION_FAILED,
            ),
        )
        for changed, expected in cases:
            with self.subTest(expected=expected):
                verdict, _reason = verdict_for(DEFAULT_REQUIREMENTS, changed)
                self.assertEqual(verdict, expected)

    def test_nonfinite_evidence_is_invalid(self) -> None:
        observed = replace(self.report.observations, test_ece_after=math.nan)

        verdict, _reason = verdict_for(DEFAULT_REQUIREMENTS, observed)

        self.assertEqual(verdict, ExperimentVerdict.INVALID)

    def test_reused_split_is_invalid(self) -> None:
        observed = replace(
            self.report.observations,
            test_split=self.report.observations.calibration_split,
        )

        verdict, _reason = verdict_for(DEFAULT_REQUIREMENTS, observed)

        self.assertEqual(verdict, ExperimentVerdict.INVALID)

    def test_stricter_policy_changes_a_passing_verdict(self) -> None:
        stricter = replace(
            DEFAULT_REQUIREMENTS,
            maximum_mean_logit_drift=self.report.observations.mean_logit_drift - 1e-6,
        )

        verdict, _reason = verdict_for(stricter, self.report.observations)

        self.assertEqual(verdict, ExperimentVerdict.QUANTIZATION_FAILED)


if __name__ == "__main__":
    unittest.main()
