"""Train, evaluate, calibrate, quantize, and account for a tiny transformer.

Run from the repository root:

    python hands_on/train_tiny_transformer.py

The corpus contains rotations of one cyclic token rule. Training, calibration, and
test rows have distinct starting-token identities. The trained model classifies the
calibration rows perfectly, so calibration loss keeps falling as temperature shrinks
and the fit lands on the grid floor. The report records that rather than presenting an
unresolved fit as a chosen temperature.

The model receives next-token labels during training, which is legitimate supervision.
It never receives the independent loss, accuracy, calibration, drift, or split-lineage
requirements that decide the report. The experiment runs on CPU and writes a
deterministic training-report.json.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path

import numpy as np
import torch

from ml_foundations import (
    TinyLanguageModel,
    TinyLMConfig,
    estimate_inference_memory,
    estimate_training_memory,
    expected_calibration_error,
    fit_temperature,
    quantize_symmetric,
    sample_from_logits,
)


ROOT = Path(__file__).parents[1]
REPORT_PATH = ROOT / "training-report.json"


class ExperimentVerdict(str, Enum):
    """Name the first independent requirement the report did not meet."""

    INVALID = "invalid"
    TRAINING_FAILED = "training_failed"
    CALIBRATION_FAILED = "calibration_failed"
    QUANTIZATION_FAILED = "quantization_failed"
    READY_FOR_LAB_USE = "ready_for_lab_use"


@dataclass(frozen=True)
class ExperimentRequirements:
    """Requirements declared without reference to generated observations."""

    maximum_final_loss: float
    minimum_test_accuracy: float
    maximum_test_ece: float
    maximum_mean_logit_drift: float


@dataclass(frozen=True)
class ExperimentObservations:
    """Measured results and split lineage consumed by the verdict."""

    initial_loss: float
    final_loss: float
    test_accuracy: float
    fitted_temperature: float
    test_ece_before: float
    test_ece_after: float
    temperature_on_grid_boundary: bool
    mean_logit_drift: float
    source_weight_bytes: int
    quantized_payload_bytes: int
    training_memory_bytes: int
    inference_memory_bytes: int
    train_split: str
    calibration_split: str
    test_split: str
    sampled_tokens: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentReport:
    """Requirements, observations, verdict, and a human-readable reason."""

    requirements: ExperimentRequirements
    observations: ExperimentObservations
    verdict: ExperimentVerdict
    reason: str


DEFAULT_REQUIREMENTS = ExperimentRequirements(
    maximum_final_loss=0.20,
    minimum_test_accuracy=0.95,
    maximum_test_ece=0.15,
    maximum_mean_logit_drift=0.03,
)


def verdict_for(
    requirements: ExperimentRequirements,
    observations: ExperimentObservations,
) -> tuple[ExperimentVerdict, str]:
    """Derive the report verdict from independent bounds and measured evidence.

    Validation and split lineage run first. Training quality, calibration, and
    quantization follow in that order, so one report has one stable primary reason.
    Equality passes every bound. Ready means this tiny lab met this declared contract.
    It does not authorize a production model or generalize beyond the test process.
    """

    numeric = (
        observations.initial_loss,
        observations.final_loss,
        observations.test_accuracy,
        observations.fitted_temperature,
        observations.test_ece_before,
        observations.test_ece_after,
        observations.mean_logit_drift,
    )
    splits = (
        observations.train_split,
        observations.calibration_split,
        observations.test_split,
    )
    if not all(math.isfinite(value) for value in numeric):
        return ExperimentVerdict.INVALID, "report contains a nonfinite measurement"
    if len(set(splits)) != 3 or any(not split for split in splits):
        return ExperimentVerdict.INVALID, "train, calibration, and test splits must differ"
    if (
        observations.final_loss > requirements.maximum_final_loss
        or observations.test_accuracy < requirements.minimum_test_accuracy
    ):
        return (
            ExperimentVerdict.TRAINING_FAILED,
            "loss or held-out accuracy missed the training-quality requirement",
        )
    if observations.test_ece_after > requirements.maximum_test_ece:
        return (
            ExperimentVerdict.CALIBRATION_FAILED,
            "held-out calibration error exceeded its requirement",
        )
    if observations.mean_logit_drift > requirements.maximum_mean_logit_drift:
        return (
            ExperimentVerdict.QUANTIZATION_FAILED,
            "quantized logits drifted beyond their requirement",
        )
    return (
        ExperimentVerdict.READY_FOR_LAB_USE,
        "all independently declared lab requirements passed",
    )


def _cyclic_rows(vocabulary_size: int, sequence_length: int) -> list[list[int]]:
    """Create one row per start token under the rule next equals current plus one."""

    return [
        [
            (start + position) % vocabulary_size
            for position in range(sequence_length)
        ]
        for start in range(vocabulary_size)
    ]


def _numpy_logits_and_labels(
    model: TinyLanguageModel, rows: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits = model(rows[:, :-1])
    labels = rows[:, 1:]
    return (
        logits.reshape(-1, model.config.vocabulary_size).cpu().numpy(),
        labels.reshape(-1).cpu().numpy(),
    )


def _quantized_copy(
    model: TinyLanguageModel,
) -> tuple[TinyLanguageModel, int, int]:
    """Quantize every parameter to logical int8, then load reconstructed floats."""

    candidate = deepcopy(model)
    reconstructed: dict[str, torch.Tensor] = {}
    source_bytes = 0
    payload_bytes = 0
    for name, parameter in model.state_dict().items():
        source = parameter.detach().cpu().numpy()
        quantized = quantize_symmetric(source, bits=8)
        source_bytes += quantized.source_bytes
        payload_bytes += quantized.packed_payload_bytes
        reconstructed[name] = torch.from_numpy(quantized.dequantize()).to(
            dtype=parameter.dtype
        )
    candidate.load_state_dict(reconstructed)
    candidate.eval()
    return candidate, source_bytes, payload_bytes


def run_experiment(
    requirements: ExperimentRequirements = DEFAULT_REQUIREMENTS,
) -> ExperimentReport:
    """Run the complete deterministic CPU experiment without writing a file."""

    torch.manual_seed(23)
    torch.set_num_threads(1)
    config = TinyLMConfig(
        vocabulary_size=12,
        maximum_sequence=8,
        model_width=24,
        heads=4,
        feed_forward_width=48,
        layers=1,
    )
    rows = _cyclic_rows(config.vocabulary_size, config.maximum_sequence)
    train = torch.tensor(rows[:8], dtype=torch.long)
    calibration = torch.tensor(rows[8:10], dtype=torch.long)
    test = torch.tensor(rows[10:12], dtype=torch.long)

    model = TinyLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    initial_loss = float(model.next_token_loss(train).detach())
    for _step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(train)
        loss.backward()
        optimizer.step()
    final_loss = float(model.next_token_loss(train).detach())
    model.eval()

    calibration_logits, calibration_labels = _numpy_logits_and_labels(
        model, calibration
    )
    test_logits, test_labels = _numpy_logits_and_labels(model, test)
    temperature_fit = fit_temperature(
        calibration_logits,
        calibration_labels,
        temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    )
    predictions = np.argmax(test_logits, axis=1)
    accuracy = float(np.mean(predictions == test_labels))
    ece_before = expected_calibration_error(
        test_logits, test_labels, bins=8
    )
    ece_after = expected_calibration_error(
        test_logits,
        test_labels,
        bins=8,
        temperature=temperature_fit.temperature,
    )

    quantized_model, source_bytes, payload_bytes = _quantized_copy(model)
    quantized_logits, _labels = _numpy_logits_and_labels(quantized_model, test)
    logit_drift = float(np.mean(np.abs(test_logits - quantized_logits)))

    parameter_count = model.parameter_count
    activation_elements = train.numel() * config.model_width * config.layers * 6
    training_memory = estimate_training_memory(
        parameter_count, activation_elements
    )
    kv_elements = (
        2 * config.layers * config.maximum_sequence * config.model_width
    )
    inference_memory = estimate_inference_memory(
        parameter_count,
        kv_elements,
        config.maximum_sequence * config.model_width,
    )

    generated = [0, 1]
    rng = np.random.default_rng(31)
    while len(generated) < config.maximum_sequence:
        context = torch.tensor([generated], dtype=torch.long)
        with torch.inference_mode():
            next_logits = model(context)[0, -1].cpu().numpy()
        generated.append(
            sample_from_logits(
                next_logits,
                temperature=0.7,
                top_k=2,
                rng=rng,
            )
        )

    observations = ExperimentObservations(
        initial_loss=round(initial_loss, 8),
        final_loss=round(final_loss, 8),
        test_accuracy=round(accuracy, 8),
        fitted_temperature=temperature_fit.temperature,
        test_ece_before=round(ece_before, 8),
        test_ece_after=round(ece_after, 8),
        temperature_on_grid_boundary=temperature_fit.on_grid_boundary,
        mean_logit_drift=round(logit_drift, 8),
        source_weight_bytes=source_bytes,
        quantized_payload_bytes=payload_bytes,
        training_memory_bytes=training_memory.total_bytes,
        inference_memory_bytes=inference_memory.total_bytes,
        train_split="cyclic-rotations/train-starts-0-7",
        calibration_split="cyclic-rotations/calibration-starts-8-9",
        test_split="cyclic-rotations/test-starts-10-11",
        sampled_tokens=tuple(generated),
    )
    verdict, reason = verdict_for(requirements, observations)
    return ExperimentReport(requirements, observations, verdict, reason)


def _json_report(report: ExperimentReport) -> dict[str, object]:
    return {
        "requirements": asdict(report.requirements),
        "observations": asdict(report.observations),
        "verdict": report.verdict.value,
        "reason": report.reason,
    }


def main() -> int:
    report = run_experiment()
    REPORT_PATH.write_text(json.dumps(_json_report(report), indent=2) + "\n")
    observed = report.observations
    print("Tiny transformer experiment")
    print(f"  train loss:       {observed.initial_loss:.4f} -> {observed.final_loss:.4f}")
    print(f"  test accuracy:    {observed.test_accuracy:.1%}")
    fit_note = (
        "grid floor, unresolved"
        if observed.temperature_on_grid_boundary
        else "interior fit"
    )
    print(
        f"  test ECE:         {observed.test_ece_before:.4f} -> "
        f"{observed.test_ece_after:.4f} at T={observed.fitted_temperature:.2f} "
        f"({fit_note})"
    )
    print(f"  int8 logit drift: {observed.mean_logit_drift:.6f} mean absolute")
    print(
        f"  weight payload:   {observed.source_weight_bytes} -> "
        f"{observed.quantized_payload_bytes} bytes"
    )
    print(
        f"  memory estimate:  train={observed.training_memory_bytes} bytes, "
        f"inference={observed.inference_memory_bytes} bytes"
    )
    print(f"  sampled tokens:   {list(observed.sampled_tokens)}")
    print(f"  verdict:          {report.verdict.value}")
    print(f"  reason:           {report.reason}")
    print("\nThe report is lab evidence for this tiny dataset, not a production claim.")
    return 0 if report.verdict is ExperimentVerdict.READY_FOR_LAB_USE else 1


if __name__ == "__main__":
    raise SystemExit(main())
