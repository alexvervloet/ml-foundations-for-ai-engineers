"""Readable NumPy and PyTorch foundations behind language-model applications."""

from .attention import causal_mask, scaled_dot_product_attention
from .calibration import (
    TemperatureFit,
    expected_calibration_error,
    fit_temperature,
)
from .losses import cross_entropy_from_logits
from .memory import (
    MemoryComponent,
    MemoryEstimate,
    estimate_inference_memory,
    estimate_training_memory,
)
from .optimization import (
    OptimizationPoint,
    OptimizationResult,
    OptimizationStatus,
    finite_difference,
    gradient_descent,
)
from .probabilities import greedy_token, sample_from_logits, stable_softmax
from .quantization import QuantizedTensor, mean_absolute_error, quantize_symmetric
from .tensors import affine, cosine_similarity
from .tiny_lm import TinyLanguageModel, TinyLMConfig
from .transformer import TinyTransformerBlock


__all__ = [
    "MemoryComponent",
    "MemoryEstimate",
    "OptimizationPoint",
    "OptimizationResult",
    "OptimizationStatus",
    "QuantizedTensor",
    "TemperatureFit",
    "TinyLanguageModel",
    "TinyLMConfig",
    "TinyTransformerBlock",
    "affine",
    "causal_mask",
    "cosine_similarity",
    "cross_entropy_from_logits",
    "estimate_inference_memory",
    "estimate_training_memory",
    "expected_calibration_error",
    "finite_difference",
    "fit_temperature",
    "gradient_descent",
    "greedy_token",
    "mean_absolute_error",
    "quantize_symmetric",
    "sample_from_logits",
    "scaled_dot_product_attention",
    "stable_softmax",
]
