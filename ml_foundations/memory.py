"""Account for retained training and inference state in named byte components.

Parameter count alone does not describe memory. Mixed-precision Adam training may
retain weights, gradients, master weights, two optimizer moments, and saved
activations. Autoregressive inference retains weights, current activations, and a KV
cache that grows with cached tokens. These formulas omit allocator overhead,
temporary kernels, fragmentation, and framework-specific reuse. Measure those on the
target runtime before sizing hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryComponent:
    """One positive byte quantity with a concrete name."""

    name: str
    bytes: int


@dataclass(frozen=True)
class MemoryEstimate:
    """Named components and their exact byte total."""

    components: tuple[MemoryComponent, ...]

    @property
    def total_bytes(self) -> int:
        return sum(component.bytes for component in self.components)


def _count(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _width(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def estimate_training_memory(
    parameter_count: int,
    activation_elements: int,
    *,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    master_weight_bytes: int = 4,
    optimizer_state_bytes_per_parameter: int = 8,
    activation_bytes: int = 2,
) -> MemoryEstimate:
    """Estimate one-replica mixed-precision training state in bytes.

    Counts are validated before widths. Zero parameters or activations are accepted
    for boundary accounting. The default optimizer component represents two fp32 Adam
    moments. The estimate is additive and deliberately does not infer checkpointing,
    sharding, recomputation, or distributed replicas.
    """

    parameters = _count("parameter_count", parameter_count)
    activations = _count("activation_elements", activation_elements)
    widths = {
        "weights": _width("parameter_bytes", parameter_bytes),
        "gradients": _width("gradient_bytes", gradient_bytes),
        "master_weights": _width("master_weight_bytes", master_weight_bytes),
        "optimizer_states": _width(
            "optimizer_state_bytes_per_parameter",
            optimizer_state_bytes_per_parameter,
        ),
        "saved_activations": _width("activation_bytes", activation_bytes),
    }
    return MemoryEstimate(
        (
            MemoryComponent("weights", parameters * widths["weights"]),
            MemoryComponent("gradients", parameters * widths["gradients"]),
            MemoryComponent(
                "master_weights", parameters * widths["master_weights"]
            ),
            MemoryComponent(
                "optimizer_states", parameters * widths["optimizer_states"]
            ),
            MemoryComponent(
                "saved_activations", activations * widths["saved_activations"]
            ),
        )
    )


def estimate_inference_memory(
    parameter_count: int,
    kv_elements: int,
    activation_elements: int,
    *,
    parameter_bytes: int = 2,
    kv_bytes: int = 2,
    activation_bytes: int = 2,
) -> MemoryEstimate:
    """Estimate one-replica inference state in bytes for one declared KV scope.

    The caller supplies KV elements after applying layers, heads, token count, batch,
    and any sharding. Keeping that derived count explicit prevents this function from
    silently assuming a model layout. Success does not prove the runtime will fit,
    because temporary workspaces and allocator behavior are outside this estimate.
    """

    parameters = _count("parameter_count", parameter_count)
    kv = _count("kv_elements", kv_elements)
    activations = _count("activation_elements", activation_elements)
    return MemoryEstimate(
        (
            MemoryComponent(
                "weights", parameters * _width("parameter_bytes", parameter_bytes)
            ),
            MemoryComponent("kv_cache", kv * _width("kv_bytes", kv_bytes)),
            MemoryComponent(
                "current_activations",
                activations * _width("activation_bytes", activation_bytes),
            ),
        )
    )
