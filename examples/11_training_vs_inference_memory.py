"""Account for retained training and inference memory by component.

Prediction: training state exceeds a weights-only figure, while doubling cached
tokens doubles only the KV component. Run with
`python examples/11_training_vs_inference_memory.py`.
"""

from ml_foundations import estimate_inference_memory, estimate_training_memory


def gibibytes(byte_count: int) -> float:
    return byte_count / 1024**3


def main() -> None:
    parameters = 1_000_000_000
    training = estimate_training_memory(
        parameters,
        activation_elements=600_000_000,
    )
    short_context = estimate_inference_memory(
        parameters,
        kv_elements=200_000_000,
        activation_elements=10_000_000,
    )
    long_context = estimate_inference_memory(
        parameters,
        kv_elements=400_000_000,
        activation_elements=10_000_000,
    )
    short_kv = next(
        item.bytes for item in short_context.components if item.name == "kv_cache"
    )
    long_kv = next(
        item.bytes for item in long_context.components if item.name == "kv_cache"
    )

    print("Retained memory estimates")
    for component in training.components:
        print(f"  training {component.name}: {gibibytes(component.bytes):.2f} GiB")
    print(f"  training total: {gibibytes(training.total_bytes):.2f} GiB")
    print(f"  inference total: {gibibytes(short_context.total_bytes):.2f} GiB")
    print(f"  doubled tokens double KV bytes: {long_kv == 2 * short_kv}")
    print("\nTakeaway: name the retained state before handing a byte figure to capacity planning.")


if __name__ == "__main__":
    main()
