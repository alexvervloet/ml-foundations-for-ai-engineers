"""Follow tensors through one causal transformer block.

Prediction: a block preserves shape, and its causal path prevents a changed future
token from altering earlier outputs. Run with `python examples/07_transformer_block.py`.
"""

import torch

from ml_foundations import TinyTransformerBlock


def main() -> None:
    torch.manual_seed(7)
    block = TinyTransformerBlock(model_width=8, heads=2, feed_forward_width=16)
    block.eval()
    tokens = torch.randn(1, 4, 8)
    changed = tokens.clone()
    changed[:, -1, :] += 100.0

    with torch.inference_mode():
        output = block(tokens, causal=True)
        counterfactual = block(changed, causal=True)
    earlier_gap = float(torch.max(torch.abs(output[:, :-1] - counterfactual[:, :-1])))

    print("One transformer block")
    print(f"  input shape:  {tuple(tokens.shape)}")
    print(f"  output shape: {tuple(output.shape)}")
    print(f"  heads by head width: {block.heads} by {block.head_width}")
    print(f"  largest earlier-token gap: {earlier_gap:.1e}")
    print("\nTakeaway: residual updates preserve shape; masking preserves causality.")


if __name__ == "__main__":
    main()
