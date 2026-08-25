"""Assemble attention, normalization, feed-forward updates, and residual paths.

A transformer block does not change sequence length or model width. It normalizes
tokens, mixes allowed positions with multi-head attention, adds that update back to
the residual stream, then repeats the pattern with a position-wise feed-forward
network. This small pre-norm block omits dropout, rotary positions, grouped-query
attention, fused kernels, and cache management.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TinyTransformerBlock(nn.Module):
    """Preserve batch, sequence, and model shape through one causal block.

    Construction rejects widths that cannot split evenly across heads. Forward
    validates rank, floating dtype, and model width before any projection. Setting
    causal to true prevents every position from reading later keys. A valid output
    proves tensor flow and masking, not that untrained weights encode useful behavior.
    """

    def __init__(self, model_width: int, heads: int, feed_forward_width: int) -> None:
        super().__init__()
        for name, value in (
            ("model_width", model_width),
            ("heads", heads),
            ("feed_forward_width", feed_forward_width),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if model_width % heads != 0:
            raise ValueError("model_width must divide evenly across heads")
        self.model_width = model_width
        self.heads = heads
        self.head_width = model_width // heads
        self.attention_norm = nn.LayerNorm(model_width)
        self.qkv = nn.Linear(model_width, model_width * 3)
        self.attention_output = nn.Linear(model_width, model_width)
        self.feed_forward_norm = nn.LayerNorm(model_width)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_width, feed_forward_width),
            nn.GELU(),
            nn.Linear(feed_forward_width, model_width),
        )

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _width = values.shape
        return (
            values.view(batch, sequence, self.heads, self.head_width)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, tokens: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        """Apply attention and feed-forward residual updates to floating tokens."""

        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape batch by sequence by model width")
        if tokens.shape[2] != self.model_width:
            raise ValueError("token width must match the configured model width")
        if not tokens.is_floating_point():
            raise TypeError("transformer block tokens must use a floating dtype")
        if not isinstance(causal, bool):
            raise TypeError("causal must be a Boolean")

        normalized = self.attention_norm(tokens)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(query),
            self._heads(key),
            self._heads(value),
            is_causal=causal,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_width),
        )
        batch, _heads, sequence, _head_width = attended.shape
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, sequence, self.model_width)
        )
        residual = tokens + self.attention_output(merged)
        return residual + self.feed_forward(self.feed_forward_norm(residual))
