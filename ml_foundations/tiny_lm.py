"""Join token embeddings, causal transformer blocks, and next-token logits.

The model maps integer token ids to vectors, adds learned position vectors, applies
causal blocks, and projects each position to one vocabulary row of logits. Training
shifts the same sequence by one token and minimizes cross-entropy. The architecture is
large enough to expose real autograd and optimizer state, but far too small and too
narrowly trained to be a useful language model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import TinyTransformerBlock


@dataclass(frozen=True)
class TinyLMConfig:
    """All shape-defining facts for one tiny language model."""

    vocabulary_size: int
    maximum_sequence: int
    model_width: int
    heads: int
    feed_forward_width: int
    layers: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must contain at least two tokens")
        if self.model_width % self.heads != 0:
            raise ValueError("model_width must divide evenly across heads")


class TinyLanguageModel(nn.Module):
    """Produce causal next-token logits under one immutable shape configuration."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        if not isinstance(config, TinyLMConfig):
            raise TypeError("config must be a TinyLMConfig")
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.model_width
        )
        self.position_embedding = nn.Embedding(
            config.maximum_sequence, config.model_width
        )
        self.blocks = nn.ModuleList(
            TinyTransformerBlock(
                config.model_width, config.heads, config.feed_forward_width
            )
            for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.model_width)
        self.output = nn.Linear(
            config.model_width, config.vocabulary_size, bias=False
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return batch by sequence by vocabulary logits for valid integer ids."""

        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape batch by nonempty sequence")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        if token_ids.shape[1] > self.config.maximum_sequence:
            raise ValueError("sequence exceeds the configured maximum")
        if torch.any(token_ids < 0) or torch.any(
            token_ids >= self.config.vocabulary_size
        ):
            raise ValueError("token id falls outside the configured vocabulary")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden, causal=True)
        return self.output(self.final_norm(hidden))

    def next_token_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return mean shifted cross-entropy for sequences with at least two tokens."""

        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise ValueError("training tokens must have shape batch by sequence")
        if token_ids.shape[1] < 2:
            raise ValueError("next-token loss needs at least two positions")
        logits = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocabulary_size),
            targets.reshape(-1),
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of learned scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())
