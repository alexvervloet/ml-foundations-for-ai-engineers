"""Tests for transformer residual, shape, and causal contracts."""

import unittest

import torch

from ml_foundations import TinyLanguageModel, TinyLMConfig, TinyTransformerBlock


class TransformerBlockTests(unittest.TestCase):
    def test_block_preserves_shape(self) -> None:
        torch.manual_seed(3)
        block = TinyTransformerBlock(8, 2, 16)
        tokens = torch.randn(2, 5, 8)
        self.assertEqual(block(tokens).shape, tokens.shape)

    def test_zeroed_block_is_residual_identity(self) -> None:
        block = TinyTransformerBlock(8, 2, 16)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        tokens = torch.randn(2, 4, 8)
        torch.testing.assert_close(block(tokens), tokens, rtol=0.0, atol=0.0)

    def test_block_rejects_invalid_construction_and_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            TinyTransformerBlock(7, 2, 16)
        block = TinyTransformerBlock(8, 2, 16)
        with self.assertRaisesRegex(ValueError, "model width"):
            block(torch.randn(1, 2, 7))
        with self.assertRaisesRegex(TypeError, "floating"):
            block(torch.ones(1, 2, 8, dtype=torch.long))


class TinyLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.config = TinyLMConfig(9, 8, 8, 2, 16, 2)
        self.model = TinyLanguageModel(self.config).eval()

    def test_logits_have_batch_sequence_vocabulary_shape(self) -> None:
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        self.assertEqual(self.model(tokens).shape, (2, 3, 9))
        self.assertGreater(self.model.parameter_count, 0)

    def test_future_token_cannot_change_earlier_logits(self) -> None:
        original = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        changed = torch.tensor([[0, 1, 8, 7]], dtype=torch.long)
        with torch.inference_mode():
            original_logits = self.model(original)
            changed_logits = self.model(changed)
        torch.testing.assert_close(
            original_logits[:, :2], changed_logits[:, :2], rtol=0.0, atol=0.0
        )

    def test_model_rejects_invalid_token_contracts(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch.long"):
            self.model(torch.tensor([[0.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.model(torch.tensor([[0, 9]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "maximum"):
            self.model(torch.zeros((1, 9), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.model.next_token_loss(torch.tensor([[1]], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
