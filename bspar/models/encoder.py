"""Shared contextual encoder wrapping a pretrained transformer."""

import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SharedEncoder(nn.Module):
    """Wraps a pretrained transformer as the shared backbone.

    The encoder produces contextualized token representations H ∈ R^{n×d}
    that are consumed by all downstream modules.
    """

    def __init__(self, model_name: str, finetune: bool = True):
        super().__init__()
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
        except (OSError, Exception):
            # Fallback: create model from config with random weights
            # This allows offline development and testing
            try:
                config = AutoConfig.from_pretrained(model_name)
            except (OSError, Exception):
                # Fully offline: create a small RoBERTa-like config
                from transformers import RobertaConfig
                config = RobertaConfig(
                    vocab_size=50265,
                    hidden_size=768,
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=514,
                )
                print(f"Warning: cannot load '{model_name}', "
                      f"using random RoBERTa-like model (6 layers)")
            self.transformer = AutoModel.from_config(config)
        self.hidden_size = self.transformer.config.hidden_size
        if not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len), optional

        Returns:
            H: (batch, seq_len, hidden_size)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state
