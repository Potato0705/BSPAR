"""L_cat: Category cross-entropy loss, conditioned on valid pairs.

Only computed on pairs that are labeled as valid (gold-positive),
since category prediction is meaningful only for actual aspect-opinion pairs.
"""

import torch
import torch.nn as nn


class CategoryCELoss(nn.Module):

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, cat_logits, cat_labels):
        """
        Args:
            cat_logits: (num_valid_pairs, num_categories) — pre-filtered
            cat_labels: (num_valid_pairs,) — gold category indices

        Returns:
            Scalar loss
        """
        if cat_logits.size(0) == 0:
            return torch.tensor(0.0, device=cat_logits.device)
        return self.ce(cat_logits, cat_labels)
