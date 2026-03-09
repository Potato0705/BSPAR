"""L_pair: BCE loss with hard-negative weighting for pair validity.

Hard negatives (e.g., correct aspect + wrong opinion, near-boundary spans)
receive higher weight to improve pair discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairBCELoss(nn.Module):

    def __init__(self, hard_neg_weight: float = 3.0):
        super().__init__()
        self.hard_neg_weight = hard_neg_weight

    def forward(self, pair_scores, pair_labels, hard_neg_mask=None):
        """
        Args:
            pair_scores: (batch, num_pairs) — raw pair validity scores
            pair_labels: (batch, num_pairs) — 1 for valid pair, 0 for invalid
            hard_neg_mask: (batch, num_pairs) — 1 for hard negatives, optional

        Returns:
            Scalar loss
        """
        ce = F.binary_cross_entropy_with_logits(
            pair_scores, pair_labels.float(), reduction="none"
        )

        if hard_neg_mask is not None:
            # Up-weight hard negatives
            weight = torch.ones_like(ce)
            weight[hard_neg_mask.bool()] = self.hard_neg_weight
            ce = ce * weight

        return ce.mean()
