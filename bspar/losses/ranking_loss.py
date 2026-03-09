"""L_rank: Pairwise margin ranking loss for quad-aware reranking.

Optimizes: max(0, δ - S(q⁺) + S(q⁻)) for all (positive, negative) pairs
within each example.
"""

import torch
import torch.nn as nn


class PairwiseMarginRankingLoss(nn.Module):

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels):
        """
        Args:
            scores: (batch, num_cands) — quad scores S(q)
            labels: (batch, num_cands) — 1 for gold-matched, 0 for negative

        Returns:
            Scalar loss
        """
        batch_size = scores.size(0)
        total_loss = torch.tensor(0.0, device=scores.device)
        num_pairs = 0

        for b in range(batch_size):
            pos_mask = labels[b] == 1
            neg_mask = labels[b] == 0

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_scores = scores[b][pos_mask]
            neg_scores = scores[b][neg_mask]

            # Pairwise margin: (num_pos, 1) vs (1, num_neg)
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            pair_loss = torch.clamp(self.margin - diff, min=0.0)
            total_loss = total_loss + pair_loss.sum()
            num_pairs += pair_loss.numel()

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss
