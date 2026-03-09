"""Stage-2 composite model: Quad-Aware Reranker.

Operates on pre-computed candidate quads from Stage-1 (real candidates).
Computes quad-level scores and applies pairwise margin ranking loss.
"""

import torch
import torch.nn as nn

from .quad_reranker import QuadReranker


class BSPARStage2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reranker = QuadReranker(
            pair_repr_size=config.pair_repr_size,
            num_categories=config.num_categories,
            cat_embedding_dim=config.cat_embedding_dim,
            aff_embedding_dim=config.aff_embedding_dim,
            num_meta_features=config.num_meta_features,
            task_type=config.task_type,
            num_sentiments=config.num_sentiments,
        )
        self.margin = config.ranking_margin

    def forward(self, pair_reprs, cat_ids, aff_input, meta_features,
                labels=None, mode="train"):
        """
        Args:
            pair_reprs:    (batch, num_cands, pair_repr_size)
            cat_ids:       (batch, num_cands) — category indices
            aff_input:     sentiment indices or (v, ar) values
            meta_features: (batch, num_cands, num_meta_features)
            labels:        (batch, num_cands) — 1 for gold, 0 for negative
            mode:          "train" or "inference"

        Returns:
            dict with quad_scores, and loss if training
        """
        quad_scores = self.reranker(pair_reprs, cat_ids, aff_input, meta_features)

        result = {"quad_scores": quad_scores}

        if mode == "train" and labels is not None:
            loss = self._compute_ranking_loss(quad_scores, labels)
            result["loss"] = loss

        return result

    def _compute_ranking_loss(self, scores, labels):
        """Pairwise margin ranking loss.

        For each (positive, negative) pair within a batch example:
            loss = max(0, margin - S(q+) + S(q-))

        Args:
            scores: (batch, num_cands)
            labels: (batch, num_cands) — 1 for gold-matched, 0 for negative
        """
        batch_size = scores.size(0)
        total_loss = torch.tensor(0.0, device=scores.device)
        num_pairs = 0

        for b in range(batch_size):
            pos_mask = labels[b] == 1
            neg_mask = labels[b] == 0

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_scores = scores[b][pos_mask]    # (num_pos,)
            neg_scores = scores[b][neg_mask]    # (num_neg,)

            # All positive-negative pairs
            # pos: (num_pos, 1), neg: (1, num_neg)
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            pair_loss = torch.clamp(self.margin - diff, min=0.0)
            total_loss = total_loss + pair_loss.sum()
            num_pairs += pair_loss.numel()

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss
