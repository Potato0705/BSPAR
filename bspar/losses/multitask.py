"""Weighted multi-task loss combiner for Stage-1.

L = λ₁·L_span + λ₂·L_pair + λ₃·L_cat + λ₄·L_aff
"""

import torch
import torch.nn as nn

from .span_loss import SpanFocalLoss
from .pair_loss import PairBCELoss
from .category_loss import CategoryCELoss
from .affective_loss import AffectiveLoss


class MultiTaskLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.span_loss = SpanFocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
        self.pair_loss = PairBCELoss(
            hard_neg_weight=config.hard_neg_weight,
        )
        self.cat_loss = CategoryCELoss(label_smoothing=0.1)
        self.aff_loss = AffectiveLoss(task_type=config.task_type)

        self.lambdas = {
            "span": config.lambda_span,
            "pair": config.lambda_pair,
            "cat": config.lambda_cat,
            "aff": config.lambda_aff,
        }

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with asp_scores, opn_scores, pair_scores,
                         cat_logits, aff_output
            targets: dict with asp_labels, opn_labels, pair_labels,
                     cat_labels, aff_labels, valid_pair_mask,
                     hard_neg_mask (optional)
        Returns:
            dict of individual and total losses
        """
        losses = {}

        losses["span"] = self.span_loss(
            predictions["asp_scores"], targets["asp_labels"],
            predictions["opn_scores"], targets["opn_labels"],
        )

        losses["pair"] = self.pair_loss(
            predictions["pair_scores"], targets["pair_labels"],
            targets.get("hard_neg_mask"),
        )

        # Category and affective losses only on valid (gold-positive) pairs
        valid = targets["valid_pair_mask"]
        if valid.any():
            losses["cat"] = self.cat_loss(
                predictions["cat_logits"][valid],
                targets["cat_labels"][valid],
            )
            losses["aff"] = self.aff_loss(
                predictions["aff_output"][valid],
                targets["aff_labels"][valid],
            )
        else:
            device = predictions["asp_scores"].device
            losses["cat"] = torch.tensor(0.0, device=device)
            losses["aff"] = torch.tensor(0.0, device=device)

        losses["total"] = sum(
            self.lambdas[k] * losses[k] for k in self.lambdas
        )

        return losses
