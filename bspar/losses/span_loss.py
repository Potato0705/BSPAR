"""L_span: Focal BCE loss for dual binary span scoring heads.

Uses focal loss to handle the severe class imbalance where most spans
are neither aspects nor opinions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanFocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_bce(self, logits, targets):
        """Focal binary cross-entropy.

        Args:
            logits: (batch, num_spans) — raw scores (pre-sigmoid)
            targets: (batch, num_spans) — 0/1 labels
        """
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * ce).mean()

    def forward(self, asp_scores, asp_labels, opn_scores, opn_labels):
        """
        Args:
            asp_scores: (batch, num_spans) — aspect unary scores
            asp_labels: (batch, num_spans) — 1 if span is a gold aspect
            opn_scores: (batch, num_spans) — opinion unary scores
            opn_labels: (batch, num_spans) — 1 if span is a gold opinion

        Returns:
            Scalar loss
        """
        loss_asp = self.focal_bce(asp_scores, asp_labels.float())
        loss_opn = self.focal_bce(opn_scores, opn_labels.float())
        return loss_asp + loss_opn
