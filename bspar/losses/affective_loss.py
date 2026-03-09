"""L_aff: Affective prediction loss.

ASQP: cross-entropy over sentiment polarity {POS, NEG, NEU}.
dimABSA: Smooth-L1 regression over (valence, arousal) ∈ [1, 5]².
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffectiveLoss(nn.Module):

    def __init__(self, task_type: str = "asqp"):
        super().__init__()
        self.task_type = task_type
        if task_type == "asqp":
            self.ce = nn.CrossEntropyLoss()

    def forward(self, aff_output, aff_labels):
        """
        Args:
            For ASQP:
                aff_output: (num_valid, num_sentiments) — logits
                aff_labels: (num_valid,) — sentiment indices
            For dimABSA:
                aff_output: (num_valid, 2) — predicted (v, ar)
                aff_labels: (num_valid, 2) — gold (v, ar)

        Returns:
            Scalar loss
        """
        if aff_output.size(0) == 0:
            return torch.tensor(0.0, device=aff_output.device)

        if self.task_type == "asqp":
            return self.ce(aff_output, aff_labels)
        else:
            return F.smooth_l1_loss(aff_output, aff_labels)
