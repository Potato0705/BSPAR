"""Module B: Pair Construction with Multi-Task Prediction Heads.

Takes pruned aspect/opinion span representations (including NULL prototypes),
constructs pair representations with structural features (distance, order),
and jointly predicts pair validity, category, and affective output.
"""

import torch
import torch.nn as nn


class PairModule(nn.Module):

    def __init__(self, span_repr_size: int, num_categories: int,
                 num_sentiments: int = 3, dist_buckets: int = 16,
                 order_types: int = 3, task_type: str = "asqp"):
        super().__init__()
        self.task_type = task_type
        self.num_categories = num_categories

        # Structural feature embeddings
        self.dist_embedding = nn.Embedding(dist_buckets, 32)
        self.order_embedding = nn.Embedding(order_types, 16)

        # Pair representation projection
        # Input: [r_a; r_o; r_a⊙r_o; |r_a-r_o|; e_dist; e_order]
        pair_input_dim = span_repr_size * 4 + 32 + 16
        self.pair_repr_size = span_repr_size * 2

        self.pair_proj = nn.Sequential(
            nn.Linear(pair_input_dim, self.pair_repr_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Prediction heads
        self.pair_scorer = nn.Linear(self.pair_repr_size, 1)
        self.cat_head = nn.Linear(self.pair_repr_size, num_categories)

        if task_type == "asqp":
            self.aff_head = nn.Linear(self.pair_repr_size, num_sentiments)
        else:  # dimabsa
            self.aff_head = nn.Linear(self.pair_repr_size, 2)  # (valence, arousal)

    def forward(self, asp_reprs: torch.Tensor, opn_reprs: torch.Tensor,
                dist_ids: torch.Tensor, order_ids: torch.Tensor) -> dict:
        """
        Args:
            asp_reprs:  (batch, num_pairs, span_repr_size)
            opn_reprs:  (batch, num_pairs, span_repr_size)
            dist_ids:   (batch, num_pairs) — distance bucket indices
            order_ids:  (batch, num_pairs) — order type indices

        Returns dict with:
            pair_reprs:  (batch, num_pairs, pair_repr_size)
            pair_scores: (batch, num_pairs)
            cat_logits:  (batch, num_pairs, num_categories)
            aff_output:  (batch, num_pairs, num_sentiments) or (batch, num_pairs, 2)
        """
        # Element-wise interactions
        hadamard = asp_reprs * opn_reprs
        abs_diff = (asp_reprs - opn_reprs).abs()

        # Structural embeddings
        e_dist = self.dist_embedding(dist_ids)
        e_order = self.order_embedding(order_ids)

        # Concatenate all components
        pair_input = torch.cat([
            asp_reprs, opn_reprs, hadamard, abs_diff, e_dist, e_order
        ], dim=-1)

        pair_reprs = self.pair_proj(pair_input)

        # Multi-task predictions
        pair_scores = self.pair_scorer(pair_reprs).squeeze(-1)
        cat_logits = self.cat_head(pair_reprs)
        aff_output = self.aff_head(pair_reprs)

        return {
            "pair_reprs": pair_reprs,
            "pair_scores": pair_scores,
            "cat_logits": cat_logits,
            "aff_output": aff_output,
        }
