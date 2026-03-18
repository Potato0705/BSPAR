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
                 order_types: int = 3, task_type: str = "asqp",
                 use_acr_refine: bool = False,
                 use_acr_cat_refine: bool = False,
                 use_acr_aff_refine: bool = False,
                 acr_hidden_dim: int = 128,
                 acr_apply_to: str = "cat_aff",
                 acr_use_layernorm: bool = True,
                 use_early_interaction_prior: bool = False,
                 early_interaction_scale: float = 0.5,
                 early_interaction_cat_weight: float = 0.5,
                 early_interaction_aff_weight: float = 0.5,
                 early_interaction_detach: bool = True):
        super().__init__()
        self.task_type = task_type
        self.num_categories = num_categories

        # Backward compatibility:
        # legacy use_acr_refine=True means apply to both cat+aff unless split flags are set.
        if bool(use_acr_refine) and not bool(use_acr_cat_refine) and not bool(use_acr_aff_refine):
            use_acr_cat_refine = True
            use_acr_aff_refine = True
        self.use_acr_cat_refine = bool(use_acr_cat_refine)
        self.use_acr_aff_refine = bool(use_acr_aff_refine)
        self.use_acr_refine = self.use_acr_cat_refine or self.use_acr_aff_refine
        self.acr_apply_to = acr_apply_to
        self.use_early_interaction_prior = bool(use_early_interaction_prior)
        self.early_interaction_scale = float(early_interaction_scale)
        self.early_interaction_cat_weight = float(early_interaction_cat_weight)
        self.early_interaction_aff_weight = float(early_interaction_aff_weight)
        self.early_interaction_detach = bool(early_interaction_detach)

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

        if self.use_acr_refine:
            self.acr_norm = (
                nn.LayerNorm(self.pair_repr_size)
                if acr_use_layernorm
                else nn.Identity()
            )
            self.acr_gamma = nn.Sequential(
                nn.Linear(span_repr_size, acr_hidden_dim),
                nn.ReLU(),
                nn.Linear(acr_hidden_dim, self.pair_repr_size),
            )
            self.acr_beta = nn.Sequential(
                nn.Linear(span_repr_size, acr_hidden_dim),
                nn.ReLU(),
                nn.Linear(acr_hidden_dim, self.pair_repr_size),
            )
            # Start from identity-like refinement to avoid destabilizing old behavior.
            nn.init.zeros_(self.acr_gamma[-1].weight)
            nn.init.zeros_(self.acr_gamma[-1].bias)
            nn.init.zeros_(self.acr_beta[-1].weight)
            nn.init.zeros_(self.acr_beta[-1].bias)

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

        # Keep base pair scoring on original pair representations.
        pair_scores_base = self.pair_scorer(pair_reprs).squeeze(-1)

        # Optional aspect-conditioned refinement for materialization heads only.
        cat_aff_reprs = pair_reprs
        if self.use_acr_refine and self.acr_apply_to == "cat_aff":
            gamma = self.acr_gamma(asp_reprs)
            beta = self.acr_beta(asp_reprs)
            cat_aff_reprs = self.acr_norm(pair_reprs) * (1.0 + gamma) + beta

        cat_input = cat_aff_reprs if self.use_acr_cat_refine else pair_reprs
        aff_input = cat_aff_reprs if self.use_acr_aff_refine else pair_reprs

        cat_logits = self.cat_head(cat_input)
        aff_output = self.aff_head(aff_input)

        pair_scores = pair_scores_base
        early_prior = torch.zeros_like(pair_scores_base)
        if self.use_early_interaction_prior:
            # Early compatibility prior from cat/aff confidence margins.
            # This moves interaction signal into pre-retention pair ranking.
            cat_top2 = torch.topk(cat_logits, k=min(2, cat_logits.size(-1)), dim=-1).values
            cat_margin = cat_top2[..., 0] - (
                cat_top2[..., 1] if cat_top2.size(-1) > 1 else 0.0
            )
            cat_signal = torch.tanh(cat_margin)

            aff_top2 = torch.topk(aff_output, k=min(2, aff_output.size(-1)), dim=-1).values
            aff_margin = aff_top2[..., 0] - (
                aff_top2[..., 1] if aff_top2.size(-1) > 1 else 0.0
            )
            aff_signal = torch.tanh(aff_margin)

            early_prior = (
                self.early_interaction_cat_weight * cat_signal +
                self.early_interaction_aff_weight * aff_signal
            )
            if self.early_interaction_detach:
                early_prior = early_prior.detach()
            pair_scores = pair_scores_base + self.early_interaction_scale * early_prior

        return {
            "pair_reprs": pair_reprs,
            "pair_scores": pair_scores,
            "pair_scores_base": pair_scores_base,
            "early_interaction_prior": early_prior,
            "cat_logits": cat_logits,
            "aff_output": aff_output,
        }
