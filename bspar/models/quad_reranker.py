"""Module C: Quad-Aware Reranker.

Takes candidate quads from Stage-1 real outputs and computes a unified
quad-level score S(q) integrating pair representation, category/affective
embeddings, and meta features.

Key design principle: trained on Stage-1 actual outputs, NOT oracle candidates,
to ensure training-inference consistency (Contribution C3).
"""

import torch
import torch.nn as nn


class QuadReranker(nn.Module):

    def __init__(self, pair_repr_size: int, num_categories: int,
                 cat_embedding_dim: int = 32, aff_embedding_dim: int = 32,
                 num_meta_features: int = 9, task_type: str = "asqp",
                 num_sentiments: int = 3):
        super().__init__()
        self.task_type = task_type

        # Category embedding for quad-level reasoning
        self.cat_embedding = nn.Embedding(num_categories, cat_embedding_dim)

        # Affective embedding
        if task_type == "asqp":
            self.aff_embedding = nn.Embedding(num_sentiments, aff_embedding_dim)
        else:
            self.aff_proj = nn.Linear(2, aff_embedding_dim)

        # Meta feature projection
        self.meta_proj = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.ReLU(),
        )

        # Quad scorer: pair_repr + cat_emb + aff_emb + meta → score
        quad_input_dim = pair_repr_size + cat_embedding_dim + aff_embedding_dim + 32
        self.quad_scorer = nn.Sequential(
            nn.Linear(quad_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, pair_reprs: torch.Tensor, cat_ids: torch.Tensor,
                aff_input: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_reprs:    (batch, num_cands, pair_repr_size)
            cat_ids:       (batch, num_cands) — category indices
            aff_input:     (batch, num_cands) for ASQP sentiment indices,
                           or (batch, num_cands, 2) for dimABSA (v, ar)
            meta_features: (batch, num_cands, num_meta_features)

        Returns:
            quad_scores: (batch, num_cands)
        """
        e_cat = self.cat_embedding(cat_ids)

        if self.task_type == "asqp":
            e_aff = self.aff_embedding(aff_input)
        else:
            e_aff = self.aff_proj(aff_input)

        e_meta = self.meta_proj(meta_features)

        quad_repr = torch.cat([pair_reprs, e_cat, e_aff, e_meta], dim=-1)
        quad_scores = self.quad_scorer(quad_repr).squeeze(-1)

        return quad_scores
