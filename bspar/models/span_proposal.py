"""Module A: Structured Span Proposal with NULL Prototypes.

Enumerates candidate spans, computes span representations via boundary
tokens + attention pooling + width embedding, scores each span as aspect
or opinion candidate with dual binary heads, and generates context-conditioned
NULL prototypes for implicit element modeling.
"""

import torch
import torch.nn as nn


class SpanProposal(nn.Module):

    def __init__(self, hidden_size: int, max_span_length: int,
                 span_repr_size: int, width_embedding_dim: int = 32):
        super().__init__()
        self.max_span_length = max_span_length
        self.span_repr_size = span_repr_size

        # Width embedding: index 0 unused, 1..max_span_length for actual widths
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)

        # Attention pooling projection
        self.attn_proj = nn.Linear(hidden_size, 1)

        # Span representation: [h_start; h_end; attn_pool; e_width] → span_repr
        raw_dim = hidden_size * 3 + width_embedding_dim
        self.span_proj = nn.Sequential(
            nn.Linear(raw_dim, span_repr_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Dual binary scoring heads (C1: independent aspect/opinion scoring)
        self.asp_scorer = nn.Linear(span_repr_size, 1)
        self.opn_scorer = nn.Linear(span_repr_size, 1)

        # Context-conditioned NULL prototypes (C1: implicit element modeling)
        self.null_asp_proj = nn.Sequential(
            nn.Linear(hidden_size, span_repr_size),
            nn.Tanh(),
        )
        self.null_opn_proj = nn.Sequential(
            nn.Linear(hidden_size, span_repr_size),
            nn.Tanh(),
        )

    def enumerate_spans(self, seq_len: int) -> list[tuple[int, int]]:
        """Generate all valid (start, end) pairs with length <= max_span_length."""
        spans = []
        for i in range(seq_len):
            for j in range(i, min(i + self.max_span_length, seq_len)):
                spans.append((i, j))
        return spans

    def compute_span_reprs(self, H: torch.Tensor,
                           span_indices: list[tuple[int, int]]) -> torch.Tensor:
        """Compute span representations for all enumerated spans.

        Args:
            H: (batch, seq_len, hidden_size)
            span_indices: list of (start, end) tuples

        Returns:
            (batch, num_spans, span_repr_size)
        """
        batch_size, _, hidden = H.shape
        device = H.device

        starts = torch.tensor([s for s, e in span_indices], device=device)
        ends = torch.tensor([e for s, e in span_indices], device=device)
        widths = ends - starts + 1

        # Boundary representations
        h_starts = H[:, starts, :]      # (batch, num_spans, hidden)
        h_ends = H[:, ends, :]          # (batch, num_spans, hidden)

        # Attention-pooled representation per span
        attn_pooled = []
        for (i, j) in span_indices:
            span_tokens = H[:, i:j+1, :]                   # (batch, span_len, hidden)
            attn_weights = self.attn_proj(span_tokens)      # (batch, span_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = (attn_weights * span_tokens).sum(dim=1)  # (batch, hidden)
            attn_pooled.append(pooled)
        attn_pooled = torch.stack(attn_pooled, dim=1)       # (batch, num_spans, hidden)

        # Width embeddings
        e_width = self.width_embedding(widths)              # (num_spans, width_dim)
        e_width = e_width.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate and project
        raw = torch.cat([h_starts, h_ends, attn_pooled, e_width], dim=-1)
        return self.span_proj(raw)

    def compute_null_prototypes(self, H: torch.Tensor,
                                attention_mask: torch.Tensor):
        """Generate context-conditioned NULL prototypes.

        NULL prototypes are conditioned on the sentence-level mean representation,
        so that implicit element semantics are context-dependent.

        Args:
            H: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)

        Returns:
            null_asp_repr: (batch, span_repr_size)
            null_opn_repr: (batch, span_repr_size)
        """
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_mean = (H * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        null_asp = self.null_asp_proj(h_mean)
        null_opn = self.null_opn_proj(h_mean)
        return null_asp, null_opn

    def forward(self, H: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Args:
            H: (batch, seq_len, hidden_size) from encoder
            attention_mask: (batch, seq_len)

        Returns dict with:
            span_reprs:    (batch, num_spans, span_repr_size)
            asp_scores:    (batch, num_spans)
            opn_scores:    (batch, num_spans)
            null_asp_repr: (batch, span_repr_size)
            null_opn_repr: (batch, span_repr_size)
            span_indices:  list of (start, end) tuples
        """
        seq_len = H.size(1)
        span_indices = self.enumerate_spans(seq_len)

        span_reprs = self.compute_span_reprs(H, span_indices)
        asp_scores = self.asp_scorer(span_reprs).squeeze(-1)
        opn_scores = self.opn_scorer(span_reprs).squeeze(-1)
        null_asp, null_opn = self.compute_null_prototypes(H, attention_mask)

        return {
            "span_reprs": span_reprs,
            "asp_scores": asp_scores,
            "opn_scores": opn_scores,
            "null_asp_repr": null_asp,
            "null_opn_repr": null_opn,
            "span_indices": span_indices,
        }
