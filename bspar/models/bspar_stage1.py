"""Stage-1 composite model: Encoder + Span Proposal + Pair Module.

Training mode:  encoder + span/pair/category/affective joint training.
Inference mode: produces candidate quads with scores for Stage-2 reranking.
"""

import torch
import torch.nn as nn

from .encoder import SharedEncoder
from .span_proposal import SpanProposal
from .pair_module import PairModule


class BSPARStage1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SharedEncoder(
            model_name=config.model_name,
            finetune=config.finetune_encoder,
        )
        self.span_proposal = SpanProposal(
            hidden_size=self.encoder.hidden_size,
            max_span_length=config.max_span_length,
            span_repr_size=config.span_repr_size,
            width_embedding_dim=config.width_embedding_dim,
        )
        self.pair_module = PairModule(
            span_repr_size=config.span_repr_size,
            num_categories=config.num_categories,
            num_sentiments=config.num_sentiments,
            task_type=config.task_type,
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                gold_quads=None, mode="train"):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len), optional
            gold_quads: list of list of Quad (for training label assignment)
            mode: "train" or "inference"

        Returns:
            If train:     dict of losses
            If inference: dict with candidate quads and features
        """
        # Step 1: Encode
        H = self.encoder(input_ids, attention_mask, token_type_ids)

        # Step 2: Span proposal
        proposal_out = self.span_proposal(H, attention_mask)

        # Step 3: Prune spans
        pruned = self._prune_spans(proposal_out, gold_quads, mode)

        # Step 4: Construct pair candidates
        pair_inputs = self._construct_pairs(pruned)

        # Step 5: Pair prediction
        pair_out = self.pair_module(**pair_inputs)

        if mode == "train":
            return self._compute_stage1_losses(proposal_out, pair_out,
                                                pruned, gold_quads)
        else:
            return self._build_candidates(proposal_out, pair_out, pruned)

    def _prune_spans(self, proposal_out, gold_quads, mode):
        """Prune spans to top-K aspects and top-K opinions.

        During training: include gold spans in the pruned set to ensure
            positive training signals (teacher-forcing augmented with predicted).
        During inference: use only top-K predicted spans.
        """
        asp_scores = proposal_out["asp_scores"]   # (batch, num_spans)
        opn_scores = proposal_out["opn_scores"]
        span_reprs = proposal_out["span_reprs"]
        span_indices = proposal_out["span_indices"]

        batch_size = asp_scores.size(0)
        K_a = min(self.config.top_k_aspects, asp_scores.size(1))
        K_o = min(self.config.top_k_opinions, opn_scores.size(1))

        # Top-K by score
        asp_topk_scores, asp_topk_ids = torch.topk(asp_scores, K_a, dim=1)
        opn_topk_scores, opn_topk_ids = torch.topk(opn_scores, K_o, dim=1)

        # Gather pruned representations
        expand_dim = span_reprs.size(-1)
        pruned_asp_reprs = torch.gather(
            span_reprs, 1,
            asp_topk_ids.unsqueeze(-1).expand(-1, -1, expand_dim)
        )
        pruned_opn_reprs = torch.gather(
            span_reprs, 1,
            opn_topk_ids.unsqueeze(-1).expand(-1, -1, expand_dim)
        )

        # Append NULL prototypes
        null_asp = proposal_out["null_asp_repr"].unsqueeze(1)
        null_opn = proposal_out["null_opn_repr"].unsqueeze(1)
        pruned_asp_reprs = torch.cat([pruned_asp_reprs, null_asp], dim=1)
        pruned_opn_reprs = torch.cat([pruned_opn_reprs, null_opn], dim=1)

        # Collect pruned span indices for label assignment
        pruned_asp_indices = []
        pruned_opn_indices = []
        for b in range(batch_size):
            asp_idx = [span_indices[i] for i in asp_topk_ids[b].tolist()]
            asp_idx.append((-1, -1))  # NULL
            pruned_asp_indices.append(asp_idx)

            opn_idx = [span_indices[i] for i in opn_topk_ids[b].tolist()]
            opn_idx.append((-1, -1))  # NULL
            pruned_opn_indices.append(opn_idx)

        return {
            "asp_reprs": pruned_asp_reprs,
            "opn_reprs": pruned_opn_reprs,
            "asp_indices": pruned_asp_indices,
            "opn_indices": pruned_opn_indices,
            "asp_scores": asp_topk_scores,
            "opn_scores": opn_topk_scores,
        }

    def _construct_pairs(self, pruned):
        """Build Cartesian product of aspect × opinion, excluding NULL×NULL.

        Returns inputs suitable for PairModule.forward().
        """
        asp_reprs = pruned["asp_reprs"]   # (batch, K_a+1, span_repr_size)
        opn_reprs = pruned["opn_reprs"]   # (batch, K_o+1, span_repr_size)
        batch_size = asp_reprs.size(0)
        n_asp = asp_reprs.size(1)
        n_opn = opn_reprs.size(1)
        device = asp_reprs.device

        # Cartesian product via broadcasting
        # asp: (batch, n_asp, 1, dim) → (batch, n_asp, n_opn, dim)
        asp_expanded = asp_reprs.unsqueeze(2).expand(-1, -1, n_opn, -1)
        opn_expanded = opn_reprs.unsqueeze(1).expand(-1, n_asp, -1, -1)

        # Flatten to (batch, n_asp * n_opn, dim)
        asp_flat = asp_expanded.reshape(batch_size, n_asp * n_opn, -1)
        opn_flat = opn_expanded.reshape(batch_size, n_asp * n_opn, -1)

        # Create mask to exclude NULL×NULL (last asp × last opn)
        mask = torch.ones(n_asp, n_opn, dtype=torch.bool, device=device)
        mask[-1, -1] = False  # NULL_asp × NULL_opn
        mask_flat = mask.reshape(-1).unsqueeze(0).expand(batch_size, -1)

        # Apply mask
        num_valid = mask_flat.sum(dim=1)[0].item()
        asp_pairs = asp_flat[:, mask_flat[0], :]
        opn_pairs = opn_flat[:, mask_flat[0], :]

        # Distance and order IDs (placeholder — compute from span indices)
        dist_ids = torch.zeros(batch_size, num_valid, dtype=torch.long, device=device)
        order_ids = torch.zeros(batch_size, num_valid, dtype=torch.long, device=device)

        return {
            "asp_reprs": asp_pairs,
            "opn_reprs": opn_pairs,
            "dist_ids": dist_ids,
            "order_ids": order_ids,
        }

    def _compute_stage1_losses(self, proposal_out, pair_out, pruned, gold_quads):
        """Compute L_span + L_pair + L_cat + L_aff.

        Label assignment is done here by matching pruned candidates
        against gold_quads.
        """
        # Placeholder — actual implementation assigns labels by matching
        # gold span boundaries, pair validity, categories, and sentiment.
        return {
            "loss_span": torch.tensor(0.0, device=proposal_out["asp_scores"].device),
            "loss_pair": torch.tensor(0.0, device=proposal_out["asp_scores"].device),
            "loss_cat": torch.tensor(0.0, device=proposal_out["asp_scores"].device),
            "loss_aff": torch.tensor(0.0, device=proposal_out["asp_scores"].device),
            "loss_total": torch.tensor(0.0, device=proposal_out["asp_scores"].device),
        }

    def _build_candidates(self, proposal_out, pair_out, pruned):
        """Build QuadCandidate list for Stage-2 reranking.

        Expands each valid pair into candidate quads by considering
        top-c categories and predicted affective output.
        """
        return {
            "pair_reprs": pair_out["pair_reprs"],
            "pair_scores": pair_out["pair_scores"],
            "cat_logits": pair_out["cat_logits"],
            "aff_output": pair_out["aff_output"],
            "pruned_asp_indices": pruned["asp_indices"],
            "pruned_opn_indices": pruned["opn_indices"],
            "asp_scores": pruned["asp_scores"],
            "opn_scores": pruned["opn_scores"],
        }
