"""Stage-1 composite model: Encoder + Span Proposal + Pair Module.

Training mode:  encoder + span/pair/category/affective joint training.
Inference mode: produces candidate quads with scores for Stage-2 reranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SharedEncoder
from .span_proposal import SpanProposal
from .pair_module import PairModule
from ..data.span_utils import compute_distance_bucket, compute_order
from ..data.preprocessor import SENTIMENT_TO_ID


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
            use_acr_refine=getattr(config, "use_acr_refine", False),
            use_acr_cat_refine=getattr(config, "use_acr_cat_refine", False),
            use_acr_aff_refine=getattr(config, "use_acr_aff_refine", False),
            acr_hidden_dim=getattr(config, "acr_hidden_dim", 128),
            acr_apply_to=getattr(config, "acr_apply_to", "cat_aff"),
            acr_use_layernorm=getattr(config, "acr_use_layernorm", True),
            use_early_interaction_prior=getattr(
                config, "use_early_interaction_prior", False
            ),
            early_interaction_scale=getattr(config, "early_interaction_scale", 0.5),
            early_interaction_cat_weight=getattr(
                config, "early_interaction_cat_weight", 0.5
            ),
            early_interaction_aff_weight=getattr(
                config, "early_interaction_aff_weight", 0.5
            ),
            early_interaction_detach=getattr(
                config, "early_interaction_detach", True
            ),
        )
        # Materialization-aware auxiliary heads (training-only).
        # They read pair representations but do not affect inference decoding.
        self.ma_aux_cat_head = nn.Linear(
            self.pair_module.pair_repr_size, config.num_categories
        )
        self.ma_aux_sent_head = (
            nn.Linear(self.pair_module.pair_repr_size, config.num_sentiments)
            if config.task_type == "asqp"
            else None
        )

        # Loss components
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.hard_neg_weight = config.hard_neg_weight

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                gold_quads=None, cat_to_id=None, word_to_subword=None,
                mode="train", gold_injection_prob=1.0,
                pair_score_threshold=None,
                pair_retention_strategy=None,
                pair_top_n=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            gold_quads: list[list[Quad]] — gold quads per example (training)
            cat_to_id: dict — category name → id mapping
            word_to_subword: list[list[tuple]] — word-to-subword alignment
            mode: "train" or "inference"
            gold_injection_prob: probability of injecting each gold span (scheduled)
            pair_score_threshold: inference pair gate threshold override
            pair_retention_strategy: topn_only | pair_gate_only | pair_gate_topn
            pair_top_n: top-N pair retention cap for inference
        """
        # Step 1: Encode
        H = self.encoder(input_ids, attention_mask, token_type_ids)

        # Step 2: Span proposal
        proposal_out = self.span_proposal(H, attention_mask)

        # Step 3: Prune spans (train mode can force-include gold spans)
        pruned = self._prune_spans(
            proposal_out,
            gold_quads=gold_quads,
            word_to_subword=word_to_subword,
            mode=mode,
            gold_injection_prob=gold_injection_prob,
        )

        # Step 4: Construct pair candidates
        pair_inputs, pair_map, pair_valid_mask = self._construct_pairs(pruned)

        # Step 5: Pair prediction
        pair_out = self.pair_module(**pair_inputs)

        if mode == "train":
            return self._compute_stage1_losses(
                proposal_out, pair_out, pruned, pair_map,
                gold_quads, cat_to_id, word_to_subword,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            return self._build_candidates(
                proposal_out,
                pair_out,
                pruned,
                pair_map,
                word_to_subword=word_to_subword,
                pair_score_threshold=pair_score_threshold,
                pair_retention_strategy=pair_retention_strategy,
                pair_top_n=pair_top_n,
                pair_valid_mask=pair_valid_mask,
            )

    @staticmethod
    def _force_include_ids(selected_ids, required_ids):
        """Force-include required span ids into a fixed-size selected id list.

        Keeps list length unchanged by replacing low-priority selected ids.
        """
        selected = list(selected_ids)
        required = []
        seen = set()
        for rid in required_ids:
            if rid not in seen:
                required.append(rid)
                seen.add(rid)

        if not required:
            return selected

        selected_set = set(selected)
        required_set = set(required)
        replace_ptr = len(selected) - 1

        for rid in required:
            if rid in selected_set:
                continue

            while replace_ptr >= 0 and selected[replace_ptr] in required_set:
                replace_ptr -= 1

            if replace_ptr < 0:
                break

            old = selected[replace_ptr]
            selected_set.discard(old)
            selected[replace_ptr] = rid
            selected_set.add(rid)
            replace_ptr -= 1

        return selected

    def _prune_spans(self, proposal_out, gold_quads=None,
                     word_to_subword=None, mode="train",
                     gold_injection_prob=1.0):
        """Keep top-K spans, and in train mode force-include gold spans.

        Args:
            gold_injection_prob: probability of injecting each gold span.
                Scheduled from 1.0 (full teacher forcing) to 0.0 (no injection).
        """
        asp_scores = proposal_out["asp_scores"]
        opn_scores = proposal_out["opn_scores"]
        span_reprs = proposal_out["span_reprs"]
        span_indices = proposal_out["span_indices"]

        batch_size = asp_scores.size(0)
        K_a = min(int(self.config.top_k_aspects), asp_scores.size(1))
        base_opinion_topk = int(self.config.top_k_opinions)
        opinion_topk = base_opinion_topk + int(
            getattr(self.config, "opinion_span_topk_delta", 0)
        )
        opinion_topk = max(1, opinion_topk)
        K_o = min(opinion_topk, opn_scores.size(1))

        asp_topk_ids = torch.topk(asp_scores, K_a, dim=1).indices
        opn_topk_ids = torch.topk(opn_scores, K_o, dim=1).indices

        # Teacher-forcing on spans: preserve top-K size while injecting gold spans.
        # gold_injection_prob controls stochastic scheduling.
        if (mode == "train" and gold_quads is not None
                and word_to_subword is not None and gold_injection_prob > 0):
            span_index_map = {span: idx for idx, span in enumerate(span_indices)}
            import random

            for b in range(batch_size):
                w2s = word_to_subword[b]
                required_asp = []
                required_opn = []

                for q in gold_quads[b]:
                    # Stochastic injection: skip this gold span with (1 - prob)
                    if gold_injection_prob < 1.0 and random.random() > gold_injection_prob:
                        continue

                    if not q.aspect.is_null:
                        a_sub = self._word_span_to_subword(
                            q.aspect.start, q.aspect.end, w2s
                        )
                        if a_sub is not None and a_sub in span_index_map:
                            required_asp.append(span_index_map[a_sub])

                    if not q.opinion.is_null:
                        o_sub = self._word_span_to_subword(
                            q.opinion.start, q.opinion.end, w2s
                        )
                        if o_sub is not None and o_sub in span_index_map:
                            required_opn.append(span_index_map[o_sub])

                asp_ids = asp_topk_ids[b].tolist()
                opn_ids = opn_topk_ids[b].tolist()
                asp_ids = self._force_include_ids(asp_ids, required_asp)
                opn_ids = self._force_include_ids(opn_ids, required_opn)

                asp_topk_ids[b] = torch.tensor(
                    asp_ids, device=asp_topk_ids.device, dtype=asp_topk_ids.dtype
                )
                opn_topk_ids[b] = torch.tensor(
                    opn_ids, device=opn_topk_ids.device, dtype=opn_topk_ids.dtype
                )

        # Selective opinion backfill gate (aspect-uncovered).
        # Only delta-added opinions are gated; base opinions are always kept.
        base_count = min(base_opinion_topk, K_o)
        opn_is_delta_mask = torch.zeros(
            batch_size,
            K_o + 1,  # +1 for NULL slot appended later
            dtype=torch.bool,
            device=asp_scores.device,
        )
        if K_o > base_count:
            opn_is_delta_mask[:, base_count:K_o] = True

        use_aspect_displacement_gate = (
            mode == "inference" and
            bool(
                getattr(
                    self.config,
                    "opinion_backfill_use_aspect_displacement_gate",
                    False,
                )
            ) and
            int(getattr(self.config, "opinion_span_topk_delta", 0)) > 0 and
            K_o > base_count and
            K_a > 0 and
            base_count > 0
        )
        asp_uncovered_mask = torch.zeros(
            batch_size,
            K_a + 1,  # +1 for NULL slot appended later
            dtype=torch.bool,
            device=asp_scores.device,
        )
        use_aspect_uncovered_gate = (
            mode == "inference" and
            bool(getattr(self.config, "opinion_backfill_use_aspect_uncovered_gate", False)) and
            int(getattr(self.config, "opinion_span_topk_delta", 0)) > 0 and
            K_o > base_count and
            K_a > 0 and
            base_count > 0
        )
        if use_aspect_uncovered_gate or use_aspect_displacement_gate:
            support_top_n = max(int(getattr(self.config, "stage1_pair_top_n", 20)), 1)
            min_base_pairs = max(
                int(getattr(self.config, "opinion_backfill_aspect_min_base_pairs", 1)),
                1,
            )

            for b in range(batch_size):
                asp_ids = asp_topk_ids[b].tolist()
                base_opn_ids = opn_topk_ids[b][:base_count].tolist()
                if not asp_ids or not base_opn_ids:
                    asp_uncovered_mask[b, :K_a] = True
                    continue

                pair_owner_ai = []
                pair_asp_reprs = []
                pair_opn_reprs = []
                pair_dist_ids = []
                pair_order_ids = []

                for ai, asp_span_idx in enumerate(asp_ids):
                    asp_span = span_indices[asp_span_idx]
                    for opn_span_idx in base_opn_ids:
                        opn_span = span_indices[opn_span_idx]
                        pair_owner_ai.append(ai)
                        pair_asp_reprs.append(span_reprs[b, asp_span_idx])
                        pair_opn_reprs.append(span_reprs[b, opn_span_idx])
                        pair_dist_ids.append(
                            compute_distance_bucket(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )
                        pair_order_ids.append(
                            compute_order(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )

                if not pair_owner_ai:
                    asp_uncovered_mask[b, :K_a] = True
                    continue

                asp_gate = torch.stack(pair_asp_reprs, dim=0).unsqueeze(0)
                opn_gate = torch.stack(pair_opn_reprs, dim=0).unsqueeze(0)
                dist_gate = torch.tensor(
                    pair_dist_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)
                order_gate = torch.tensor(
                    pair_order_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)

                with torch.no_grad():
                    gate_out = self.pair_module(
                        asp_reprs=asp_gate,
                        opn_reprs=opn_gate,
                        dist_ids=dist_gate,
                        order_ids=order_gate,
                    )
                    gate_scores = torch.sigmoid(gate_out["pair_scores"][0]).tolist()

                ranked_ids = sorted(
                    range(len(gate_scores)),
                    key=lambda pid: gate_scores[pid],
                    reverse=True,
                )
                support_ids = ranked_ids[:min(support_top_n, len(ranked_ids))]
                asp_support = [0] * K_a
                for pid in support_ids:
                    asp_support[pair_owner_ai[pid]] += 1

                for ai in range(K_a):
                    if asp_support[ai] < min_base_pairs:
                        asp_uncovered_mask[b, ai] = True

        asp_delta_displacement_keep_mask = torch.zeros(
            batch_size,
            K_a + 1,  # +1 for NULL slot appended later
            K_o + 1,  # +1 for NULL slot appended later
            dtype=torch.bool,
            device=asp_scores.device,
        )
        asp_base_displacement_drop_mask = torch.zeros(
            batch_size,
            K_a + 1,  # +1 for NULL slot appended later
            K_o + 1,  # +1 for NULL slot appended later
            dtype=torch.bool,
            device=asp_scores.device,
        )
        if use_aspect_displacement_gate:
            displacement_margin = float(
                getattr(self.config, "opinion_backfill_displacement_margin", 0.0)
            )
            max_repl_per_aspect = max(
                int(
                    getattr(
                        self.config,
                        "opinion_backfill_max_replacements_per_aspect",
                        1,
                    )
                ),
                1,
            )

            for b in range(batch_size):
                asp_ids = asp_topk_ids[b].tolist()
                base_opn_ids = opn_topk_ids[b][:base_count].tolist()
                delta_opn_ids = opn_topk_ids[b][base_count:K_o].tolist()
                if not asp_ids or not base_opn_ids or not delta_opn_ids:
                    continue

                all_opn_ids = base_opn_ids + delta_opn_ids

                pair_owner_ai = []
                pair_owner_oi = []
                pair_asp_reprs = []
                pair_opn_reprs = []
                pair_dist_ids = []
                pair_order_ids = []

                for ai, asp_span_idx in enumerate(asp_ids):
                    asp_span = span_indices[asp_span_idx]
                    for oi_local, opn_span_idx in enumerate(all_opn_ids):
                        opn_span = span_indices[opn_span_idx]
                        pair_owner_ai.append(ai)
                        pair_owner_oi.append(oi_local)
                        pair_asp_reprs.append(span_reprs[b, asp_span_idx])
                        pair_opn_reprs.append(span_reprs[b, opn_span_idx])
                        pair_dist_ids.append(
                            compute_distance_bucket(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )
                        pair_order_ids.append(
                            compute_order(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )

                if not pair_owner_ai:
                    continue

                asp_gate = torch.stack(pair_asp_reprs, dim=0).unsqueeze(0)
                opn_gate = torch.stack(pair_opn_reprs, dim=0).unsqueeze(0)
                dist_gate = torch.tensor(
                    pair_dist_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)
                order_gate = torch.tensor(
                    pair_order_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)

                with torch.no_grad():
                    gate_out = self.pair_module(
                        asp_reprs=asp_gate,
                        opn_reprs=opn_gate,
                        dist_ids=dist_gate,
                        order_ids=order_gate,
                    )
                    gate_scores = torch.sigmoid(gate_out["pair_scores"][0]).tolist()

                score_matrix = [
                    [0.0 for _ in range(len(all_opn_ids))]
                    for _ in range(K_a)
                ]
                for pid, score in enumerate(gate_scores):
                    ai = pair_owner_ai[pid]
                    oi_local = pair_owner_oi[pid]
                    score_matrix[ai][oi_local] = score

                base_len = len(base_opn_ids)
                delta_len = len(delta_opn_ids)
                for ai in range(K_a):
                    delta_scores = []
                    for delta_local in range(delta_len):
                        oi_local = base_len + delta_local
                        delta_scores.append((score_matrix[ai][oi_local], delta_local))
                    delta_scores.sort(key=lambda x: x[0], reverse=True)
                    if not delta_scores:
                        continue

                    asp_is_uncovered = bool(asp_uncovered_mask[b, ai].item())
                    if asp_is_uncovered:
                        for _, delta_local in delta_scores[:max_repl_per_aspect]:
                            oi_global = base_count + delta_local
                            asp_delta_displacement_keep_mask[b, ai, oi_global] = True
                        continue

                    base_scores = score_matrix[ai][:base_len]
                    base_rank = sorted(
                        range(base_len), key=lambda oi: base_scores[oi]
                    )
                    limit = min(max_repl_per_aspect, len(delta_scores), len(base_rank))
                    for ridx in range(limit):
                        delta_score, delta_local = delta_scores[ridx]
                        weakest_base_local = base_rank[ridx]
                        weakest_base_score = base_scores[weakest_base_local]
                        if delta_score - weakest_base_score < displacement_margin:
                            continue

                        oi_delta_global = base_count + delta_local
                        oi_base_global = weakest_base_local
                        asp_delta_displacement_keep_mask[b, ai, oi_delta_global] = True
                        asp_base_displacement_drop_mask[b, ai, oi_base_global] = True

        asp_delta_gain_keep_mask = torch.zeros(
            batch_size,
            K_a + 1,  # +1 for NULL slot appended later
            K_o + 1,  # +1 for NULL slot appended later
            dtype=torch.bool,
            device=asp_scores.device,
        )
        use_marginal_gain_gate = (
            mode == "inference" and
            bool(getattr(self.config, "opinion_backfill_use_marginal_gain_gate", False)) and
            int(getattr(self.config, "opinion_span_topk_delta", 0)) > 0 and
            K_o > base_count and
            K_a > 0 and
            base_count > 0
        )
        if use_marginal_gain_gate:
            gain_margin = float(getattr(self.config, "opinion_backfill_gain_margin", 0.0))
            max_delta_per_aspect = max(
                int(getattr(self.config, "opinion_backfill_max_delta_per_aspect", 1)),
                1,
            )

            for b in range(batch_size):
                asp_ids = asp_topk_ids[b].tolist()
                base_opn_ids = opn_topk_ids[b][:base_count].tolist()
                delta_opn_ids = opn_topk_ids[b][base_count:K_o].tolist()
                if not asp_ids or not base_opn_ids or not delta_opn_ids:
                    continue

                all_opn_ids = base_opn_ids + delta_opn_ids

                pair_owner_ai = []
                pair_owner_oi = []
                pair_asp_reprs = []
                pair_opn_reprs = []
                pair_dist_ids = []
                pair_order_ids = []

                for ai, asp_span_idx in enumerate(asp_ids):
                    asp_span = span_indices[asp_span_idx]
                    for oi_local, opn_span_idx in enumerate(all_opn_ids):
                        opn_span = span_indices[opn_span_idx]
                        pair_owner_ai.append(ai)
                        pair_owner_oi.append(oi_local)
                        pair_asp_reprs.append(span_reprs[b, asp_span_idx])
                        pair_opn_reprs.append(span_reprs[b, opn_span_idx])
                        pair_dist_ids.append(
                            compute_distance_bucket(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )
                        pair_order_ids.append(
                            compute_order(
                                asp_span[0], asp_span[1], opn_span[0], opn_span[1]
                            )
                        )

                if not pair_owner_ai:
                    continue

                asp_gate = torch.stack(pair_asp_reprs, dim=0).unsqueeze(0)
                opn_gate = torch.stack(pair_opn_reprs, dim=0).unsqueeze(0)
                dist_gate = torch.tensor(
                    pair_dist_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)
                order_gate = torch.tensor(
                    pair_order_ids, device=asp_scores.device, dtype=torch.long
                ).unsqueeze(0)

                with torch.no_grad():
                    gate_out = self.pair_module(
                        asp_reprs=asp_gate,
                        opn_reprs=opn_gate,
                        dist_ids=dist_gate,
                        order_ids=order_gate,
                    )
                    gate_scores = torch.sigmoid(gate_out["pair_scores"][0]).tolist()

                score_matrix = [
                    [0.0 for _ in range(len(all_opn_ids))]
                    for _ in range(K_a)
                ]
                for pid, score in enumerate(gate_scores):
                    ai = pair_owner_ai[pid]
                    oi_local = pair_owner_oi[pid]
                    score_matrix[ai][oi_local] = score

                base_len = len(base_opn_ids)
                base_best_scores = []
                for ai in range(K_a):
                    base_best_scores.append(max(score_matrix[ai][:base_len]))

                accepted = {}
                for delta_local in range(len(delta_opn_ids)):
                    oi_local = base_len + delta_local

                    best_ai = 0
                    best_score = score_matrix[0][oi_local]
                    for ai in range(1, K_a):
                        score = score_matrix[ai][oi_local]
                        if score > best_score:
                            best_ai = ai
                            best_score = score

                    gain = best_score - base_best_scores[best_ai]
                    if gain < gain_margin:
                        continue
                    accepted.setdefault(best_ai, []).append((gain, delta_local))

                for ai, cand_list in accepted.items():
                    cand_list.sort(key=lambda x: x[0], reverse=True)
                    for _, delta_local in cand_list[:max_delta_per_aspect]:
                        oi_global = base_count + delta_local
                        asp_delta_gain_keep_mask[b, ai, oi_global] = True

        asp_topk_scores = torch.gather(asp_scores, 1, asp_topk_ids)
        opn_topk_scores = torch.gather(opn_scores, 1, opn_topk_ids)

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

        # Collect span indices per batch
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
            "asp_topk_scores": asp_topk_scores,
            "opn_topk_scores": opn_topk_scores,
            "opn_is_delta_mask": opn_is_delta_mask,
            "asp_uncovered_mask": asp_uncovered_mask,
            "aspect_uncovered_gate_active": use_aspect_uncovered_gate,
            "asp_delta_displacement_keep_mask": asp_delta_displacement_keep_mask,
            "asp_base_displacement_drop_mask": asp_base_displacement_drop_mask,
            "aspect_displacement_gate_active": use_aspect_displacement_gate,
            "asp_delta_gain_keep_mask": asp_delta_gain_keep_mask,
            "marginal_gain_gate_active": use_marginal_gain_gate,
        }

    def _construct_pairs(self, pruned):
        """Build Cartesian product of aspect × opinion, excluding NULL×NULL."""
        asp_reprs = pruned["asp_reprs"]
        opn_reprs = pruned["opn_reprs"]
        batch_size = asp_reprs.size(0)
        n_asp = asp_reprs.size(1)
        n_opn = opn_reprs.size(1)
        device = asp_reprs.device

        asp_expanded = asp_reprs.unsqueeze(2).expand(-1, -1, n_opn, -1)
        opn_expanded = opn_reprs.unsqueeze(1).expand(-1, n_asp, -1, -1)

        asp_flat = asp_expanded.reshape(batch_size, n_asp * n_opn, -1)
        opn_flat = opn_expanded.reshape(batch_size, n_asp * n_opn, -1)

        # Mask out NULL×NULL
        mask = torch.ones(n_asp, n_opn, dtype=torch.bool, device=device)
        mask[-1, -1] = False
        mask_flat = mask.reshape(-1)

        num_valid = mask_flat.sum().item()
        asp_pairs = asp_flat[:, mask_flat, :]
        opn_pairs = opn_flat[:, mask_flat, :]

        # Build pair_map: which (asp_idx, opn_idx) each pair position maps to
        pair_map = []
        for ai in range(n_asp):
            for oi in range(n_opn):
                if mask[ai, oi]:
                    pair_map.append((ai, oi))

        # Compute distance and order IDs from span indices
        dist_ids = torch.zeros(batch_size, num_valid, dtype=torch.long, device=device)
        order_ids = torch.zeros(batch_size, num_valid, dtype=torch.long, device=device)

        for b in range(batch_size):
            asp_indices = pruned["asp_indices"][b]
            opn_indices = pruned["opn_indices"][b]
            for p, (ai, oi) in enumerate(pair_map):
                a_span = asp_indices[ai]
                o_span = opn_indices[oi]
                dist_ids[b, p] = compute_distance_bucket(
                    a_span[0], a_span[1], o_span[0], o_span[1]
                )
                order_ids[b, p] = compute_order(
                    a_span[0], a_span[1], o_span[0], o_span[1]
                )

        pair_inputs = {
            "asp_reprs": asp_pairs,
            "opn_reprs": opn_pairs,
            "dist_ids": dist_ids,
            "order_ids": order_ids,
        }

        pair_valid_mask = torch.ones(
            batch_size, num_valid, dtype=torch.bool, device=device
        )
        if bool(pruned.get("aspect_uncovered_gate_active", False)):
            opn_is_delta_mask = pruned.get("opn_is_delta_mask")
            asp_uncovered_mask = pruned.get("asp_uncovered_mask")
            for b in range(batch_size):
                for p, (ai, oi) in enumerate(pair_map):
                    is_delta = bool(opn_is_delta_mask[b, oi].item())
                    asp_uncovered = bool(asp_uncovered_mask[b, ai].item())
                    if is_delta and not asp_uncovered:
                        pair_valid_mask[b, p] = False

        if bool(pruned.get("marginal_gain_gate_active", False)):
            opn_is_delta_mask = pruned.get("opn_is_delta_mask")
            asp_delta_gain_keep_mask = pruned.get("asp_delta_gain_keep_mask")
            for b in range(batch_size):
                for p, (ai, oi) in enumerate(pair_map):
                    if not bool(opn_is_delta_mask[b, oi].item()):
                        continue
                    keep = bool(asp_delta_gain_keep_mask[b, ai, oi].item())
                    if not keep:
                        pair_valid_mask[b, p] = False

        if bool(pruned.get("aspect_displacement_gate_active", False)):
            opn_is_delta_mask = pruned.get("opn_is_delta_mask")
            asp_delta_displacement_keep_mask = pruned.get(
                "asp_delta_displacement_keep_mask"
            )
            asp_base_displacement_drop_mask = pruned.get(
                "asp_base_displacement_drop_mask"
            )
            for b in range(batch_size):
                for p, (ai, oi) in enumerate(pair_map):
                    is_delta = bool(opn_is_delta_mask[b, oi].item())
                    if is_delta:
                        keep = bool(
                            asp_delta_displacement_keep_mask[b, ai, oi].item()
                        )
                        if not keep:
                            pair_valid_mask[b, p] = False
                        continue

                    drop = bool(asp_base_displacement_drop_mask[b, ai, oi].item())
                    if drop:
                        pair_valid_mask[b, p] = False

        return pair_inputs, pair_map, pair_valid_mask

    @staticmethod
    def _word_span_to_subword(word_start, word_end, w2s):
        """Convert word-level span to subword-level span using alignment.

        Args:
            word_start: word-level start index (inclusive)
            word_end: word-level end index (inclusive)
            w2s: list of (sub_start, sub_end) per word

        Returns:
            (sub_start, sub_end) inclusive, or None if out of range.
        """
        if word_start < 0 or word_end < 0:
            return (-1, -1)
        if word_start >= len(w2s) or word_end >= len(w2s):
            return None
        sub_start = w2s[word_start][0]
        sub_end = w2s[word_end][1]
        return (sub_start, sub_end)

    @staticmethod
    def _subword_span_to_word(sub_start, sub_end, w2s):
        """Convert subword-level span back to word-level span.

        Args:
            sub_start: subword start index (inclusive)
            sub_end: subword end index (inclusive)
            w2s: list of (sub_start, sub_end) per word

        Returns:
            (word_start, word_end) inclusive, or (-1, -1) for NULL.
        """
        if sub_start < 0 or sub_end < 0:
            return (-1, -1)
        word_start = None
        word_end = None
        for w_idx, (ws, we) in enumerate(w2s):
            if ws <= sub_start <= we:
                word_start = w_idx
            if ws <= sub_end <= we:
                word_end = w_idx
        if word_start is not None and word_end is not None:
            return (word_start, word_end)
        return (-1, -1)

    @staticmethod
    def _span_overlap(span_a, span_b):
        """Whether two inclusive spans overlap."""
        if span_a == (-1, -1) or span_b == (-1, -1):
            return False
        return not (span_a[1] < span_b[0] or span_b[1] < span_a[0])

    def _compute_pair_rank_loss(
        self,
        pair_probs,
        pair_labels,
        null_related_mask,
        semantic_nearmiss_rank_mask,
        cat_confused_mask,
        pair_valid_mask=None,
    ):
        """Margin loss over the hardest Stage-1 pair negatives."""
        cfg = self.config
        margin = float(getattr(cfg, "pair_rank_margin", 0.1))
        semantic_weight = float(getattr(cfg, "pair_rank_semantic_weight", 1.0))
        hard_neg_cap = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)

        hard_neg_mask = (
            (pair_labels == 0.0) &
            (
                null_related_mask.bool() |
                semantic_nearmiss_rank_mask.bool() |
                cat_confused_mask.bool()
            )
        )

        total_loss = pair_probs.sum() * 0.0
        total_pairs = 0

        for b in range(pair_probs.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            pos_scores = pair_probs[b][(pair_labels[b] == 1.0) & valid_mask]
            neg_mask_b = hard_neg_mask[b] & valid_mask
            hard_neg_scores = pair_probs[b][neg_mask_b]
            hard_neg_semantic_mask = (
                semantic_nearmiss_rank_mask[b].bool() |
                cat_confused_mask[b].bool()
            )[neg_mask_b]

            if pos_scores.numel() == 0 or hard_neg_scores.numel() == 0:
                continue

            hard_neg_weights = torch.ones_like(hard_neg_scores)
            if semantic_weight != 1.0:
                hard_neg_weights[hard_neg_semantic_mask] = semantic_weight

            if hard_neg_scores.numel() > hard_neg_cap:
                topk_ids = torch.topk(
                    hard_neg_scores,
                    hard_neg_cap,
                    largest=True,
                ).indices
                hard_neg_scores = hard_neg_scores[topk_ids]
                hard_neg_weights = hard_neg_weights[topk_ids]

            diff = pos_scores.unsqueeze(1) - hard_neg_scores.unsqueeze(0)
            rank_loss = torch.clamp(margin - diff, min=0.0)
            rank_loss = rank_loss * hard_neg_weights.unsqueeze(0)
            total_loss = total_loss + rank_loss.sum()
            total_pairs += rank_loss.numel()

        if total_pairs > 0:
            total_loss = total_loss / total_pairs

        return total_loss

    def _compute_pacr_loss(
        self,
        pair_probs,
        pair_labels,
        pair_map,
        pair_valid_mask=None,
    ):
        """Per-Aspect Competitive Ranking (PACR) loss.

        For each positive pair (a, o_pos), find hardest non-gold opinion
        under the same aspect a and enforce:
            margin + s(a, o_neg) - s(a, o_pos) <= 0.
        """
        cfg = self.config
        margin = float(getattr(cfg, "pacr_margin", 0.1))
        same_aspect_only = bool(getattr(cfg, "pacr_same_aspect_only", True))
        hardneg_topk = max(int(getattr(cfg, "pacr_hardneg_topk", 1)), 1)

        total_loss = pair_probs.sum() * 0.0
        active_pairs = 0
        violation_count = 0
        pos_score_sum = 0.0
        neg_score_sum = 0.0

        aspect_to_pair_ids = {}
        for pid, (ai, _oi) in enumerate(pair_map):
            if ai not in aspect_to_pair_ids:
                aspect_to_pair_ids[ai] = []
            aspect_to_pair_ids[ai].append(pid)

        for b in range(pair_probs.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            pos_ids = torch.where((pair_labels[b] == 1.0) & valid_mask)[0]
            if pos_ids.numel() == 0:
                continue

            for pos_pid in pos_ids.tolist():
                pos_ai, _ = pair_map[pos_pid]
                if same_aspect_only:
                    candidate_neg_ids = aspect_to_pair_ids.get(pos_ai, [])
                else:
                    candidate_neg_ids = list(range(len(pair_map)))
                if not candidate_neg_ids:
                    continue

                neg_ids = [
                    nid for nid in candidate_neg_ids
                    if valid_mask[nid].item() and pair_labels[b, nid].item() == 0.0
                ]
                if not neg_ids:
                    continue

                neg_scores = pair_probs[b, neg_ids]
                if neg_scores.numel() == 0:
                    continue

                if neg_scores.numel() > hardneg_topk:
                    topk_scores = torch.topk(
                        neg_scores, hardneg_topk, largest=True
                    ).values
                    hardneg_score = topk_scores.mean()
                else:
                    hardneg_score = neg_scores.max()

                pos_score = pair_probs[b, pos_pid]
                loss_item = torch.clamp(margin + hardneg_score - pos_score, min=0.0)
                total_loss = total_loss + loss_item
                active_pairs += 1
                pos_score_sum += float(pos_score.detach().item())
                neg_score_sum += float(hardneg_score.detach().item())
                if float(loss_item.detach().item()) > 0:
                    violation_count += 1

        if active_pairs > 0:
            total_loss = total_loss / active_pairs

        pacr_stats = {
            "active_pairs": int(active_pairs),
            "mean_pos_score": (
                pos_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "mean_hardneg_score": (
                neg_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "violation_rate": (
                violation_count / active_pairs if active_pairs > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_pairs > 0 else 0.0,
        }
        return total_loss, pacr_stats

    def _collect_agml_aspect_groups(
        self,
        pair_labels,
        pair_map,
        pair_valid_mask=None,
        same_aspect_only=True,
    ):
        """Collect same-aspect candidate/gold/non-gold ids for AGML-style losses."""
        batch_size, num_pairs = pair_labels.shape
        aspect_to_pair_ids = {}
        for pid, (ai, _oi) in enumerate(pair_map):
            if ai not in aspect_to_pair_ids:
                aspect_to_pair_ids[ai] = []
            aspect_to_pair_ids[ai].append(pid)

        groups = []
        for b in range(batch_size):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones(
                    num_pairs, dtype=torch.bool, device=pair_labels.device
                )
            )
            pos_ids = torch.where((pair_labels[b] == 1.0) & valid_mask)[0]
            if pos_ids.numel() == 0:
                continue

            gold_by_aspect = {}
            for pid in pos_ids.tolist():
                ai, _ = pair_map[pid]
                if ai not in gold_by_aspect:
                    gold_by_aspect[ai] = []
                gold_by_aspect[ai].append(pid)

            if same_aspect_only:
                global_candidate_ids = None
            else:
                global_candidate_ids = torch.where(valid_mask)[0].tolist()
                if not global_candidate_ids:
                    continue

            for ai, gold_ids in gold_by_aspect.items():
                if same_aspect_only:
                    candidate_ids = [
                        pid for pid in aspect_to_pair_ids.get(ai, [])
                        if valid_mask[pid].item()
                    ]
                else:
                    candidate_ids = global_candidate_ids

                if not candidate_ids:
                    continue

                valid_gold_ids = [pid for pid in gold_ids if valid_mask[pid].item()]
                if not valid_gold_ids:
                    continue

                gold_id_set = set(valid_gold_ids)
                neg_ids = [pid for pid in candidate_ids if pid not in gold_id_set]
                groups.append(
                    {
                        "batch_index": b,
                        "aspect_index": ai,
                        "candidate_ids": candidate_ids,
                        "gold_ids": valid_gold_ids,
                        "neg_ids": neg_ids,
                    }
                )
        return groups

    def _compute_agml_loss(
        self,
        pair_scores,
        groups,
    ):
        """Aspect-Conditioned Gold-Mass Loss (AGML)."""
        tau = max(float(getattr(self.config, "agml_tau", 1.0)), 1e-6)
        total_loss = pair_scores.sum() * 0.0
        active_aspects = 0
        gold_mass_sum = 0.0

        for g in groups:
            b = g["batch_index"]
            cand_scores = pair_scores[b, g["candidate_ids"]] / tau
            gold_scores = pair_scores[b, g["gold_ids"]] / tau
            log_denom = torch.logsumexp(cand_scores, dim=0)
            log_num = torch.logsumexp(gold_scores, dim=0)
            loss_item = -(log_num - log_denom)
            total_loss = total_loss + loss_item
            active_aspects += 1
            gold_mass_sum += float(torch.exp((log_num - log_denom).detach()).item())

        if active_aspects > 0:
            total_loss = total_loss / active_aspects

        agml_stats = {
            "active_aspects": int(active_aspects),
            "mean_gold_mass": (
                gold_mass_sum / active_aspects if active_aspects > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_aspects > 0 else 0.0,
        }
        return total_loss, agml_stats

    def _compute_agml_comp_loss(
        self,
        pair_scores,
        groups,
    ):
        """AGML-CS: suppress strongest same-aspect non-gold competitors."""
        cfg = self.config
        margin = float(getattr(cfg, "agml_comp_margin", 0.05))
        topk = max(int(getattr(cfg, "agml_comp_topk", 3)), 1)

        total_loss = pair_scores.sum() * 0.0
        active_aspects = 0
        violation_count = 0
        pos_group_sum = 0.0
        neg_group_sum = 0.0

        for g in groups:
            neg_ids = g["neg_ids"]
            if not neg_ids:
                continue

            b = g["batch_index"]
            pos_scores = pair_scores[b, g["gold_ids"]]
            s_pos = torch.logsumexp(pos_scores, dim=0)

            neg_scores = pair_scores[b, neg_ids]
            k = min(topk, neg_scores.numel())
            topk_neg = torch.topk(neg_scores, k, largest=True).values
            s_neg = torch.logsumexp(topk_neg, dim=0)

            loss_item = torch.clamp(margin + s_neg - s_pos, min=0.0)
            total_loss = total_loss + loss_item
            active_aspects += 1
            pos_group_sum += float(s_pos.detach().item())
            neg_group_sum += float(s_neg.detach().item())
            if float(loss_item.detach().item()) > 0.0:
                violation_count += 1

        if active_aspects > 0:
            total_loss = total_loss / active_aspects

        comp_stats = {
            "active_aspects": int(active_aspects),
            "mean_pos_group": (
                pos_group_sum / active_aspects if active_aspects > 0 else 0.0
            ),
            "mean_neg_group": (
                neg_group_sum / active_aspects if active_aspects > 0 else 0.0
            ),
            "violation_rate": (
                violation_count / active_aspects if active_aspects > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_aspects > 0 else 0.0,
        }
        return total_loss, comp_stats

    def _compute_agml_br_loss(
        self,
        pair_scores,
        pair_labels,
        pair_valid_mask=None,
    ):
        """AGML-BR: global sentence-level boundary ranking for gold pairs.

        For each sentence, let b be the score of the top-N boundary pair
        (N = stage1_pair_top_n). For each gold pair ranked outside top-N:
            relu(margin + stopgrad(b) - s_gold).
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        margin = float(getattr(cfg, "agml_br_margin", 0.05))

        total_loss = pair_scores.sum() * 0.0
        active_gold_pairs = 0
        violation_count = 0
        boundary_sum = 0.0
        gold_score_sum = 0.0

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if len(valid_ids) < top_n:
                continue

            # Sentence-level valid pair ranking and top-N boundary score.
            sorted_ids = sorted(
                valid_ids,
                key=lambda pid: float(pair_scores[b, pid].detach().item()),
                reverse=True,
            )
            rank_map = {pid: ridx + 1 for ridx, pid in enumerate(sorted_ids)}
            boundary_pid = sorted_ids[top_n - 1]
            boundary_score = pair_scores[b, boundary_pid].detach()

            gold_ids = torch.where((pair_labels[b] == 1.0) & valid_mask)[0].tolist()
            if not gold_ids:
                continue

            for pid in gold_ids:
                rank = rank_map.get(pid)
                if rank is None or rank <= top_n:
                    continue

                gold_score = pair_scores[b, pid]
                loss_item = torch.clamp(margin + boundary_score - gold_score, min=0.0)
                total_loss = total_loss + loss_item
                active_gold_pairs += 1
                boundary_sum += float(boundary_score.item())
                gold_score_sum += float(gold_score.detach().item())
                if float(loss_item.detach().item()) > 0.0:
                    violation_count += 1

        if active_gold_pairs > 0:
            total_loss = total_loss / active_gold_pairs

        br_stats = {
            "active_gold_pairs": int(active_gold_pairs),
            "mean_boundary": (
                boundary_sum / active_gold_pairs if active_gold_pairs > 0 else 0.0
            ),
            "mean_gold_score": (
                gold_score_sum / active_gold_pairs if active_gold_pairs > 0 else 0.0
            ),
            "violation_rate": (
                violation_count / active_gold_pairs if active_gold_pairs > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_gold_pairs > 0 else 0.0,
        }
        return total_loss, br_stats

    def _compute_cbr_v1_loss(
        self,
        pair_scores,
        pair_labels,
        pair_valid_mask=None,
    ):
        """CBR-v1: cutoff-aware boundary ranking loss on top-N retention boundary.

        Uses the same candidate universe and sorting path as topn_only retention:
        - valid candidate ids from pair_valid_mask
        - final pair scores from current pair_scores tensor (after sigmoid)
        - descending sort over valid candidates
        """
        cfg = self.config
        K = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        b = max(int(getattr(cfg, "cbr_v1_buffer", 3)), 1)
        margin = float(getattr(cfg, "cbr_v1_margin", 0.03))
        detach_cutoff = bool(getattr(cfg, "cbr_v1_detach_cutoff", True))

        pair_probs = torch.sigmoid(pair_scores)
        total_loss = pair_scores.sum() * 0.0
        active_samples = 0
        active_positive_pairs = 0
        total_positive_pairs = 0
        cutoff_gap_sum = 0.0

        for b_idx in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b_idx].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b_idx], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0]
            if valid_ids.numel() == 0:
                continue

            pos_ids = torch.where((pair_labels[b_idx] == 1.0) & valid_mask)[0]
            neg_ids = torch.where((pair_labels[b_idx] == 0.0) & valid_mask)[0]
            if pos_ids.numel() == 0 or neg_ids.numel() == 0:
                continue

            total_positive_pairs += int(pos_ids.numel())
            g_x = int(pos_ids.numel())
            n_x = int(neg_ids.numel())
            k_eff = max(1, min(n_x, K - min(g_x - 1, K - 1)))

            neg_scores = pair_probs[b_idx, neg_ids]
            tau_x = torch.topk(neg_scores, k=k_eff, largest=True).values[-1]
            tau_for_loss = tau_x.detach() if detach_cutoff else tau_x

            # Same ranking path as retention: sort valid candidates by final pair score.
            valid_scores = pair_probs[b_idx, valid_ids]
            sorted_valid = valid_ids[torch.argsort(valid_scores, descending=True)]
            rank_map = {int(pid): ridx + 1 for ridx, pid in enumerate(sorted_valid.tolist())}

            sample_terms = []
            for pid in pos_ids.tolist():
                s_p = pair_probs[b_idx, pid]
                rank_p = rank_map.get(int(pid), K + 1)
                in_boundary_band = rank_p > (K - b)
                below_cutoff_band = bool((s_p < (tau_x + margin)).item())
                if not (in_boundary_band or below_cutoff_band):
                    continue

                term = torch.clamp(margin + tau_for_loss - s_p, min=0.0)
                sample_terms.append(term)
                active_positive_pairs += 1
                cutoff_gap_sum += float((tau_for_loss - s_p.detach()).item())

            if sample_terms:
                total_loss = total_loss + torch.stack(sample_terms).mean()
                active_samples += 1

        if active_samples > 0:
            total_loss = total_loss / active_samples

        cbr_stats = {
            "num_samples_with_active_boundary_loss": int(active_samples),
            "boundary_active_positive_ratio": (
                active_positive_pairs / total_positive_pairs
                if total_positive_pairs > 0 else 0.0
            ),
            "avg_cutoff_gap": (
                cutoff_gap_sum / active_positive_pairs
                if active_positive_pairs > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_samples > 0 else 0.0,
        }
        return total_loss, cbr_stats

    def _compute_ma_aux_loss(
        self,
        pair_reprs,
        pair_scores,
        pair_labels,
        ma_cat_targets,
        ma_sent_targets=None,
        pair_valid_mask=None,
    ):
        """Materialization-aware dual auxiliary on pair representations.

        Supervise category/sentiment multi-hot labels on:
        1) gold pairs
        2) high-score non-gold competitors (retained-topn by default)
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        neg_source = str(getattr(cfg, "ma_aux_neg_source", "retained")).lower()
        hardneg_topk = max(int(getattr(cfg, "ma_aux_hardneg_topk", top_n)), 1)

        pair_probs = torch.sigmoid(pair_scores).detach()
        cat_loss_sum = pair_scores.sum() * 0.0
        sent_loss_sum = pair_scores.sum() * 0.0
        active_pairs = 0
        pos_pairs = 0
        neg_pairs = 0

        has_sent_head = (
            self.ma_aux_sent_head is not None and
            ma_sent_targets is not None and
            self.config.task_type == "asqp"
        )

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if not valid_ids:
                continue

            pos_ids = torch.where((pair_labels[b] == 1.0) & valid_mask)[0].tolist()

            if neg_source == "retained":
                ranked_valid = sorted(
                    valid_ids,
                    key=lambda pid: float(pair_probs[b, pid].item()),
                    reverse=True,
                )
                retained_ids = ranked_valid[:top_n]
                neg_ids = [pid for pid in retained_ids if pair_labels[b, pid].item() == 0.0]
            else:
                non_gold_ids = [pid for pid in valid_ids if pair_labels[b, pid].item() == 0.0]
                non_gold_ids = sorted(
                    non_gold_ids,
                    key=lambda pid: float(pair_probs[b, pid].item()),
                    reverse=True,
                )[:hardneg_topk]
                neg_ids = non_gold_ids

            selected_ids = []
            seen = set()
            for pid in pos_ids + neg_ids:
                if pid in seen:
                    continue
                selected_ids.append(pid)
                seen.add(pid)
            if not selected_ids:
                continue

            reps = pair_reprs[b, selected_ids]
            cat_logits = self.ma_aux_cat_head(reps)
            cat_targets = ma_cat_targets[b, selected_ids]
            cat_per_pair = F.binary_cross_entropy_with_logits(
                cat_logits, cat_targets, reduction="none"
            ).mean(dim=-1)
            cat_loss_sum = cat_loss_sum + cat_per_pair.sum()

            if has_sent_head:
                sent_logits = self.ma_aux_sent_head(reps)
                sent_targets = ma_sent_targets[b, selected_ids]
                sent_per_pair = F.binary_cross_entropy_with_logits(
                    sent_logits, sent_targets, reduction="none"
                ).mean(dim=-1)
                sent_loss_sum = sent_loss_sum + sent_per_pair.sum()

            active_pairs += len(selected_ids)
            pos_pairs += len(pos_ids)
            neg_pairs += len([pid for pid in selected_ids if pair_labels[b, pid].item() == 0.0])

        if active_pairs > 0:
            cat_loss_mean = cat_loss_sum / active_pairs
            sent_loss_mean = sent_loss_sum / active_pairs if has_sent_head else pair_scores.sum() * 0.0
            loss_ma = cat_loss_mean + sent_loss_mean
        else:
            cat_loss_mean = pair_scores.sum() * 0.0
            sent_loss_mean = pair_scores.sum() * 0.0
            loss_ma = pair_scores.sum() * 0.0

        ma_stats = {
            "active_pairs": int(active_pairs),
            "pos_pairs": int(pos_pairs),
            "neg_pairs": int(neg_pairs),
            "cat_loss_mean": float(cat_loss_mean.detach().item()) if active_pairs > 0 else 0.0,
            "sent_loss_mean": float(sent_loss_mean.detach().item()) if active_pairs > 0 else 0.0,
            "loss_mean": float(loss_ma.detach().item()) if active_pairs > 0 else 0.0,
        }
        return loss_ma, ma_stats

    def _compute_mbl_loss(
        self,
        cat_logits,
        aff_logits,
        pair_scores,
        pair_labels,
        ma_cat_targets,
        ma_sent_targets=None,
        pair_valid_mask=None,
        enable_cat=True,
        enable_sent=True,
    ):
        """Decode-aligned materialization boundary losses on retained gold pairs.

        Category boundary: top-c boundary within this pair's category logits.
        Sentiment boundary: strongest wrong sentiment logit within this pair.
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        top_c = max(int(getattr(cfg, "top_c_categories", 3)), 1)
        cat_margin = float(getattr(cfg, "cat_mbl_margin", 0.05))
        sent_margin = float(getattr(cfg, "sent_mbl_margin", 0.05))

        cat_loss_sum = pair_scores.sum() * 0.0
        sent_loss_sum = pair_scores.sum() * 0.0
        cat_active_pairs = 0
        sent_active_pairs = 0
        cat_violation_pairs = 0
        sent_violation_pairs = 0
        active_pair_keys = set()

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if not valid_ids:
                continue

            ranked_valid = sorted(
                valid_ids,
                key=lambda pid: float(pair_scores[b, pid].detach().item()),
                reverse=True,
            )
            retained_ids = ranked_valid[:top_n]
            gold_retained_ids = [
                pid for pid in retained_ids if pair_labels[b, pid].item() == 1.0
            ]
            if not gold_retained_ids:
                continue

            for pid in gold_retained_ids:
                did_activate = False

                gold_cat_mask = ma_cat_targets[b, pid] > 0.5
                if enable_cat and gold_cat_mask.any():
                    pair_cat_logits = cat_logits[b, pid]
                    k = min(top_c, pair_cat_logits.numel())
                    if k > 0:
                        cat_boundary = torch.topk(
                            pair_cat_logits, k, largest=True
                        ).values[-1].detach()
                        gold_cat_logits = pair_cat_logits[gold_cat_mask]
                        cat_loss_vec = torch.clamp(
                            cat_margin + cat_boundary - gold_cat_logits,
                            min=0.0,
                        )
                        cat_loss_item = cat_loss_vec.mean()
                        cat_loss_sum = cat_loss_sum + cat_loss_item
                        cat_active_pairs += 1
                        did_activate = True
                        if float(cat_loss_item.detach().item()) > 0.0:
                            cat_violation_pairs += 1

                use_sent = (
                    enable_sent and
                    aff_logits is not None and
                    ma_sent_targets is not None and
                    self.config.task_type == "asqp"
                )
                if use_sent:
                    gold_sent_mask = ma_sent_targets[b, pid] > 0.5
                    wrong_sent_mask = ~gold_sent_mask
                    if gold_sent_mask.any() and wrong_sent_mask.any():
                        pair_sent_logits = aff_logits[b, pid]
                        sent_boundary = pair_sent_logits[wrong_sent_mask].max().detach()
                        gold_sent_logits = pair_sent_logits[gold_sent_mask]
                        sent_loss_vec = torch.clamp(
                            sent_margin + sent_boundary - gold_sent_logits,
                            min=0.0,
                        )
                        sent_loss_item = sent_loss_vec.mean()
                        sent_loss_sum = sent_loss_sum + sent_loss_item
                        sent_active_pairs += 1
                        did_activate = True
                        if float(sent_loss_item.detach().item()) > 0.0:
                            sent_violation_pairs += 1

                if did_activate:
                    active_pair_keys.add((b, pid))

        cat_loss_mean = (
            cat_loss_sum / cat_active_pairs
            if cat_active_pairs > 0
            else pair_scores.sum() * 0.0
        )
        sent_loss_mean = (
            sent_loss_sum / sent_active_pairs
            if sent_active_pairs > 0
            else pair_scores.sum() * 0.0
        )
        loss_mbl = cat_loss_mean + sent_loss_mean

        stats = {
            "active_pairs": int(len(active_pair_keys)),
            "cat_active_pairs": int(cat_active_pairs),
            "sent_active_pairs": int(sent_active_pairs),
            "cat_violation_rate": (
                cat_violation_pairs / cat_active_pairs if cat_active_pairs > 0 else 0.0
            ),
            "sent_violation_rate": (
                sent_violation_pairs / sent_active_pairs if sent_active_pairs > 0 else 0.0
            ),
            "cat_loss_mean": (
                float(cat_loss_mean.detach().item()) if cat_active_pairs > 0 else 0.0
            ),
            "sent_loss_mean": (
                float(sent_loss_mean.detach().item()) if sent_active_pairs > 0 else 0.0
            ),
            "loss_mean": (
                float(loss_mbl.detach().item()) if len(active_pair_keys) > 0 else 0.0
            ),
        }
        return cat_loss_mean, sent_loss_mean, stats

    def _compute_romr_v1_loss(
        self,
        cat_logits,
        aff_logits,
        pair_scores,
        pair_labels,
        pair_map,
        ma_cat_targets,
        ma_sent_targets=None,
        pair_valid_mask=None,
    ):
        """Retained-only same-aspect materialization auxiliary (ROMR-v1).

        Build retained set from the exact top-N retention path (topn_only, valid mask,
        sigmoid(pair_scores) ranking), then within retained pairs form same-aspect groups.
        For each group with retained gold and retained non-gold pairs, rank gold
        materialization confidence above hardest retained non-gold confidence.
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        margin = float(getattr(cfg, "romr_v1_margin", 0.05))
        detach_selection = bool(getattr(cfg, "romr_v1_detach_selection", True))

        rank_scores = torch.sigmoid(pair_scores.detach() if detach_selection else pair_scores)
        total_loss = pair_scores.sum() * 0.0
        active_aspects = 0
        active_pairs = 0
        violation_pairs = 0
        pos_score_sum = 0.0
        neg_score_sum = 0.0

        has_aff = (
            aff_logits is not None and
            ma_sent_targets is not None and
            self.config.task_type == "asqp"
        )

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if not valid_ids:
                continue

            ranked_valid = sorted(
                valid_ids,
                key=lambda pid: float(rank_scores[b, pid].item()),
                reverse=True,
            )
            retained_ids = ranked_valid[:top_n]
            if not retained_ids:
                continue

            grouped = {}
            for pid in retained_ids:
                asp_idx, _ = pair_map[pid]
                grouped.setdefault(int(asp_idx), []).append(int(pid))

            for _, group_ids in grouped.items():
                pos_ids = [pid for pid in group_ids if pair_labels[b, pid].item() == 1.0]
                neg_ids = [pid for pid in group_ids if pair_labels[b, pid].item() == 0.0]
                if not pos_ids or not neg_ids:
                    continue

                neg_scores = []
                for pid in neg_ids:
                    cat_conf = cat_logits[b, pid].max()
                    if has_aff:
                        aff_conf = aff_logits[b, pid].max()
                        neg_scores.append(cat_conf + aff_conf)
                    else:
                        neg_scores.append(cat_conf)
                if not neg_scores:
                    continue
                hard_neg = torch.stack(neg_scores).max()

                terms = []
                for pid in pos_ids:
                    gold_cat_mask = ma_cat_targets[b, pid] > 0.5
                    if not gold_cat_mask.any():
                        continue
                    pos_cat = cat_logits[b, pid][gold_cat_mask].mean()

                    if has_aff:
                        gold_sent_mask = ma_sent_targets[b, pid] > 0.5
                        if gold_sent_mask.any():
                            pos_aff = aff_logits[b, pid][gold_sent_mask].mean()
                        else:
                            pos_aff = aff_logits[b, pid].max()
                        pos_score = pos_cat + pos_aff
                    else:
                        pos_score = pos_cat

                    term = torch.clamp(margin + hard_neg - pos_score, min=0.0)
                    terms.append(term)
                    active_pairs += 1
                    pos_score_sum += float(pos_score.detach().item())
                    neg_score_sum += float(hard_neg.detach().item())
                    if float(term.detach().item()) > 0.0:
                        violation_pairs += 1

                if terms:
                    total_loss = total_loss + torch.stack(terms).mean()
                    active_aspects += 1

        if active_aspects > 0:
            total_loss = total_loss / active_aspects

        stats = {
            "active_aspects": int(active_aspects),
            "active_pairs": int(active_pairs),
            "mean_pos_score": (
                pos_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "mean_hardneg_score": (
                neg_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "violation_rate": (
                violation_pairs / active_pairs if active_pairs > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_aspects > 0 else 0.0,
        }
        return total_loss, stats

    def _compute_homr_v1_loss(
        self,
        pair_reprs,
        pair_scores,
        pair_labels,
        pair_map,
        ma_cat_targets,
        ma_sent_targets=None,
        pair_valid_mask=None,
    ):
        """Head-Only Materialization Refinement (HOMR-v1).

        Uses retained top-N pairs from the real retention path, but computes
        materialization ranking on detached pair representations so this branch
        updates head parameters only and does not backprop into pair-ranking trunk.
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        margin = float(getattr(cfg, "homr_v1_margin", 0.05))
        detach_selection = bool(getattr(cfg, "homr_v1_detach_selection", True))

        rank_scores = torch.sigmoid(pair_scores.detach() if detach_selection else pair_scores)

        # Head-only path: detach shared pair representations so gradients from HOMR
        # do not flow back into pair representation / pair scoring trunk.
        homr_reprs = pair_reprs.detach()
        cat_logits_h = self.pair_module.cat_head(homr_reprs)
        aff_logits_h = (
            self.pair_module.aff_head(homr_reprs)
            if self.config.task_type == "asqp"
            else None
        )

        total_loss = pair_scores.sum() * 0.0
        active_aspects = 0
        active_pairs = 0
        violation_pairs = 0
        pos_score_sum = 0.0
        neg_score_sum = 0.0
        top_c = max(int(getattr(cfg, "top_c_categories", 3)), 1)

        has_aff = (
            aff_logits_h is not None and
            ma_sent_targets is not None and
            self.config.task_type == "asqp"
        )

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if not valid_ids:
                continue

            ranked_valid = sorted(
                valid_ids,
                key=lambda pid: float(rank_scores[b, pid].item()),
                reverse=True,
            )
            retained_ids = ranked_valid[:top_n]
            if not retained_ids:
                continue

            grouped = {}
            for pid in retained_ids:
                asp_idx, _ = pair_map[pid]
                grouped.setdefault(int(asp_idx), []).append(int(pid))

            for _, group_ids in grouped.items():
                pos_ids = [pid for pid in group_ids if pair_labels[b, pid].item() == 1.0]
                neg_ids = [pid for pid in group_ids if pair_labels[b, pid].item() == 0.0]
                if not pos_ids or not neg_ids:
                    continue

                neg_scores = []
                for pid in neg_ids:
                    cat_vals = torch.topk(
                        cat_logits_h[b, pid],
                        k=min(top_c, cat_logits_h.size(-1)),
                        largest=True,
                    ).values
                    neg_cat = cat_vals.mean()
                    if has_aff:
                        neg_aff = aff_logits_h[b, pid].max()
                        neg_scores.append(neg_cat + neg_aff)
                    else:
                        neg_scores.append(neg_cat)
                if not neg_scores:
                    continue
                hard_neg = torch.stack(neg_scores).max()

                terms = []
                for pid in pos_ids:
                    gold_cat_mask = ma_cat_targets[b, pid] > 0.5
                    if not gold_cat_mask.any():
                        continue
                    pos_cat = cat_logits_h[b, pid][gold_cat_mask].mean()
                    if has_aff:
                        gold_sent_mask = ma_sent_targets[b, pid] > 0.5
                        if gold_sent_mask.any():
                            pos_aff = aff_logits_h[b, pid][gold_sent_mask].mean()
                        else:
                            pos_aff = aff_logits_h[b, pid].max()
                        pos_score = pos_cat + pos_aff
                    else:
                        pos_score = pos_cat

                    term = torch.clamp(margin + hard_neg - pos_score, min=0.0)
                    terms.append(term)
                    active_pairs += 1
                    pos_score_sum += float(pos_score.detach().item())
                    neg_score_sum += float(hard_neg.detach().item())
                    if float(term.detach().item()) > 0.0:
                        violation_pairs += 1

                if terms:
                    total_loss = total_loss + torch.stack(terms).mean()
                    active_aspects += 1

        if active_aspects > 0:
            total_loss = total_loss / active_aspects

        stats = {
            "active_aspects": int(active_aspects),
            "active_pairs": int(active_pairs),
            "mean_pos_score": (
                pos_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "mean_hardneg_score": (
                neg_score_sum / active_pairs if active_pairs > 0 else 0.0
            ),
            "violation_rate": (
                violation_pairs / active_pairs if active_pairs > 0 else 0.0
            ),
            "loss_mean": float(total_loss.detach().item()) if active_aspects > 0 else 0.0,
        }
        return total_loss, stats

    def _compute_rph_v1_loss(
        self,
        pair_reprs,
        pair_scores,
        pair_labels,
        pair_valid_mask=None,
    ):
        """Retained Probe Head (RPH-v1) on detached retained pair representations.

        - Retained set is built from the real retention path:
          pair_valid_mask + sigmoid(pair_scores) ranking + top_n.
        - Probe reads pair_reprs.detach(), so gradients do not flow back into
          pair ranking trunk.
        """
        cfg = self.config
        top_n = max(int(getattr(cfg, "stage1_pair_top_n", 20)), 1)
        detach_selection = bool(getattr(cfg, "rph_v1_detach_selection", True))

        rank_scores = torch.sigmoid(
            pair_scores.detach() if detach_selection else pair_scores
        )
        probe_logits = self.pair_module.rph_probe_head(pair_reprs.detach()).squeeze(-1)

        total_loss = pair_scores.sum() * 0.0
        active_samples = 0
        active_pairs = 0
        pos_pairs = 0
        neg_pairs = 0
        pos_prob_sum = 0.0
        neg_prob_sum = 0.0

        for b in range(pair_scores.size(0)):
            valid_mask = (
                pair_valid_mask[b].bool()
                if pair_valid_mask is not None
                else torch.ones_like(pair_labels[b], dtype=torch.bool)
            )
            valid_ids = torch.where(valid_mask)[0].tolist()
            if not valid_ids:
                continue

            ranked_valid = sorted(
                valid_ids,
                key=lambda pid: float(rank_scores[b, pid].item()),
                reverse=True,
            )
            retained_ids = ranked_valid[:top_n]
            if not retained_ids:
                continue

            logits_b = probe_logits[b, retained_ids]
            targets_b = pair_labels[b, retained_ids]
            loss_b = F.binary_cross_entropy_with_logits(logits_b, targets_b)
            total_loss = total_loss + loss_b
            active_samples += 1
            active_pairs += len(retained_ids)

            probs_b = torch.sigmoid(logits_b.detach())
            pos_mask = targets_b > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_pairs += int(pos_mask.sum().item())
                pos_prob_sum += float(probs_b[pos_mask].sum().item())
            if neg_mask.any():
                neg_pairs += int(neg_mask.sum().item())
                neg_prob_sum += float(probs_b[neg_mask].sum().item())

        if active_samples > 0:
            total_loss = total_loss / active_samples

        stats = {
            "active_pairs": int(active_pairs),
            "pos_pairs": int(pos_pairs),
            "neg_pairs": int(neg_pairs),
            "mean_pos_prob": (pos_prob_sum / pos_pairs if pos_pairs > 0 else 0.0),
            "mean_neg_prob": (neg_prob_sum / neg_pairs if neg_pairs > 0 else 0.0),
            "loss_mean": float(total_loss.detach().item()) if active_samples > 0 else 0.0,
        }
        return total_loss, stats

    def _compute_stage1_losses(self, proposal_out, pair_out, pruned, pair_map,
                                gold_quads, cat_to_id, word_to_subword,
                                pair_valid_mask=None):
        """Compute L_span + L_pair + L_cat + L_aff with full label assignment."""
        device = proposal_out["asp_scores"].device
        batch_size = proposal_out["asp_scores"].size(0)
        span_indices = proposal_out["span_indices"]

        # Build lookup set for fast span matching
        span_index_map = {span: idx for idx, span in enumerate(span_indices)}

        # =====================================================================
        # L_span: Assign binary labels to all enumerated spans
        # =====================================================================
        num_spans = len(span_indices)
        asp_labels = torch.zeros(batch_size, num_spans, device=device)
        opn_labels = torch.zeros(batch_size, num_spans, device=device)

        for b in range(batch_size):
            if gold_quads is None or gold_quads[b] is None:
                continue
            w2s = word_to_subword[b] if word_to_subword else None
            for q in gold_quads[b]:
                if not q.aspect.is_null and w2s is not None:
                    # Convert word-level span to subword-level
                    target = self._word_span_to_subword(
                        q.aspect.start, q.aspect.end, w2s
                    )
                    if target is not None and target in span_index_map:
                        asp_labels[b, span_index_map[target]] = 1.0
                if not q.opinion.is_null and w2s is not None:
                    target = self._word_span_to_subword(
                        q.opinion.start, q.opinion.end, w2s
                    )
                    if target is not None and target in span_index_map:
                        opn_labels[b, span_index_map[target]] = 1.0

        # Focal BCE for span loss
        loss_span = (
            self._focal_bce(proposal_out["asp_scores"], asp_labels) +
            self._focal_bce(proposal_out["opn_scores"], opn_labels)
        )
        # =====================================================================
        # L_pair, L_cat, L_aff: Assign labels to pruned pair candidates
        # =====================================================================
        num_pairs = len(pair_map)
        pair_labels = torch.zeros(batch_size, num_pairs, device=device)
        cat_labels = torch.full((batch_size, num_pairs), -1,
                                dtype=torch.long, device=device)
        aff_labels = torch.full((batch_size, num_pairs), -1,
                                dtype=torch.long, device=device)
        easy_neg_mask = torch.zeros(batch_size, num_pairs, device=device)
        span_nearmiss_mask = torch.zeros(batch_size, num_pairs, device=device)
        cat_confused_mask = torch.zeros(batch_size, num_pairs, device=device)
        null_related_mask = torch.zeros(batch_size, num_pairs, device=device)
        semantic_nearmiss_rank_mask = torch.zeros(batch_size, num_pairs, device=device)
        ma_cat_targets = torch.zeros(
            batch_size, num_pairs, self.config.num_categories, device=device
        )
        ma_sent_targets = (
            torch.zeros(batch_size, num_pairs, self.config.num_sentiments, device=device)
            if self.config.task_type == "asqp"
            else None
        )

        for b in range(batch_size):
            if gold_quads is None or gold_quads[b] is None:
                continue

            asp_indices = pruned["asp_indices"][b]
            opn_indices = pruned["opn_indices"][b]
            w2s = word_to_subword[b] if word_to_subword else None

            # Build gold pair set (converted to subword-level)
            gold_pairs = {}
            gold_pair_multi = {}
            for q in gold_quads[b]:
                if not q.aspect.is_null and w2s is not None:
                    a_span = self._word_span_to_subword(
                        q.aspect.start, q.aspect.end, w2s
                    )
                    if a_span is None:
                        continue
                elif q.aspect.is_null:
                    a_span = (-1, -1)
                else:
                    continue

                if not q.opinion.is_null and w2s is not None:
                    o_span = self._word_span_to_subword(
                        q.opinion.start, q.opinion.end, w2s
                    )
                    if o_span is None:
                        continue
                elif q.opinion.is_null:
                    o_span = (-1, -1)
                else:
                    continue

                pair_key = (a_span, o_span)
                gold_pairs[pair_key] = q
                if pair_key not in gold_pair_multi:
                    gold_pair_multi[pair_key] = {
                        "cat_ids": set(),
                        "sent_ids": set(),
                    }
                if cat_to_id and q.category in cat_to_id:
                    gold_pair_multi[pair_key]["cat_ids"].add(cat_to_id[q.category])
                if self.config.task_type == "asqp" and q.sentiment in SENTIMENT_TO_ID:
                    gold_pair_multi[pair_key]["sent_ids"].add(SENTIMENT_TO_ID[q.sentiment])

            gold_asp_spans = set()
            gold_opn_spans = set()
            for q in gold_quads[b]:
                if not q.aspect.is_null and w2s is not None:
                    sub = self._word_span_to_subword(
                        q.aspect.start, q.aspect.end, w2s
                    )
                    if sub is not None:
                        gold_asp_spans.add(sub)
                if not q.opinion.is_null and w2s is not None:
                    sub = self._word_span_to_subword(
                        q.opinion.start, q.opinion.end, w2s
                    )
                    if sub is not None:
                        gold_opn_spans.add(sub)

            for p, (ai, oi) in enumerate(pair_map):
                if pair_valid_mask is not None and not pair_valid_mask[b, p]:
                    continue
                a_span = asp_indices[ai]
                o_span = opn_indices[oi]
                pair_key = (a_span, o_span)

                if pair_key in gold_pairs:
                    pair_labels[b, p] = 1.0
                    q = gold_pairs[pair_key]
                    if cat_to_id and q.category in cat_to_id:
                        cat_labels[b, p] = cat_to_id[q.category]
                    if q.sentiment in SENTIMENT_TO_ID:
                        aff_labels[b, p] = SENTIMENT_TO_ID[q.sentiment]
                    multi = gold_pair_multi.get(pair_key)
                    if multi is not None:
                        for cid in multi["cat_ids"]:
                            if 0 <= int(cid) < self.config.num_categories:
                                ma_cat_targets[b, p, int(cid)] = 1.0
                        if ma_sent_targets is not None:
                            for sid in multi["sent_ids"]:
                                if 0 <= int(sid) < self.config.num_sentiments:
                                    ma_sent_targets[b, p, int(sid)] = 1.0
                else:
                    a_is_gold = (a_span in gold_asp_spans and a_span != (-1, -1))
                    o_is_gold = (o_span in gold_opn_spans and o_span != (-1, -1))
                    if a_is_gold and o_is_gold:
                        cat_confused_mask[b, p] = 1.0
                    elif a_is_gold or o_is_gold:
                        span_nearmiss_mask[b, p] = 1.0
                    else:
                        easy_neg_mask[b, p] = 1.0

                    is_null_related = (a_span == (-1, -1) or o_span == (-1, -1))
                    if is_null_related:
                        null_related_mask[b, p] = 1.0
                    elif cat_confused_mask[b, p] == 0:
                        a_overlap = any(
                            self._span_overlap(a_span, gold_span)
                            for gold_span in gold_asp_spans
                        )
                        o_overlap = any(
                            self._span_overlap(o_span, gold_span)
                            for gold_span in gold_opn_spans
                        )
                        if a_overlap and o_overlap:
                            semantic_nearmiss_rank_mask[b, p] = 1.0

        # L_pair: difficulty-aware weighted BCE (+ optional focal modulation)
        pair_bce = F.binary_cross_entropy_with_logits(
            pair_out["pair_scores"], pair_labels, reduction="none"
        )
        cfg = self.config
        valid_pair_weight = (
            pair_valid_mask.float()
            if pair_valid_mask is not None
            else torch.ones_like(pair_bce)
        )
        weight = torch.full_like(pair_bce, cfg.pair_easy_neg_weight)
        weight[span_nearmiss_mask.bool()] = cfg.pair_span_nearmiss_weight
        weight[cat_confused_mask.bool()] = cfg.pair_cat_confused_weight
        weight[(pair_labels == 1.0)] = cfg.pair_pos_weight
        weight = weight * valid_pair_weight

        pair_probs = torch.sigmoid(pair_out["pair_scores"])
        pair_focal_gamma = getattr(cfg, "pair_focal_gamma", 0.0)
        if pair_focal_gamma > 0:
            p_t = pair_probs * pair_labels + (1 - pair_probs) * (1 - pair_labels)
            pair_focal = (1 - p_t) ** pair_focal_gamma
        else:
            pair_focal = torch.ones_like(pair_bce)

        pair_loss_num = (pair_bce * weight * pair_focal).sum()
        pair_loss_den = valid_pair_weight.sum().clamp_min(1.0)
        loss_pair = pair_loss_num / pair_loss_den
        if float(getattr(cfg, "lambda_pair_rank", 0.0)) > 0:
            loss_pair_rank = self._compute_pair_rank_loss(
                pair_probs=pair_probs,
                pair_labels=pair_labels,
                null_related_mask=null_related_mask,
                semantic_nearmiss_rank_mask=semantic_nearmiss_rank_mask,
                cat_confused_mask=cat_confused_mask,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_pair_rank = torch.tensor(0.0, device=device)
        pacr_enabled = bool(getattr(cfg, "use_pacr_loss", False))
        if pacr_enabled:
            loss_pacr, pacr_stats = self._compute_pacr_loss(
                pair_probs=pair_probs,
                pair_labels=pair_labels,
                pair_map=pair_map,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_pacr = torch.tensor(0.0, device=device)
            pacr_stats = {
                "active_pairs": 0,
                "mean_pos_score": 0.0,
                "mean_hardneg_score": 0.0,
                "violation_rate": 0.0,
                "loss_mean": 0.0,
            }
        agml_enabled = bool(getattr(cfg, "use_agml_loss", False))
        agml_comp_enabled = bool(getattr(cfg, "use_agml_comp_loss", False))
        agml_br_enabled = bool(getattr(cfg, "use_agml_br_loss", False))
        cbr_enabled = bool(getattr(cfg, "use_cbr_v1_loss", False))
        romr_enabled = bool(getattr(cfg, "use_romr_v1_loss", False))
        homr_enabled = bool(getattr(cfg, "use_homr_v1_loss", False))
        rph_enabled = bool(getattr(cfg, "use_rph_v1_loss", False))
        ma_enabled = bool(getattr(cfg, "use_ma_aux", False))
        mbl_legacy_enabled = bool(getattr(cfg, "use_mbl_loss", False))
        cat_mbl_enabled = (
            bool(getattr(cfg, "use_cat_mbl_loss", False)) or mbl_legacy_enabled
        )
        sent_mbl_enabled = (
            bool(getattr(cfg, "use_sent_mbl_loss", False)) or mbl_legacy_enabled
        )
        mbl_enabled = cat_mbl_enabled or sent_mbl_enabled
        mbl_lambda_cat = float(
            getattr(cfg, "mbl_lambda_cat", getattr(cfg, "mbl_lambda", 0.05))
        )
        mbl_lambda_sent = float(
            getattr(cfg, "mbl_lambda_sent", getattr(cfg, "mbl_lambda", 0.05))
        )
        if agml_enabled:
            agml_groups = self._collect_agml_aspect_groups(
                pair_labels=pair_labels,
                pair_map=pair_map,
                pair_valid_mask=pair_valid_mask,
                same_aspect_only=bool(getattr(cfg, "agml_same_aspect_only", True)),
            )
            loss_agml, agml_stats = self._compute_agml_loss(
                pair_scores=pair_out["pair_scores"],
                groups=agml_groups,
            )
            if agml_comp_enabled:
                loss_agml_comp, agml_comp_stats = self._compute_agml_comp_loss(
                    pair_scores=pair_out["pair_scores"],
                    groups=agml_groups,
                )
            else:
                loss_agml_comp = torch.tensor(0.0, device=device)
                agml_comp_stats = {
                    "active_aspects": 0,
                    "mean_pos_group": 0.0,
                    "mean_neg_group": 0.0,
                    "violation_rate": 0.0,
                    "loss_mean": 0.0,
                }
        else:
            loss_agml = torch.tensor(0.0, device=device)
            agml_stats = {
                "active_aspects": 0,
                "mean_gold_mass": 0.0,
                "loss_mean": 0.0,
            }
            loss_agml_comp = torch.tensor(0.0, device=device)
            agml_comp_stats = {
                "active_aspects": 0,
                "mean_pos_group": 0.0,
                "mean_neg_group": 0.0,
                "violation_rate": 0.0,
                "loss_mean": 0.0,
            }
        if agml_br_enabled:
            loss_agml_br, agml_br_stats = self._compute_agml_br_loss(
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_agml_br = torch.tensor(0.0, device=device)
            agml_br_stats = {
                "active_gold_pairs": 0,
                "mean_boundary": 0.0,
                "mean_gold_score": 0.0,
                "violation_rate": 0.0,
                "loss_mean": 0.0,
            }
        if cbr_enabled:
            loss_cbr, cbr_stats = self._compute_cbr_v1_loss(
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_cbr = torch.tensor(0.0, device=device)
            cbr_stats = {
                "num_samples_with_active_boundary_loss": 0,
                "boundary_active_positive_ratio": 0.0,
                "avg_cutoff_gap": 0.0,
                "loss_mean": 0.0,
            }
        if romr_enabled:
            loss_romr, romr_stats = self._compute_romr_v1_loss(
                cat_logits=pair_out["cat_logits"],
                aff_logits=pair_out["aff_output"],
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                pair_map=pair_map,
                ma_cat_targets=ma_cat_targets,
                ma_sent_targets=ma_sent_targets,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_romr = torch.tensor(0.0, device=device)
            romr_stats = {
                "active_aspects": 0,
                "active_pairs": 0,
                "mean_pos_score": 0.0,
                "mean_hardneg_score": 0.0,
                "violation_rate": 0.0,
                "loss_mean": 0.0,
            }
        if homr_enabled:
            loss_homr, homr_stats = self._compute_homr_v1_loss(
                pair_reprs=pair_out["pair_reprs"],
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                pair_map=pair_map,
                ma_cat_targets=ma_cat_targets,
                ma_sent_targets=ma_sent_targets,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_homr = torch.tensor(0.0, device=device)
            homr_stats = {
                "active_aspects": 0,
                "active_pairs": 0,
                "mean_pos_score": 0.0,
                "mean_hardneg_score": 0.0,
                "violation_rate": 0.0,
                "loss_mean": 0.0,
            }
        if rph_enabled:
            loss_rph, rph_stats = self._compute_rph_v1_loss(
                pair_reprs=pair_out["pair_reprs"],
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_rph = torch.tensor(0.0, device=device)
            rph_stats = {
                "active_pairs": 0,
                "pos_pairs": 0,
                "neg_pairs": 0,
                "mean_pos_prob": 0.0,
                "mean_neg_prob": 0.0,
                "loss_mean": 0.0,
            }
        if mbl_enabled:
            loss_cat_mbl, loss_sent_mbl, mbl_stats = self._compute_mbl_loss(
                cat_logits=pair_out["cat_logits"],
                aff_logits=pair_out["aff_output"],
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                ma_cat_targets=ma_cat_targets,
                ma_sent_targets=ma_sent_targets,
                pair_valid_mask=pair_valid_mask,
                enable_cat=cat_mbl_enabled,
                enable_sent=sent_mbl_enabled,
            )
            loss_mbl = (
                (mbl_lambda_cat * loss_cat_mbl if cat_mbl_enabled else 0.0) +
                (mbl_lambda_sent * loss_sent_mbl if sent_mbl_enabled else 0.0)
            )
            mbl_stats["loss_mean"] = float(loss_mbl.detach().item())
        else:
            loss_cat_mbl = torch.tensor(0.0, device=device)
            loss_sent_mbl = torch.tensor(0.0, device=device)
            loss_mbl = torch.tensor(0.0, device=device)
            mbl_stats = {
                "active_pairs": 0,
                "cat_active_pairs": 0,
                "sent_active_pairs": 0,
                "cat_violation_rate": 0.0,
                "sent_violation_rate": 0.0,
                "cat_loss_mean": 0.0,
                "sent_loss_mean": 0.0,
                "loss_mean": 0.0,
            }
        if ma_enabled:
            loss_ma_aux, ma_aux_stats = self._compute_ma_aux_loss(
                pair_reprs=pair_out["pair_reprs"],
                pair_scores=pair_out["pair_scores"],
                pair_labels=pair_labels,
                ma_cat_targets=ma_cat_targets,
                ma_sent_targets=ma_sent_targets,
                pair_valid_mask=pair_valid_mask,
            )
        else:
            loss_ma_aux = torch.tensor(0.0, device=device)
            ma_aux_stats = {
                "active_pairs": 0,
                "pos_pairs": 0,
                "neg_pairs": 0,
                "cat_loss_mean": 0.0,
                "sent_loss_mean": 0.0,
                "loss_mean": 0.0,
            }

        # L_cat and L_aff: only on positive pairs
        valid_mask = (pair_labels == 1.0) & (cat_labels >= 0)
        if pair_valid_mask is not None:
            valid_mask = valid_mask & pair_valid_mask.bool()
        if valid_mask.any():
            loss_cat = F.cross_entropy(
                pair_out["cat_logits"][valid_mask],
                cat_labels[valid_mask],
            )
        else:
            loss_cat = torch.tensor(0.0, device=device)

        valid_aff = (pair_labels == 1.0) & (aff_labels >= 0)
        if pair_valid_mask is not None:
            valid_aff = valid_aff & pair_valid_mask.bool()
        if valid_aff.any():
            loss_aff = F.cross_entropy(
                pair_out["aff_output"][valid_aff],
                aff_labels[valid_aff],
            )
        else:
            loss_aff = torch.tensor(0.0, device=device)

        # Weighted combination
        loss_total = (
            cfg.lambda_span * loss_span +
            cfg.lambda_pair * loss_pair +
            cfg.lambda_pair_rank * loss_pair_rank +
            (cfg.pacr_lambda * loss_pacr if pacr_enabled else 0.0) +
            (cfg.agml_lambda * loss_agml if agml_enabled else 0.0) +
            (cfg.agml_comp_lambda * loss_agml_comp if agml_comp_enabled else 0.0) +
            (cfg.agml_br_lambda * loss_agml_br if agml_br_enabled else 0.0) +
            (cfg.cbr_v1_lambda * loss_cbr if cbr_enabled else 0.0) +
            (cfg.romr_v1_lambda * loss_romr if romr_enabled else 0.0) +
            (cfg.homr_v1_lambda * loss_homr if homr_enabled else 0.0) +
            (cfg.rph_v1_lambda * loss_rph if rph_enabled else 0.0) +
            (loss_mbl if mbl_enabled else 0.0) +
            (cfg.ma_aux_lambda * loss_ma_aux if ma_enabled else 0.0) +
            cfg.lambda_cat * loss_cat +
            cfg.lambda_aff * loss_aff
        )

        out = {
            "loss_total": loss_total,
            "loss_span": loss_span.detach(),
            "loss_pair": loss_pair.detach(),
            "loss_pair_rank": loss_pair_rank.detach(),
            "loss_pacr": loss_pacr.detach(),
            "loss_agml": loss_agml.detach(),
            "loss_agml_comp": loss_agml_comp.detach(),
            "loss_agml_br": loss_agml_br.detach(),
            "loss_cbr": loss_cbr.detach(),
            "loss_romr": loss_romr.detach(),
            "loss_homr": loss_homr.detach(),
            "loss_rph": loss_rph.detach(),
            "loss_cat_mbl": loss_cat_mbl.detach(),
            "loss_sent_mbl": loss_sent_mbl.detach(),
            "loss_mbl": loss_mbl.detach(),
            "loss_ma_aux": loss_ma_aux.detach(),
            "loss_cat": loss_cat.detach(),
            "loss_aff": loss_aff.detach(),
        }
        if pacr_enabled:
            out.update({
                "pacr_active_pairs": pacr_stats["active_pairs"],
                "pacr_mean_pos_score": pacr_stats["mean_pos_score"],
                "pacr_mean_hardneg_score": pacr_stats["mean_hardneg_score"],
                "pacr_violation_rate": pacr_stats["violation_rate"],
                "pacr_loss_mean": pacr_stats["loss_mean"],
            })
        if agml_enabled:
            out.update({
                "agml_active_aspects": agml_stats["active_aspects"],
                "agml_mean_gold_mass": agml_stats["mean_gold_mass"],
                "agml_loss_mean": agml_stats["loss_mean"],
            })
        if agml_comp_enabled:
            out.update({
                "agml_comp_active_aspects": agml_comp_stats["active_aspects"],
                "agml_comp_mean_pos_group": agml_comp_stats["mean_pos_group"],
                "agml_comp_mean_neg_group": agml_comp_stats["mean_neg_group"],
                "agml_comp_violation_rate": agml_comp_stats["violation_rate"],
                "agml_comp_loss_mean": agml_comp_stats["loss_mean"],
            })
        if agml_br_enabled:
            out.update({
                "agml_br_active_gold_pairs": agml_br_stats["active_gold_pairs"],
                "agml_br_mean_boundary": agml_br_stats["mean_boundary"],
                "agml_br_mean_gold_score": agml_br_stats["mean_gold_score"],
                "agml_br_violation_rate": agml_br_stats["violation_rate"],
                "agml_br_loss_mean": agml_br_stats["loss_mean"],
            })
        if cbr_enabled:
            out.update({
                "num_samples_with_active_boundary_loss": (
                    cbr_stats["num_samples_with_active_boundary_loss"]
                ),
                "boundary_active_positive_ratio": (
                    cbr_stats["boundary_active_positive_ratio"]
                ),
                "avg_cutoff_gap": cbr_stats["avg_cutoff_gap"],
                "cbr_loss_mean": cbr_stats["loss_mean"],
            })
        if romr_enabled:
            out.update({
                "romr_active_aspects": romr_stats["active_aspects"],
                "romr_active_pairs": romr_stats["active_pairs"],
                "romr_mean_pos_score": romr_stats["mean_pos_score"],
                "romr_mean_hardneg_score": romr_stats["mean_hardneg_score"],
                "romr_violation_rate": romr_stats["violation_rate"],
                "romr_loss_mean": romr_stats["loss_mean"],
            })
        if homr_enabled:
            out.update({
                "homr_active_aspects": homr_stats["active_aspects"],
                "homr_active_pairs": homr_stats["active_pairs"],
                "homr_mean_pos_score": homr_stats["mean_pos_score"],
                "homr_mean_hardneg_score": homr_stats["mean_hardneg_score"],
                "homr_violation_rate": homr_stats["violation_rate"],
                "homr_loss_mean": homr_stats["loss_mean"],
            })
        if rph_enabled:
            out.update({
                "rph_active_pairs": rph_stats["active_pairs"],
                "rph_pos_pairs": rph_stats["pos_pairs"],
                "rph_neg_pairs": rph_stats["neg_pairs"],
                "rph_mean_pos_prob": rph_stats["mean_pos_prob"],
                "rph_mean_neg_prob": rph_stats["mean_neg_prob"],
                "rph_loss_mean": rph_stats["loss_mean"],
            })
        if mbl_enabled:
            out.update({
                "mbl_active_pairs": mbl_stats["active_pairs"],
                "mbl_cat_active_pairs": mbl_stats["cat_active_pairs"],
                "mbl_sent_active_pairs": mbl_stats["sent_active_pairs"],
                "mbl_cat_violation_rate": mbl_stats["cat_violation_rate"],
                "mbl_sent_violation_rate": mbl_stats["sent_violation_rate"],
                "mbl_cat_loss_mean": mbl_stats["cat_loss_mean"],
                "mbl_sent_loss_mean": mbl_stats["sent_loss_mean"],
                "mbl_loss_mean": mbl_stats["loss_mean"],
            })
        if ma_enabled:
            out.update({
                "ma_aux_active_pairs": ma_aux_stats["active_pairs"],
                "ma_aux_pos_pairs": ma_aux_stats["pos_pairs"],
                "ma_aux_neg_pairs": ma_aux_stats["neg_pairs"],
                "ma_aux_cat_loss_mean": ma_aux_stats["cat_loss_mean"],
                "ma_aux_sent_loss_mean": ma_aux_stats["sent_loss_mean"],
                "ma_aux_loss_mean": ma_aux_stats["loss_mean"],
            })
        return out

    def _focal_bce(self, logits, targets):
        """Focal binary cross-entropy loss."""
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * ce).mean()

    @staticmethod
    def _select_pair_ids(scored_ids, scored_values, strategy, pair_thr, pair_top_n):
        """Select retained pair ids according configured strategy."""
        if strategy == "topn_only":
            if pair_top_n is None or pair_top_n <= 0:
                return scored_ids
            return scored_ids[:pair_top_n]

        if strategy == "pair_gate_only":
            return [pid for pid in scored_ids if scored_values[pid] >= pair_thr]

        if strategy == "pair_gate_topn":
            gated = [pid for pid in scored_ids if scored_values[pid] >= pair_thr]
            if pair_top_n is None or pair_top_n <= 0:
                return gated
            return gated[:pair_top_n]

        raise ValueError(f"Unknown pair retention strategy: {strategy}")

    def _build_candidates(self, proposal_out, pair_out, pruned, pair_map,
                          word_to_subword=None, pair_score_threshold=None,
                          pair_retention_strategy=None, pair_top_n=None,
                          pair_valid_mask=None):
        """Build candidate quads for Stage-2 reranking."""
        batch_size = pair_out["pair_scores"].size(0)
        candidates_per_example = []
        selected_pair_ids_per_example = []
        pair_thr = (
            getattr(self.config, "stage1_pair_score_threshold", 0.01)
            if pair_score_threshold is None else pair_score_threshold
        )
        retention_strategy = (
            getattr(self.config, "stage1_pair_retention_strategy", "topn_only")
            if pair_retention_strategy is None else pair_retention_strategy
        )
        retention_top_n = (
            getattr(self.config, "stage1_pair_top_n", 20)
            if pair_top_n is None else pair_top_n
        )

        for b in range(batch_size):
            asp_indices = pruned["asp_indices"][b]
            opn_indices = pruned["opn_indices"][b]
            w2s = word_to_subword[b] if word_to_subword else None

            cat_probs = torch.softmax(pair_out["cat_logits"][b], dim=-1)
            aff_preds = torch.argmax(pair_out["aff_output"][b], dim=-1)
            pair_scores = torch.sigmoid(pair_out["pair_scores"][b]).tolist()
            valid_pair_ids = range(len(pair_map))
            if pair_valid_mask is not None:
                valid_pair_ids = [
                    pid for pid in range(len(pair_map))
                    if bool(pair_valid_mask[b, pid].item())
                ]
            scored_pair_ids = sorted(
                valid_pair_ids,
                key=lambda pid: pair_scores[pid],
                reverse=True,
            )
            selected_pair_ids = self._select_pair_ids(
                scored_pair_ids,
                pair_scores,
                strategy=retention_strategy,
                pair_thr=pair_thr,
                pair_top_n=retention_top_n,
            )
            selected_pair_ids_per_example.append(selected_pair_ids)

            # RPH-v1 decode-time local reweight:
            # keep retained set fixed, only reorder/reweight within retained pairs.
            selected_for_materialize = list(selected_pair_ids)
            pair_score_for_decode = {int(pid): float(pair_scores[pid]) for pid in selected_pair_ids}
            rph_probe_prob = {int(pid): 0.5 for pid in selected_pair_ids}
            use_rph_decode = bool(getattr(self.config, "use_rph_v1_decode_reweight", False))
            if use_rph_decode and len(selected_pair_ids) > 0:
                alpha = float(getattr(self.config, "rph_v1_decode_alpha", 0.10))
                with torch.no_grad():
                    retained_reprs = pair_out["pair_reprs"][b, selected_pair_ids].detach()
                    retained_logits = self.pair_module.rph_probe_head(retained_reprs).squeeze(-1)
                    retained_probs = torch.sigmoid(retained_logits).cpu().tolist()
                for pid, prob in zip(selected_pair_ids, retained_probs):
                    pid = int(pid)
                    rph_probe_prob[pid] = float(prob)
                    pair_score_for_decode[pid] = float(pair_scores[pid] + alpha * (prob - 0.5))
                selected_for_materialize = sorted(
                    selected_for_materialize,
                    key=lambda pid: pair_score_for_decode[int(pid)],
                    reverse=True,
                )

            example_cands = []
            for p in selected_for_materialize:
                ai, oi = pair_map[p]
                pair_score_base = float(pair_scores[p])
                pair_score = float(pair_score_for_decode.get(int(p), pair_score_base))
                probe_prob = float(rph_probe_prob.get(int(p), 0.5))
                a_span_sub = asp_indices[ai]
                o_span_sub = opn_indices[oi]

                # Convert subword spans back to word-level for evaluation
                if w2s is not None:
                    a_span = self._subword_span_to_word(
                        a_span_sub[0], a_span_sub[1], w2s
                    ) if a_span_sub != (-1, -1) else (-1, -1)
                    o_span = self._subword_span_to_word(
                        o_span_sub[0], o_span_sub[1], w2s
                    ) if o_span_sub != (-1, -1) else (-1, -1)
                else:
                    a_span = a_span_sub
                    o_span = o_span_sub

                # Top-c categories
                topk_probs, topk_cats = torch.topk(
                    cat_probs[p],
                    min(self.config.top_c_categories, cat_probs.size(-1))
                )

                for rank in range(topk_cats.size(0)):
                    cat_id = topk_cats[rank].item()
                    cat_prob = topk_probs[rank].item()
                    cat_entropy = -(cat_probs[p] * (cat_probs[p] + 1e-10).log()).sum().item()

                    asp_len = a_span[1] - a_span[0] + 1 if a_span[0] >= 0 else 0
                    opn_len = o_span[1] - o_span[0] + 1 if o_span[0] >= 0 else 0

                    cand = {
                        "pair_repr": pair_out["pair_reprs"][b, p].detach(),
                        "pair_score": pair_score,
                        "pair_score_base": pair_score_base,
                        "pair_score_calibrated": pair_score,
                        "rph_probe_prob": probe_prob,
                        "asp_span": a_span,
                        "opn_span": o_span,
                        "category_id": cat_id,
                        "affective": aff_preds[p].item(),
                        "asp_score": pruned["asp_topk_scores"][b, ai].item() if ai < pruned["asp_topk_scores"].size(1) else 0.0,
                        "opn_score": pruned["opn_topk_scores"][b, oi].item() if oi < pruned["opn_topk_scores"].size(1) else 0.0,
                        "cat_prob": cat_prob,
                        "cat_entropy": cat_entropy,
                        "has_null_asp": a_span == (-1, -1),
                        "has_null_opn": o_span == (-1, -1),
                        "asp_length": asp_len,
                        "opn_length": opn_len,
                        "meta_features": [
                            pruned["asp_topk_scores"][b, ai].item() if ai < pruned["asp_topk_scores"].size(1) else 0.0,
                            pruned["opn_topk_scores"][b, oi].item() if oi < pruned["opn_topk_scores"].size(1) else 0.0,
                            pair_score,
                            cat_prob,
                            cat_entropy,
                            float(a_span == (-1, -1)),
                            float(o_span == (-1, -1)),
                            float(asp_len),
                            float(opn_len),
                        ],
                    }
                    example_cands.append(cand)

            candidates_per_example.append(example_cands)

        return {
            "candidates": candidates_per_example,
            "pair_reprs": pair_out["pair_reprs"],
            "pair_scores": pair_out["pair_scores"],
            "cat_logits": pair_out["cat_logits"],
            "aff_output": pair_out["aff_output"],
            "pair_map": pair_map,
            "asp_indices": pruned["asp_indices"],
            "opn_indices": pruned["opn_indices"],
            "selected_pair_ids": selected_pair_ids_per_example,
            "pair_retention_strategy": retention_strategy,
            "pair_top_n": retention_top_n,
            "pair_score_threshold": pair_thr,
        }
