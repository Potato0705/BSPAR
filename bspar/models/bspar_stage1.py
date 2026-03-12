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
        pair_inputs, pair_map = self._construct_pairs(pruned)

        # Step 5: Pair prediction
        pair_out = self.pair_module(**pair_inputs)

        if mode == "train":
            return self._compute_stage1_losses(
                proposal_out, pair_out, pruned, pair_map,
                gold_quads, cat_to_id, word_to_subword
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
        K_a = min(self.config.top_k_aspects, asp_scores.size(1))
        K_o = min(self.config.top_k_opinions, opn_scores.size(1))

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

        return {
            "asp_reprs": asp_pairs,
            "opn_reprs": opn_pairs,
            "dist_ids": dist_ids,
            "order_ids": order_ids,
        }, pair_map

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

    def _compute_stage1_losses(self, proposal_out, pair_out, pruned, pair_map,
                                gold_quads, cat_to_id, word_to_subword):
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
        null_neg_mask = torch.zeros(batch_size, num_pairs, device=device)

        for b in range(batch_size):
            if gold_quads is None or gold_quads[b] is None:
                continue

            asp_indices = pruned["asp_indices"][b]
            opn_indices = pruned["opn_indices"][b]
            w2s = word_to_subword[b] if word_to_subword else None

            # Build gold pair set (converted to subword-level)
            gold_pairs = {}
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

                gold_pairs[(a_span, o_span)] = q

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
                else:
                    a_is_gold = (a_span in gold_asp_spans and a_span != (-1, -1))
                    o_is_gold = (o_span in gold_opn_spans and o_span != (-1, -1))
                    if a_is_gold and o_is_gold:
                        cat_confused_mask[b, p] = 1.0
                    elif a_is_gold or o_is_gold:
                        span_nearmiss_mask[b, p] = 1.0
                    else:
                        easy_neg_mask[b, p] = 1.0
                    # Also track NULL-related negatives for ranking loss
                    if a_span == (-1, -1) or o_span == (-1, -1):
                        null_neg_mask[b, p] = 1.0

        # L_pair: difficulty-aware weighted BCE (+ optional focal modulation)
        pair_bce = F.binary_cross_entropy_with_logits(
            pair_out["pair_scores"], pair_labels, reduction="none"
        )
        cfg = self.config
        weight = torch.full_like(pair_bce, cfg.pair_easy_neg_weight)
        weight[span_nearmiss_mask.bool()] = cfg.pair_span_nearmiss_weight
        weight[cat_confused_mask.bool()] = cfg.pair_cat_confused_weight
        weight[(pair_labels == 1.0)] = cfg.pair_pos_weight

        pair_focal_gamma = getattr(cfg, "pair_focal_gamma", 0.0)
        if pair_focal_gamma > 0:
            pair_probs = torch.sigmoid(pair_out["pair_scores"])
            p_t = pair_probs * pair_labels + (1 - pair_probs) * (1 - pair_labels)
            pair_focal = (1 - p_t) ** pair_focal_gamma
        else:
            pair_focal = torch.ones_like(pair_bce)

        loss_pair = (pair_bce * weight * pair_focal).mean()

        # L_pair_rank: margin ranking loss on hard negatives
        loss_pair_rank = torch.tensor(0.0, device=device)
        if cfg.lambda_pair_rank > 0:
            pair_scores_sig = torch.sigmoid(pair_out["pair_scores"])
            # Combined hard neg mask: NULL-related OR span near-miss OR cat-confused
            rank_neg_mask = ((null_neg_mask + span_nearmiss_mask + cat_confused_mask) > 0).float()
            margin = cfg.pair_rank_margin
            num_rank_pairs = 0
            for b in range(batch_size):
                pos_idx = (pair_labels[b] == 1.0).nonzero(as_tuple=True)[0]
                neg_idx = (rank_neg_mask[b] > 0).nonzero(as_tuple=True)[0]
                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue
                pos_scores = pair_scores_sig[b, pos_idx]       # (num_pos,)
                neg_scores = pair_scores_sig[b, neg_idx]       # (num_neg,)
                # Pairwise: (num_pos, 1) vs (1, num_neg)
                diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                pair_rank_loss = torch.clamp(margin - diff, min=0.0)
                loss_pair_rank = loss_pair_rank + pair_rank_loss.sum()
                num_rank_pairs += pair_rank_loss.numel()
            if num_rank_pairs > 0:
                loss_pair_rank = loss_pair_rank / num_rank_pairs

        # L_cat and L_aff: only on positive pairs
        valid_mask = (pair_labels == 1.0) & (cat_labels >= 0)
        if valid_mask.any():
            loss_cat = F.cross_entropy(
                pair_out["cat_logits"][valid_mask],
                cat_labels[valid_mask],
            )
        else:
            loss_cat = torch.tensor(0.0, device=device)

        valid_aff = (pair_labels == 1.0) & (aff_labels >= 0)
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
            cfg.lambda_cat * loss_cat +
            cfg.lambda_aff * loss_aff +
            cfg.lambda_pair_rank * loss_pair_rank
        )

        return {
            "loss_total": loss_total,
            "loss_span": loss_span.detach(),
            "loss_pair": loss_pair.detach(),
            "loss_cat": loss_cat.detach(),
            "loss_aff": loss_aff.detach(),
            "loss_pair_rank": loss_pair_rank.detach() if isinstance(loss_pair_rank, torch.Tensor) else loss_pair_rank,
        }

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
                          pair_retention_strategy=None, pair_top_n=None):
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
            scored_pair_ids = sorted(
                range(len(pair_map)),
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

            example_cands = []
            for p in selected_pair_ids:
                ai, oi = pair_map[p]
                pair_score = pair_scores[p]
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
