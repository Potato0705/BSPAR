"""Stage-2 composite model: Quad-Aware Reranker.

Operates on pre-computed candidate quads from Stage-1 (real candidates).
Computes quad-level scores and applies ranking/calibration losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quad_reranker import QuadReranker


class BSPARStage2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reranker = QuadReranker(
            pair_repr_size=config.pair_repr_size,
            num_categories=config.num_categories,
            cat_embedding_dim=config.cat_embedding_dim,
            aff_embedding_dim=config.aff_embedding_dim,
            num_meta_features=config.num_meta_features,
            task_type=config.task_type,
            num_sentiments=config.num_sentiments,
        )
        self.margin = config.ranking_margin

        # Lightweight pair-prior head for pair-level calibration.
        self.pair_prior_head = nn.Sequential(
            nn.Linear(config.pair_repr_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, pair_reprs, cat_ids, aff_input, meta_features,
                labels=None, mode="train", asp_spans=None, opn_spans=None):
        """
        Args:
            pair_reprs:    (batch, num_cands, pair_repr_size)
            cat_ids:       (batch, num_cands) category indices
            aff_input:     sentiment indices or (v, ar) values
            meta_features: (batch, num_cands, num_meta_features)
            labels:        (batch, num_cands) 1 for gold, 0 for negative
            mode:          "train" or "inference"

        Returns:
            dict with quad_scores, and losses/metrics if training
        """
        quad_scores_raw = self.reranker(pair_reprs, cat_ids, aff_input, meta_features)
        pair_prior_logits = self.pair_prior_head(pair_reprs).squeeze(-1)

        alpha = (
            getattr(self.config, "stage2_pair_prior_alpha", 0.0)
            if getattr(self.config, "stage2_use_pair_prior", False)
            else 0.0
        )
        quad_scores = quad_scores_raw + alpha * pair_prior_logits

        result = {
            "quad_scores": quad_scores,
            "quad_scores_raw": quad_scores_raw,
            "pair_prior_logits": pair_prior_logits,
        }

        if mode == "train" and labels is not None:
            loss_rank = self._compute_ranking_loss(quad_scores, labels)

            # Optional L_group (kept for backward compatibility; usually disabled).
            loss_group = quad_scores.sum() * 0.0
            group_stats = {
                "group_count_all": 0,
                "group_count_conflict": 0,
                "group_acc_correct_all": 0.0,
                "group_acc_correct_conflict": 0.0,
                "group_hits1_conflict": 0.0,
                "group_rr_sum_conflict": 0.0,
            }
            group_loss_raw, group_stats = self._compute_group_listwise_loss(
                quad_scores, labels, asp_spans, opn_spans
            )
            if getattr(self.config, "stage2_use_group_loss", False):
                loss_group = group_loss_raw

            # Pair-prior BCE on unique (aspect, opinion) groups.
            pair_logits_g, pair_labels_g = self._build_pair_group_logits_labels(
                pair_prior_logits, labels, asp_spans, opn_spans
            )
            loss_pair_prior = quad_scores.sum() * 0.0
            if (
                getattr(self.config, "stage2_use_pair_prior", False)
                and pair_labels_g.numel() > 0
            ):
                pos_weight_val = max(
                    float(getattr(self.config, "stage2_pair_prior_pos_weight", 1.0)),
                    1e-6,
                )
                pos_weight = torch.tensor([pos_weight_val], device=quad_scores.device)
                loss_pair_prior = F.binary_cross_entropy_with_logits(
                    pair_logits_g,
                    pair_labels_g,
                    pos_weight=pos_weight,
                )

            lambda_group = float(getattr(self.config, "stage2_group_loss_lambda", 0.0))
            lambda_pair = float(getattr(self.config, "stage2_pair_prior_lambda", 0.0))
            loss = loss_rank + lambda_group * loss_group + lambda_pair * loss_pair_prior

            group_count_all = max(group_stats["group_count_all"], 1)
            group_count_conflict = max(group_stats["group_count_conflict"], 1)

            result["loss"] = loss
            result["loss_rank"] = loss_rank
            result["loss_group"] = loss_group
            result["loss_pair_prior"] = loss_pair_prior

            result["group_count_all"] = group_stats["group_count_all"]
            result["group_count_conflict"] = group_stats["group_count_conflict"]
            result["group_acc_correct_all_raw"] = group_stats["group_acc_correct_all"]
            result["group_acc_correct_conflict_raw"] = group_stats["group_acc_correct_conflict"]
            result["group_hits1_conflict_raw"] = group_stats["group_hits1_conflict"]
            result["group_rr_sum_conflict_raw"] = group_stats["group_rr_sum_conflict"]
            result["group_accuracy"] = group_stats["group_acc_correct_all"] / group_count_all
            result["conflict_group_accuracy"] = (
                group_stats["group_acc_correct_conflict"] / group_count_conflict
            )
            result["group_hits1"] = group_stats["group_hits1_conflict"] / group_count_conflict
            result["group_mrr"] = group_stats["group_rr_sum_conflict"] / group_count_conflict

            result["pair_prior_group_logits"] = pair_logits_g
            result["pair_prior_group_labels"] = pair_labels_g
            result["pair_prior_group_count"] = int(pair_labels_g.numel())
            result["pair_prior_group_pos_count"] = int((pair_labels_g == 1).sum().item())

        return result

    def _compute_ranking_loss(self, scores, labels):
        """Pairwise margin ranking loss within each example."""
        batch_size = scores.size(0)
        total_loss = scores.sum() * 0.0
        num_pairs = 0

        for b in range(batch_size):
            pos_mask = labels[b] == 1
            neg_mask = labels[b] == 0

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_scores = scores[b][pos_mask]
            neg_scores = scores[b][neg_mask]

            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            pair_loss = torch.clamp(self.margin - diff, min=0.0)
            total_loss = total_loss + pair_loss.sum()
            num_pairs += pair_loss.numel()

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss

    @staticmethod
    def _span_to_index(span_tensor, span_map):
        """Map span tuple to a stable integer id for one example."""
        span = (int(span_tensor[0].item()), int(span_tensor[1].item()))
        if span == (-1, -1):
            return -1
        if span not in span_map:
            span_map[span] = len(span_map)
        return span_map[span]

    def _build_pair_group_logits_labels(self, pair_logits, labels, asp_spans, opn_spans):
        """Build pair-level logits/labels from candidate-level tensors.

        Pair label: 1 if any candidate in the same (aspect, opinion) group is positive.
        """
        device = pair_logits.device
        empty_logits = torch.zeros(0, device=device, dtype=pair_logits.dtype)
        empty_labels = torch.zeros(0, device=device, dtype=pair_logits.dtype)

        if asp_spans is None or opn_spans is None:
            return empty_logits, empty_labels

        logits_out = []
        labels_out = []

        batch_size = pair_logits.size(0)
        for b in range(batch_size):
            valid = labels[b] >= 0
            if valid.sum().item() == 0:
                continue

            s_b = pair_logits[b][valid]
            y_b = labels[b][valid]
            a_b = asp_spans[b][valid]
            o_b = opn_spans[b][valid]

            group_to_ids = {}
            for i in range(s_b.size(0)):
                key = (
                    int(a_b[i, 0].item()),
                    int(a_b[i, 1].item()),
                    int(o_b[i, 0].item()),
                    int(o_b[i, 1].item()),
                )
                if key not in group_to_ids:
                    group_to_ids[key] = []
                group_to_ids[key].append(i)

            for ids in group_to_ids.values():
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                logit_g = s_b[ids_t].mean()
                label_g = 1.0 if (y_b[ids_t] == 1).any().item() else 0.0
                logits_out.append(logit_g)
                labels_out.append(label_g)

        if not logits_out:
            return empty_logits, empty_labels

        logits_tensor = torch.stack(logits_out)
        labels_tensor = torch.tensor(labels_out, dtype=torch.float, device=device)
        return logits_tensor, labels_tensor

    def _compute_group_listwise_loss(self, scores, labels, asp_spans, opn_spans):
        """Softmax-style group objective over pair groups.

        Group key is (aspect_span_index_or_NULL, opinion_span_index_or_NULL),
        where NULL maps to -1. Supports multi-positive groups.
        """
        if asp_spans is None or opn_spans is None:
            return scores.sum() * 0.0, {
                "group_count_all": 0,
                "group_count_conflict": 0,
                "group_acc_correct_all": 0.0,
                "group_acc_correct_conflict": 0.0,
                "group_hits1_conflict": 0.0,
                "group_rr_sum_conflict": 0.0,
            }

        tau = max(getattr(self.config, "stage2_group_tau", 1.0), 1e-6)
        total_group_loss = scores.sum() * 0.0
        num_conflict_groups = 0
        num_groups_all = 0

        group_acc_correct_all = 0.0
        group_acc_correct_conflict = 0.0
        group_hits1_conflict = 0.0
        group_rr_sum_conflict = 0.0

        batch_size = scores.size(0)
        for b in range(batch_size):
            valid = labels[b] >= 0
            if valid.sum().item() < 2:
                continue

            s_b = scores[b][valid]
            y_b = labels[b][valid]
            a_b = asp_spans[b][valid]
            o_b = opn_spans[b][valid]

            asp_map = {}
            opn_map = {}
            group_to_ids = {}

            for i in range(s_b.size(0)):
                a_idx = self._span_to_index(a_b[i], asp_map)
                o_idx = self._span_to_index(o_b[i], opn_map)
                key = (a_idx, o_idx)
                if key not in group_to_ids:
                    group_to_ids[key] = []
                group_to_ids[key].append(i)

            for cand_ids in group_to_ids.values():
                if len(cand_ids) < 2:
                    continue

                ids_t = torch.tensor(cand_ids, dtype=torch.long, device=scores.device)
                s_g = s_b[ids_t]
                y_g = y_b[ids_t]
                pos_mask = y_g == 1
                neg_mask = y_g == 0

                num_groups_all += 1
                top_local = torch.argmax(s_g).item()
                top_is_pos = 1.0 if y_g[top_local].item() == 1 else 0.0
                group_acc_correct_all += top_is_pos

                if not (pos_mask.any() and neg_mask.any()):
                    continue

                num_conflict_groups += 1
                group_acc_correct_conflict += top_is_pos
                group_hits1_conflict += top_is_pos

                sorted_local = torch.argsort(s_g, descending=True)
                rr = 0.0
                for rank, local_idx in enumerate(sorted_local.tolist(), start=1):
                    if y_g[local_idx].item() == 1:
                        rr = 1.0 / rank
                        break
                group_rr_sum_conflict += rr

                logits = s_g / tau
                log_denom = torch.logsumexp(logits, dim=0)
                log_num = torch.logsumexp(logits[pos_mask], dim=0)
                total_group_loss = total_group_loss - (log_num - log_denom)

        if num_conflict_groups > 0:
            total_group_loss = total_group_loss / num_conflict_groups

        return total_group_loss, {
            "group_count_all": num_groups_all,
            "group_count_conflict": num_conflict_groups,
            "group_acc_correct_all": group_acc_correct_all,
            "group_acc_correct_conflict": group_acc_correct_conflict,
            "group_hits1_conflict": group_hits1_conflict,
            "group_rr_sum_conflict": group_rr_sum_conflict,
        }
