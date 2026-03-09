"""Span pruning: select top-K aspect/opinion candidates by unary score."""

import torch


def prune_spans(asp_scores, opn_scores, span_reprs, span_indices,
                top_k_asp, top_k_opn, score_threshold=0.0):
    """Prune spans to top-K aspects and top-K opinions.

    Args:
        asp_scores: (num_spans,) — aspect unary scores
        opn_scores: (num_spans,) — opinion unary scores
        span_reprs: (num_spans, span_repr_size)
        span_indices: list of (start, end) tuples
        top_k_asp: max aspect candidates to keep
        top_k_opn: max opinion candidates to keep
        score_threshold: minimum score to keep

    Returns:
        pruned_aspects: list of (index, span_indices, repr, score)
        pruned_opinions: list of (index, span_indices, repr, score)
    """
    # Aspect pruning
    asp_mask = asp_scores > score_threshold
    asp_valid_scores = asp_scores.clone()
    asp_valid_scores[~asp_mask] = float("-inf")
    k_a = min(top_k_asp, asp_valid_scores.size(0))
    asp_topk_scores, asp_topk_ids = torch.topk(asp_valid_scores, k_a)

    # Opinion pruning
    opn_mask = opn_scores > score_threshold
    opn_valid_scores = opn_scores.clone()
    opn_valid_scores[~opn_mask] = float("-inf")
    k_o = min(top_k_opn, opn_valid_scores.size(0))
    opn_topk_scores, opn_topk_ids = torch.topk(opn_valid_scores, k_o)

    pruned_aspects = []
    for rank, idx in enumerate(asp_topk_ids.tolist()):
        if asp_topk_scores[rank].item() == float("-inf"):
            break
        pruned_aspects.append({
            "global_idx": idx,
            "span": span_indices[idx],
            "repr": span_reprs[idx],
            "score": asp_topk_scores[rank].item(),
        })

    pruned_opinions = []
    for rank, idx in enumerate(opn_topk_ids.tolist()):
        if opn_topk_scores[rank].item() == float("-inf"):
            break
        pruned_opinions.append({
            "global_idx": idx,
            "span": span_indices[idx],
            "repr": span_reprs[idx],
            "score": opn_topk_scores[rank].item(),
        })

    return pruned_aspects, pruned_opinions
