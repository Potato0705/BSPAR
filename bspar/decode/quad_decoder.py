"""Quad candidate expansion, NMS deduplication, and final selection."""

import torch


def expand_quads(pair_candidates, cat_logits, aff_output,
                 top_c: int = 3, task_type: str = "asqp"):
    """Expand each pair into candidate quads by top-c categories.

    Args:
        pair_candidates: list of pair dicts
        cat_logits: (num_pairs, num_categories) — category logits
        aff_output: (num_pairs, num_sentiments) or (num_pairs, 2)
        top_c: number of category candidates per pair
        task_type: "asqp" or "dimabsa"

    Returns:
        list of quad candidate dicts
    """
    cat_probs = torch.softmax(cat_logits, dim=-1)
    quad_candidates = []

    for i, pair in enumerate(pair_candidates):
        # Top-c categories for this pair
        probs = cat_probs[i]
        topk_probs, topk_ids = torch.topk(probs, min(top_c, probs.size(0)))

        for rank in range(topk_ids.size(0)):
            cat_id = topk_ids[rank].item()
            cat_prob = topk_probs[rank].item()

            if task_type == "asqp":
                sentiment_id = torch.argmax(aff_output[i]).item()
                affective = sentiment_id
            else:
                v = aff_output[i, 0].item()
                ar = aff_output[i, 1].item()
                affective = (v, ar)

            # Compute category entropy as confidence signal
            entropy = -(probs * (probs + 1e-10).log()).sum().item()

            quad_candidates.append({
                "pair_idx": i,
                "asp_span": pair["asp_span"],
                "opn_span": pair["opn_span"],
                "category_id": cat_id,
                "category_prob": cat_prob,
                "category_entropy": entropy,
                "affective": affective,
                "asp_score": pair["asp_score"],
                "opn_score": pair["opn_score"],
                "has_null_asp": pair["has_null_asp"],
                "has_null_opn": pair["has_null_opn"],
            })

    return quad_candidates


def nms_dedup(quad_candidates, scores):
    """Non-maximum suppression for quad deduplication.

    Suppress a quad if a higher-scored quad shares the same aspect span
    AND same category, or same opinion span AND same category.

    Args:
        quad_candidates: list of quad dicts, sorted by score descending
        scores: corresponding S(q) values

    Returns:
        list of selected quad dicts
    """
    selected = []
    selected_keys = set()

    sorted_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i], reverse=True)

    for idx in sorted_indices:
        quad = quad_candidates[idx]
        asp_key = (quad["asp_span"], quad["category_id"])
        opn_key = (quad["opn_span"], quad["category_id"])

        if asp_key in selected_keys or opn_key in selected_keys:
            continue  # suppressed

        selected.append(quad)
        selected_keys.add(asp_key)
        selected_keys.add(opn_key)

    return selected
