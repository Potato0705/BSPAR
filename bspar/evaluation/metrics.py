"""Evaluation metrics for BSPAR.

Primary metric: Quad-level F1 (exact match on all four elements).
Secondary metrics: Pair F1, Span F1, Category accuracy.
A3 diagnostics: gold pair ranking analysis for pair scoring evaluation.
"""

import math
from collections import Counter


def quad_f1(predictions, golds, match_affective=True):
    """Compute precision, recall, F1 at the quad level.

    A predicted quad is correct iff it exactly matches a gold quad
    on all four elements (aspect, opinion, category, sentiment/VA).

    Args:
        predictions: list of list of Quad (per example)
        golds: list of list of Quad (per example)
        match_affective: whether to require affective match

    Returns:
        dict with precision, recall, f1, counts
    """
    tp = 0
    total_pred = 0
    total_gold = 0

    for preds, golds_i in zip(predictions, golds):
        total_pred += len(preds)
        total_gold += len(golds_i)

        # Greedy matching to avoid double-counting
        matched_gold = set()
        for p in preds:
            for g_idx, g in enumerate(golds_i):
                if g_idx in matched_gold:
                    continue
                if p.matches(g, match_affective=match_affective):
                    tp += 1
                    matched_gold.add(g_idx)
                    break

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "total_pred": total_pred,
        "total_gold": total_gold,
    }


def pair_f1(predictions, golds):
    """F1 on (aspect, opinion) pairs only, ignoring category/sentiment."""
    tp = 0
    total_pred = 0
    total_gold = 0

    for preds, golds_i in zip(predictions, golds):
        pred_pairs = {(p.aspect, p.opinion) for p in preds}
        gold_pairs = {(g.aspect, g.opinion) for g in golds_i}
        total_pred += len(pred_pairs)
        total_gold += len(gold_pairs)
        tp += len(pred_pairs & gold_pairs)

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def span_f1(predictions, golds, role="aspect"):
    """F1 on individual span extraction (aspect or opinion)."""
    tp = 0
    total_pred = 0
    total_gold = 0

    for preds, golds_i in zip(predictions, golds):
        if role == "aspect":
            pred_spans = {p.aspect for p in preds}
            gold_spans = {g.aspect for g in golds_i}
        else:
            pred_spans = {p.opinion for p in preds}
            gold_spans = {g.opinion for g in golds_i}

        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
        tp += len(pred_spans & gold_spans)

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def category_accuracy(predictions, golds):
    """Category accuracy on correctly paired quads."""
    correct = 0
    total = 0

    for preds, golds_i in zip(predictions, golds):
        for p in preds:
            for g in golds_i:
                if p.aspect == g.aspect and p.opinion == g.opinion:
                    total += 1
                    if p.category == g.category:
                        correct += 1
                    break

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def compute_quad_f1(pred_cand_lists, gold_quad_lists, id_to_cat, cat_to_id):
    """Compute Quad-F1 from candidate dicts and gold Quad objects.

    This bridges the gap between Stage-1 candidate dict output and
    the Quad-based metrics above.

    Args:
        pred_cand_lists: list[list[dict]] — candidate dicts per example
        gold_quad_lists: list[list[Quad]] — gold quads per example
        id_to_cat: dict — category id → name
        cat_to_id: dict — category name → id

    Returns:
        dict with quad_f1, span_f1, and other metrics
    """
    from ..data.schema import Span, Quad
    from ..data.preprocessor import ID_TO_SENTIMENT

    all_pred_quads = []
    all_gold_quads = []

    for preds, golds in zip(pred_cand_lists, gold_quad_lists):
        pred_quads = []
        for c in preds:
            a_span = c["asp_span"]
            o_span = c["opn_span"]

            if a_span == (-1, -1):
                aspect = Span.null("aspect")
            else:
                aspect = Span(start=a_span[0], end=a_span[1], text="")

            if o_span == (-1, -1):
                opinion = Span.null("opinion")
            else:
                opinion = Span(start=o_span[0], end=o_span[1], text="")

            category = id_to_cat.get(c["category_id"], "")
            sentiment = ID_TO_SENTIMENT.get(c["affective"], "NEU")

            pred_quads.append(Quad(
                aspect=aspect, opinion=opinion,
                category=category, sentiment=sentiment,
            ))

        all_pred_quads.append(pred_quads)
        all_gold_quads.append(golds)

    qf1 = quad_f1(all_pred_quads, all_gold_quads)
    sf1_asp = span_f1(all_pred_quads, all_gold_quads, role="aspect")
    sf1_opn = span_f1(all_pred_quads, all_gold_quads, role="opinion")

    return {
        "quad_f1": qf1["f1"],
        "quad_precision": qf1["precision"],
        "quad_recall": qf1["recall"],
        "span_f1": (sf1_asp["f1"] + sf1_opn["f1"]) / 2,
        "asp_f1": sf1_asp["f1"],
        "opn_f1": sf1_opn["f1"],
    }


def compute_a3_diagnostics(all_candidates, gold_quad_lists, cat_to_id,
                           top_n=20):
    """Compute A3 diagnostic metrics from raw Stage-1 candidates.

    For each gold quad whose (asp, opn) pair appears in the candidate set,
    analyze its rank and the types of negatives that outrank it.

    Args:
        all_candidates: list[list[dict]] — all candidates per example (before
            greedy decode, before threshold filtering)
        gold_quad_lists: list[list[Quad]] — gold quads per example
        cat_to_id: dict — category name → id
        top_n: retention budget

    Returns:
        dict with A3 diagnostic metrics
    """
    from ..data.preprocessor import SENTIMENT_TO_ID

    gold_pair_ranks = []
    score_gaps = []           # score@top_n - gold_pair_score
    null_outranker_counts = 0
    near_miss_outranker_counts = 0
    other_outranker_counts = 0
    total_first_outrankers = 0
    total_a3 = 0             # gold pairs present in candidates but outside top_n
    total_gold_in_cands = 0  # gold pairs found in candidate set
    total_gold = 0
    samples_with_positive = 0
    total_samples = len(all_candidates)

    for ex_idx, (cands, golds) in enumerate(zip(all_candidates, gold_quad_lists)):
        if not cands:
            total_gold += len(golds)
            continue

        # Sort candidates by pair_score descending (same as retention)
        sorted_cands = sorted(cands, key=lambda c: c["pair_score"], reverse=True)

        # Build gold pair set: (asp_span, opn_span) -> quad
        gold_pair_keys = {}
        for q in golds:
            total_gold += 1
            a_key = (q.aspect.start, q.aspect.end) if not q.aspect.is_null else (-1, -1)
            o_key = (q.opinion.start, q.opinion.end) if not q.opinion.is_null else (-1, -1)
            cat_id = cat_to_id.get(q.category, -1)
            sent_id = SENTIMENT_TO_ID.get(q.sentiment, -1)
            gold_pair_keys[(a_key, o_key, cat_id, sent_id)] = q

        # Find gold pairs in candidate list and compute ranks
        # Use (asp_span, opn_span, category_id, affective) as full quad key
        has_positive_in_topn = False
        for gold_key, q in gold_pair_keys.items():
            g_asp, g_opn, g_cat, g_sent = gold_key
            # Find this gold quad in sorted candidates
            found_rank = None
            gold_score = None
            for rank, c in enumerate(sorted_cands):
                c_asp = c["asp_span"]
                c_opn = c["opn_span"]
                c_cat = c["category_id"]
                c_aff = c["affective"]
                if (c_asp == g_asp and c_opn == g_opn and
                        c_cat == g_cat and c_aff == g_sent):
                    found_rank = rank
                    gold_score = c["pair_score"]
                    break

            if found_rank is None:
                continue  # Gold quad not in candidates at all (A1/A2 issue)

            total_gold_in_cands += 1
            gold_pair_ranks.append(found_rank)

            if found_rank < top_n:
                has_positive_in_topn = True
            else:
                # A3: gold pair present but outside top_n
                total_a3 += 1
                # Score gap: score of candidate at position top_n vs gold
                if len(sorted_cands) > top_n - 1:
                    score_at_topn = sorted_cands[top_n - 1]["pair_score"]
                    score_gaps.append(score_at_topn - gold_score)

                # Analyze first outranker (the candidate at rank top_n-1 or
                # the first non-gold that displaced this gold)
                # Look at candidates ranked above this gold to find first outranker
                for r in range(min(found_rank, len(sorted_cands))):
                    outranker = sorted_cands[r]
                    o_asp = outranker["asp_span"]
                    o_opn = outranker["opn_span"]
                    # Skip if this is also a gold pair
                    is_gold_outranker = False
                    for gk in gold_pair_keys:
                        if (o_asp == gk[0] and o_opn == gk[1] and
                                outranker["category_id"] == gk[2] and
                                outranker["affective"] == gk[3]):
                            is_gold_outranker = True
                            break
                    if is_gold_outranker:
                        continue

                    # Classify outranker type
                    total_first_outrankers += 1
                    is_null = (o_asp == (-1, -1) or o_opn == (-1, -1))
                    is_near_miss = (
                        (o_asp == g_asp and o_asp != (-1, -1)) or
                        (o_opn == g_opn and o_opn != (-1, -1))
                    )
                    if is_null:
                        null_outranker_counts += 1
                    elif is_near_miss:
                        near_miss_outranker_counts += 1
                    else:
                        other_outranker_counts += 1
                    break  # Only first outranker

        if has_positive_in_topn:
            samples_with_positive += 1

    # Compute summary statistics
    result = {
        "total_gold": total_gold,
        "total_gold_in_cands": total_gold_in_cands,
        "a3_count": total_a3,
        "sample_has_positive_after_retention_ratio": (
            samples_with_positive / total_samples if total_samples > 0 else 0.0
        ),
    }

    if gold_pair_ranks:
        sorted_ranks = sorted(gold_pair_ranks)
        result["gold_pair_mean_rank"] = sum(gold_pair_ranks) / len(gold_pair_ranks)
        result["gold_pair_median_rank"] = sorted_ranks[len(sorted_ranks) // 2]
        n = len(sorted_ranks)
        result["gold_pair_q90_rank"] = sorted_ranks[min(int(n * 0.9), n - 1)]

    if score_gaps:
        sorted_gaps = sorted(score_gaps)
        result["score_gap_mean"] = sum(score_gaps) / len(score_gaps)
        result["score_gap_median"] = sorted_gaps[len(sorted_gaps) // 2]

    if total_first_outrankers > 0:
        result["null_outranker_ratio"] = null_outranker_counts / total_first_outrankers
        result["near_miss_outranker_ratio"] = near_miss_outranker_counts / total_first_outrankers
        result["other_outranker_ratio"] = other_outranker_counts / total_first_outrankers
    else:
        result["null_outranker_ratio"] = 0.0
        result["near_miss_outranker_ratio"] = 0.0
        result["other_outranker_ratio"] = 0.0

    return result
