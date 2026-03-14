"""Evaluation metrics for BSPAR.

Primary metric: Quad-level F1 (exact match on all four elements).
Secondary metrics: Pair F1, Span F1, Category accuracy.
"""

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


def _quantile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * q)
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def _span_overlap(span_a, span_b):
    if span_a == (-1, -1) or span_b == (-1, -1):
        return False
    return not (span_a[1] < span_b[0] or span_b[1] < span_a[0])


def _classify_a3_outranker(cand_pair, gold_pair, gold_asps, gold_opns):
    del gold_pair
    cand_asp, cand_opn = cand_pair
    if cand_asp == (-1, -1) or cand_opn == (-1, -1):
        return "NULL"

    asp_near = (
        cand_asp in gold_asps or
        any(_span_overlap(cand_asp, gold_asp) for gold_asp in gold_asps)
    )
    opn_near = (
        cand_opn in gold_opns or
        any(_span_overlap(cand_opn, gold_opn) for gold_opn in gold_opns)
    )
    if asp_near and opn_near:
        return "near_miss"
    return "other"


def compute_a3_diagnostics(example_records, pair_top_n):
    """Summarize A3 behavior from per-example Stage-1 pair-space records.

    Args:
        example_records: list of dict. Each dict must contain:
            pair_scores: list[float]
            pair_map: list[tuple[int, int]]
            asp_indices: list[tuple[int, int]]
            opn_indices: list[tuple[int, int]]
            selected_pair_ids: list[int]
            gold_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
        pair_top_n: fixed top-N pair retention budget
    """
    a3_gold_pair_ranks = []
    a3_score_gaps = []
    first_outranker_type_counts = Counter()

    total_gold_pairs = 0
    total_gold_pairs_in_pair_space = 0
    total_a3_gold_pairs = 0
    sample_with_gold_pair = 0
    sample_has_positive_after_retention = 0

    for record in example_records:
        pair_scores = list(record.get("pair_scores", []))
        pair_map = list(record.get("pair_map", []))
        asp_indices = [tuple(span) for span in record.get("asp_indices", [])]
        opn_indices = [tuple(span) for span in record.get("opn_indices", [])]
        selected_pair_ids = set(record.get("selected_pair_ids", []))
        gold_pairs = {
            (tuple(pair[0]), tuple(pair[1]))
            for pair in record.get("gold_pairs", [])
        }

        if gold_pairs:
            sample_with_gold_pair += 1
        total_gold_pairs += len(gold_pairs)

        if not pair_scores or not pair_map:
            continue

        pairid_to_key = {}
        for pid, (ai, oi) in enumerate(pair_map):
            if ai >= len(asp_indices) or oi >= len(opn_indices):
                continue
            pairid_to_key[pid] = (asp_indices[ai], opn_indices[oi])

        retained_pairs = {
            pairid_to_key[pid]
            for pid in selected_pair_ids
            if pid in pairid_to_key
        }
        if gold_pairs & retained_pairs:
            sample_has_positive_after_retention += 1

        sorted_ids = sorted(
            pairid_to_key.keys(),
            key=lambda pid: pair_scores[pid],
            reverse=True,
        )
        if not sorted_ids:
            continue

        rank_map = {pid: rank + 1 for rank, pid in enumerate(sorted_ids)}
        top_n = int(pair_top_n) if pair_top_n is not None else len(sorted_ids)
        top_n = max(1, min(top_n, len(sorted_ids)))
        top_n_score = pair_scores[sorted_ids[top_n - 1]]

        pairkey_to_pid = {}
        for pid, pair_key in pairid_to_key.items():
            if pair_key not in pairkey_to_pid:
                pairkey_to_pid[pair_key] = pid

        gold_asps = {pair[0] for pair in gold_pairs}
        gold_opns = {pair[1] for pair in gold_pairs}

        for gold_pair in gold_pairs:
            pid = pairkey_to_pid.get(gold_pair)
            if pid is None:
                continue

            total_gold_pairs_in_pair_space += 1
            if pid in selected_pair_ids:
                continue

            total_a3_gold_pairs += 1
            a3_gold_pair_ranks.append(rank_map[pid])
            a3_score_gaps.append(top_n_score - pair_scores[pid])

            first_type = "other"
            for outrank_pid in sorted_ids:
                if rank_map[outrank_pid] >= rank_map[pid]:
                    break
                cand_pair = pairid_to_key[outrank_pid]
                if cand_pair in gold_pairs:
                    continue
                first_type = _classify_a3_outranker(
                    cand_pair,
                    gold_pair,
                    gold_asps,
                    gold_opns,
                )
                break
            first_outranker_type_counts[first_type] += 1

    outranker_total = sum(first_outranker_type_counts.values())
    return {
        "counts": {
            "total_gold_pairs": int(total_gold_pairs),
            "total_gold_pairs_in_pair_space": int(total_gold_pairs_in_pair_space),
            "total_a3_gold_pairs": int(total_a3_gold_pairs),
        },
        "gold_pair_rank": {
            "mean": (
                float(sum(a3_gold_pair_ranks) / len(a3_gold_pair_ranks))
                if a3_gold_pair_ranks else 0.0
            ),
            "median": _quantile(a3_gold_pair_ranks, 0.5),
        },
        "score_topn_minus_gold_pair": {
            "mean": (
                float(sum(a3_score_gaps) / len(a3_score_gaps))
                if a3_score_gaps else 0.0
            ),
            "median": _quantile(a3_score_gaps, 0.5),
        },
        "first_outranker_type_ratio": {
            label: {
                "count": int(first_outranker_type_counts.get(label, 0)),
                "ratio": (
                    float(first_outranker_type_counts.get(label, 0) / outranker_total)
                    if outranker_total > 0 else 0.0
                ),
            }
            for label in ["NULL", "near_miss", "other"]
        },
        "sample_has_positive_after_retention_ratio": (
            float(sample_has_positive_after_retention / sample_with_gold_pair)
            if sample_with_gold_pair > 0 else 0.0
        ),
    }
