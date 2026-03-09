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
