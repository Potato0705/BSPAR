"""Hard negative construction for pair and quad discrimination.

Hard negatives are crucial for training discriminative pair/quad scoring.
They prioritize candidates that are easily confused with gold quads.
"""


def construct_hard_negative_pairs(pruned_asp_indices, pruned_opn_indices,
                                  gold_quads, span_indices):
    """Identify hard negative pairs for L_pair training.

    Hard negative categories:
    1. Correct aspect + wrong opinion (near-boundary)
    2. Wrong aspect + correct opinion (near-boundary)
    3. Near-boundary spans (off by 1-2 tokens from gold)
    4. Category-confusable pairs (different gold quad's aspect with
       current opinion)

    Args:
        pruned_asp_indices: list of (start, end) for pruned aspects
        pruned_opn_indices: list of (start, end) for pruned opinions
        gold_quads: list of Quad
        span_indices: full list of enumerated spans

    Returns:
        hard_neg_mask: list of bool, True for hard negative pairs
        pair_labels: list of 0/1 for each pair
    """
    gold_asp = set()
    gold_opn = set()
    gold_pairs = set()

    for q in gold_quads:
        a = (q.aspect.start, q.aspect.end) if not q.aspect.is_null else (-1, -1)
        o = (q.opinion.start, q.opinion.end) if not q.opinion.is_null else (-1, -1)
        gold_asp.add(a)
        gold_opn.add(o)
        gold_pairs.add((a, o))

    pair_labels = []
    hard_neg_mask = []

    for asp_span in pruned_asp_indices:
        for opn_span in pruned_opn_indices:
            # Skip NULL×NULL
            if asp_span == (-1, -1) and opn_span == (-1, -1):
                continue

            pair = (asp_span, opn_span)

            if pair in gold_pairs:
                pair_labels.append(1)
                hard_neg_mask.append(False)
            else:
                pair_labels.append(0)
                # Check if this is a hard negative
                is_hard = (
                    (asp_span in gold_asp and opn_span not in gold_opn) or
                    (asp_span not in gold_asp and opn_span in gold_opn) or
                    _is_near_boundary(asp_span, gold_asp) or
                    _is_near_boundary(opn_span, gold_opn)
                )
                hard_neg_mask.append(is_hard)

    return pair_labels, hard_neg_mask


def _is_near_boundary(span, gold_spans, tolerance=2):
    """Check if a span is within tolerance tokens of any gold span boundary."""
    s, e = span
    if s == -1:
        return False
    for gs, ge in gold_spans:
        if gs == -1:
            continue
        if abs(s - gs) <= tolerance and abs(e - ge) <= tolerance:
            if (s, e) != (gs, ge):  # not exact match
                return True
    return False
