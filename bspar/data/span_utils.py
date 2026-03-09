"""Span utility functions: enumeration, distance bucketing, order computation."""

import torch


def enumerate_spans(seq_len: int, max_span_length: int) -> list[tuple[int, int]]:
    """Generate all valid (start, end) pairs."""
    spans = []
    for i in range(seq_len):
        for j in range(i, min(i + max_span_length, seq_len)):
            spans.append((i, j))
    return spans


def compute_distance_bucket(asp_start, asp_end, opn_start, opn_end,
                            num_buckets: int = 16) -> int:
    """Compute bucketed distance between aspect and opinion spans.

    Distance = minimum token gap between two spans.
    Bucketed logarithmically for robustness.

    Returns bucket index in [0, num_buckets).
    """
    if asp_start == -1 or opn_start == -1:
        return num_buckets - 1  # special bucket for NULL involvement

    if asp_end < opn_start:
        dist = opn_start - asp_end - 1
    elif opn_end < asp_start:
        dist = asp_start - opn_end - 1
    else:
        dist = 0  # overlapping spans

    # Logarithmic bucketing: 0, 1, 2, 3, 4-5, 6-8, 9-13, 14-20, 21+
    bucket_boundaries = [0, 1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200, 500, 1000]
    for b, boundary in enumerate(bucket_boundaries):
        if dist <= boundary:
            return min(b, num_buckets - 2)
    return num_buckets - 2


def compute_order(asp_start, asp_end, opn_start, opn_end) -> int:
    """Compute ordering between aspect and opinion.

    Returns:
        0: aspect before opinion
        1: opinion before aspect
        2: involves NULL
    """
    if asp_start == -1 or opn_start == -1:
        return 2  # NULL involvement
    if asp_start <= opn_start:
        return 0
    return 1


def assign_span_labels(span_indices, gold_quads):
    """Assign binary labels for aspect/opinion span identification.

    Args:
        span_indices: list of (start, end) tuples
        gold_quads: list of Quad objects

    Returns:
        asp_labels: list of 0/1
        opn_labels: list of 0/1
    """
    gold_asp_spans = set()
    gold_opn_spans = set()
    for q in gold_quads:
        if not q.aspect.is_null:
            gold_asp_spans.add((q.aspect.start, q.aspect.end))
        if not q.opinion.is_null:
            gold_opn_spans.add((q.opinion.start, q.opinion.end))

    asp_labels = [1 if s in gold_asp_spans else 0 for s in span_indices]
    opn_labels = [1 if s in gold_opn_spans else 0 for s in span_indices]

    return asp_labels, opn_labels
