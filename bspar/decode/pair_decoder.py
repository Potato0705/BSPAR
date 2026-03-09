"""Pair candidate construction from pruned aspect/opinion spans."""


def construct_pair_candidates(pruned_aspects, pruned_opinions,
                              null_asp_repr, null_opn_repr):
    """Build candidate pairs from pruned spans + NULL prototypes.

    Rules:
    1. Cartesian product: every aspect × every opinion
    2. Include NULL_asp × explicit_opn and explicit_asp × NULL_opn
    3. Exclude NULL_asp × NULL_opn

    Args:
        pruned_aspects: list of dicts with span, repr, score
        pruned_opinions: list of dicts with span, repr, score
        null_asp_repr: tensor (span_repr_size,)
        null_opn_repr: tensor (span_repr_size,)

    Returns:
        list of pair candidate dicts
    """
    candidates = []

    # Explicit × Explicit
    for asp in pruned_aspects:
        for opn in pruned_opinions:
            # Skip self-pairing (same span as both aspect and opinion)
            if asp["span"] == opn["span"]:
                continue
            candidates.append({
                "asp_span": asp["span"],
                "opn_span": opn["span"],
                "asp_repr": asp["repr"],
                "opn_repr": opn["repr"],
                "asp_score": asp["score"],
                "opn_score": opn["score"],
                "has_null_asp": False,
                "has_null_opn": False,
            })

    # NULL_asp × Explicit_opn
    for opn in pruned_opinions:
        candidates.append({
            "asp_span": (-1, -1),
            "opn_span": opn["span"],
            "asp_repr": null_asp_repr,
            "opn_repr": opn["repr"],
            "asp_score": 0.0,
            "opn_score": opn["score"],
            "has_null_asp": True,
            "has_null_opn": False,
        })

    # Explicit_asp × NULL_opn
    for asp in pruned_aspects:
        candidates.append({
            "asp_span": asp["span"],
            "opn_span": (-1, -1),
            "asp_repr": asp["repr"],
            "opn_repr": null_opn_repr,
            "asp_score": asp["score"],
            "opn_score": 0.0,
            "has_null_asp": False,
            "has_null_opn": True,
        })

    # NULL × NULL is excluded

    return candidates
