"""Final quad output formatting: recover text spans and format labels."""

from ..data.schema import Span, Quad


def format_predictions(selected_quads, tokens, category_map,
                       sentiment_map=None, task_type="asqp"):
    """Convert decoded quads to final Quad objects with text.

    Args:
        selected_quads: list of quad dicts from NMS
        tokens: list of token strings
        category_map: dict mapping category_id → category_name
        sentiment_map: dict mapping sentiment_id → sentiment_name (ASQP)
        task_type: "asqp" or "dimabsa"

    Returns:
        list of Quad objects
    """
    results = []

    for quad in selected_quads:
        # Aspect span
        asp_span = quad["asp_span"]
        if asp_span == (-1, -1):
            aspect = Span.null("aspect")
        else:
            start, end = asp_span
            text = " ".join(tokens[start:end + 1])
            aspect = Span(start=start, end=end, text=text)

        # Opinion span
        opn_span = quad["opn_span"]
        if opn_span == (-1, -1):
            opinion = Span.null("opinion")
        else:
            start, end = opn_span
            text = " ".join(tokens[start:end + 1])
            opinion = Span(start=start, end=end, text=text)

        # Category
        category = category_map.get(quad["category_id"], "UNKNOWN")

        # Affective
        if task_type == "asqp":
            sentiment = sentiment_map.get(quad["affective"], "NEU")
            result = Quad(aspect=aspect, opinion=opinion,
                          category=category, sentiment=sentiment)
        else:
            v, ar = quad["affective"]
            v = max(1.0, min(5.0, v))    # clip to valid range
            ar = max(1.0, min(5.0, ar))
            result = Quad(aspect=aspect, opinion=opinion,
                          category=category, valence=v, arousal=ar)

        results.append(result)

    return results
