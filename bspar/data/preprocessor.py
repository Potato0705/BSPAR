"""Raw ASQP data → unified Example format converter.

Supports the standard ASQP format:
    sentence####(aspect, category, sentiment, opinion);(aspect, category, sentiment, opinion);...

Where aspect/opinion can be "NULL" for implicit elements.
"""

import re
from pathlib import Path

from .schema import Span, Quad, Example


# Standard ASQP categories and sentiments
ASQP_CATEGORIES = [
    "food quality", "food prices", "food style_options",
    "restaurant general", "restaurant prices", "restaurant miscellaneous",
    "service general",
    "ambience general",
    "location general",
    "drinks quality", "drinks prices", "drinks style_options",
]

ASQP_SENTIMENTS = ["positive", "negative", "neutral"]

SENTIMENT_MAP = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
SENTIMENT_TO_ID = {"POS": 0, "NEG": 1, "NEU": 2}
ID_TO_SENTIMENT = {0: "POS", 1: "NEG", 2: "NEU"}


def build_category_map(categories: list[str]) -> tuple[dict, dict]:
    """Build category name → id and id → name mappings."""
    cat_to_id = {c: i for i, c in enumerate(categories)}
    id_to_cat = {i: c for i, c in enumerate(categories)}
    return cat_to_id, id_to_cat


def parse_asqp_line(line: str) -> tuple[str, list[tuple]]:
    """Parse a single ASQP format line.

    Format: sentence####(aspect, category, sentiment, opinion);...

    Returns:
        text: the input sentence
        raw_quads: list of (aspect_text, category, sentiment, opinion_text)
    """
    parts = line.strip().split("####")
    if len(parts) != 2:
        raise ValueError(f"Invalid ASQP line format: {line}")

    text = parts[0].strip()
    quads_str = parts[1].strip()

    raw_quads = []
    # Match (aspect, category, sentiment, opinion)
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, quads_str)

    for match in matches:
        fields = [f.strip() for f in match.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Invalid quad format: ({match})")
        aspect_text, category, sentiment, opinion_text = fields
        raw_quads.append((aspect_text, category, sentiment, opinion_text))

    return text, raw_quads


def find_span_in_text(text_tokens: list[str], target: str) -> tuple[int, int]:
    """Find the token-level start and end indices of target in text_tokens.

    Uses case-insensitive matching, tries exact multi-token match first.

    Returns:
        (start, end) inclusive indices, or (-1, -1) if not found.
    """
    if target.upper() == "NULL":
        return (-1, -1)

    target_tokens = target.lower().split()
    text_lower = [t.lower() for t in text_tokens]

    for i in range(len(text_lower) - len(target_tokens) + 1):
        if text_lower[i:i + len(target_tokens)] == target_tokens:
            return (i, i + len(target_tokens) - 1)

    # Fallback: try substring matching
    text_joined = " ".join(text_lower)
    target_joined = " ".join(target_tokens)
    idx = text_joined.find(target_joined)
    if idx >= 0:
        # Convert character index to token index
        char_count = 0
        start_tok = -1
        for ti, tok in enumerate(text_lower):
            if char_count == idx:
                start_tok = ti
            char_count += len(tok) + 1  # +1 for space
        if start_tok >= 0:
            return (start_tok, start_tok + len(target_tokens) - 1)

    return (-1, -1)


def load_asqp_file(filepath: str, categories: list[str] = None) -> list[Example]:
    """Load an ASQP format file into a list of Example objects.

    Args:
        filepath: path to .txt file
        categories: list of valid categories (for validation)

    Returns:
        list of Example objects with gold quads
    """
    if categories is None:
        categories = ASQP_CATEGORIES

    examples = []
    path = Path(filepath)

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                text, raw_quads = parse_asqp_line(line)
            except ValueError as e:
                print(f"Warning: skipping line {line_idx}: {e}")
                continue

            tokens = text.split()
            token_offsets = []
            pos = 0
            for tok in tokens:
                start = text.index(tok, pos)
                end = start + len(tok)
                token_offsets.append((start, end))
                pos = end

            quads = []
            for asp_text, category, sentiment, opn_text in raw_quads:
                # Find aspect span
                asp_start, asp_end = find_span_in_text(tokens, asp_text)
                if asp_text.upper() == "NULL":
                    aspect = Span.null("aspect")
                elif asp_start >= 0:
                    aspect = Span(start=asp_start, end=asp_end, text=asp_text)
                else:
                    # Span not found — skip this quad
                    print(f"Warning: aspect '{asp_text}' not found in '{text}'")
                    continue

                # Find opinion span
                opn_start, opn_end = find_span_in_text(tokens, opn_text)
                if opn_text.upper() == "NULL":
                    opinion = Span.null("opinion")
                elif opn_start >= 0:
                    opinion = Span(start=opn_start, end=opn_end, text=opn_text)
                else:
                    print(f"Warning: opinion '{opn_text}' not found in '{text}'")
                    continue

                # Normalize sentiment
                sent_normalized = SENTIMENT_MAP.get(sentiment.lower(), "NEU")

                quads.append(Quad(
                    aspect=aspect,
                    opinion=opinion,
                    category=category,
                    sentiment=sent_normalized,
                ))

            example = Example(
                id=f"{path.stem}_{line_idx}",
                text=text,
                tokens=tokens,
                token_offsets=token_offsets,
                quads=quads,
            )
            examples.append(example)

    return examples
