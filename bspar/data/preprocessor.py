"""Raw data → unified Example format converter.

Supports three input formats:
  1. ASQP-tuple: sentence####(aspect, category, sentiment, opinion);...
  2. ASQP-list:  sentence####[['aspect','category','sentiment','opinion'], ...]
  3. ACOS-tsv:   sentence\tasp_s,asp_e CAT sentiment opn_s,opn_e\t...

Where aspect/opinion can be "NULL" or "-1,-1" for implicit elements.
"""

import ast
import re
from pathlib import Path

from .schema import Span, Quad, Example


# ── Category / sentiment constants ──────────────────────────────────────────

ASQP_CATEGORIES = [
    "food quality", "food prices", "food style_options",
    "restaurant general", "restaurant prices", "restaurant miscellaneous",
    "service general",
    "ambience general",
    "location general",
    "drinks quality", "drinks prices", "drinks style_options",
]

ACOS_LAPTOP_CATEGORIES = [
    "LAPTOP#GENERAL", "LAPTOP#QUALITY", "LAPTOP#DESIGN_FEATURES",
    "LAPTOP#OPERATION_PERFORMANCE", "LAPTOP#USABILITY", "LAPTOP#CONNECTIVITY",
    "LAPTOP#PORTABILITY", "LAPTOP#MISCELLANEOUS", "LAPTOP#PRICE",
    "SUPPORT#GENERAL", "SUPPORT#QUALITY", "SUPPORT#DESIGN_FEATURES",
    "SUPPORT#OPERATION_PERFORMANCE", "SUPPORT#PRICE",
    "OS#GENERAL", "OS#QUALITY", "OS#DESIGN_FEATURES",
    "OS#OPERATION_PERFORMANCE", "OS#USABILITY", "OS#MISCELLANEOUS",
    "DISPLAY#GENERAL", "DISPLAY#QUALITY", "DISPLAY#DESIGN_FEATURES",
    "DISPLAY#OPERATION_PERFORMANCE", "DISPLAY#USABILITY",
    "CPU#GENERAL", "CPU#QUALITY", "CPU#DESIGN_FEATURES",
    "CPU#OPERATION_PERFORMANCE",
    "MEMORY#GENERAL", "MEMORY#QUALITY", "MEMORY#DESIGN_FEATURES",
    "MEMORY#OPERATION_PERFORMANCE",
    "HARD_DISC#GENERAL", "HARD_DISC#QUALITY", "HARD_DISC#DESIGN_FEATURES",
    "HARD_DISC#OPERATION_PERFORMANCE", "HARD_DISC#MISCELLANEOUS",
    "BATTERY#GENERAL", "BATTERY#QUALITY", "BATTERY#DESIGN_FEATURES",
    "BATTERY#OPERATION_PERFORMANCE",
    "POWER_SUPPLY#GENERAL", "POWER_SUPPLY#QUALITY",
    "POWER_SUPPLY#DESIGN_FEATURES", "POWER_SUPPLY#CONNECTIVITY",
    "KEYBOARD#GENERAL", "KEYBOARD#QUALITY", "KEYBOARD#DESIGN_FEATURES",
    "KEYBOARD#OPERATION_PERFORMANCE", "KEYBOARD#USABILITY",
    "MOUSE#GENERAL", "MOUSE#QUALITY", "MOUSE#DESIGN_FEATURES",
    "MOUSE#OPERATION_PERFORMANCE", "MOUSE#USABILITY",
    "FANS_COOLING#GENERAL", "FANS_COOLING#QUALITY",
    "FANS_COOLING#DESIGN_FEATURES", "FANS_COOLING#OPERATION_PERFORMANCE",
    "OPTICAL_DRIVES#GENERAL", "OPTICAL_DRIVES#QUALITY",
    "OPTICAL_DRIVES#DESIGN_FEATURES", "OPTICAL_DRIVES#OPERATION_PERFORMANCE",
    "PORTS#GENERAL", "PORTS#QUALITY", "PORTS#DESIGN_FEATURES",
    "PORTS#OPERATION_PERFORMANCE", "PORTS#CONNECTIVITY", "PORTS#USABILITY",
    "GRAPHICS#GENERAL", "GRAPHICS#QUALITY", "GRAPHICS#DESIGN_FEATURES",
    "GRAPHICS#OPERATION_PERFORMANCE",
    "MULTIMEDIA_DEVICES#GENERAL", "MULTIMEDIA_DEVICES#QUALITY",
    "MULTIMEDIA_DEVICES#DESIGN_FEATURES",
    "MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE",
    "MULTIMEDIA_DEVICES#CONNECTIVITY", "MULTIMEDIA_DEVICES#USABILITY",
    "HARDWARE#GENERAL", "HARDWARE#QUALITY", "HARDWARE#DESIGN_FEATURES",
    "HARDWARE#OPERATION_PERFORMANCE", "HARDWARE#USABILITY",
    "SOFTWARE#GENERAL", "SOFTWARE#QUALITY", "SOFTWARE#DESIGN_FEATURES",
    "SOFTWARE#OPERATION_PERFORMANCE", "SOFTWARE#USABILITY",
    "SOFTWARE#PORTABILITY", "SOFTWARE#PRICE",
    "SHIPPING#GENERAL", "SHIPPING#QUALITY", "SHIPPING#PRICE",
    "COMPANY#GENERAL", "COMPANY#QUALITY", "COMPANY#DESIGN_FEATURES",
    "COMPANY#OPERATION_PERFORMANCE", "COMPANY#PRICE",
    "WARRANTY#GENERAL", "WARRANTY#QUALITY",
]

ACOS_RESTAURANT_CATEGORIES = [
    "RESTAURANT#GENERAL", "RESTAURANT#PRICES", "RESTAURANT#MISCELLANEOUS",
    "FOOD#QUALITY", "FOOD#PRICES", "FOOD#STYLE_OPTIONS",
    "SERVICE#GENERAL",
    "AMBIENCE#GENERAL",
    "LOCATION#GENERAL",
    "DRINKS#QUALITY", "DRINKS#PRICES", "DRINKS#STYLE_OPTIONS",
]

ASQP_SENTIMENTS = ["positive", "negative", "neutral"]

SENTIMENT_MAP = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
SENTIMENT_TO_ID = {"POS": 0, "NEG": 1, "NEU": 2}
ID_TO_SENTIMENT = {0: "POS", 1: "NEG", 2: "NEU"}

# ACOS uses integer sentiments: 0=negative, 1=positive, 2=neutral
ACOS_SENTIMENT_MAP = {0: "NEG", 1: "POS", 2: "NEU"}


def build_category_map(categories: list[str]) -> tuple[dict, dict]:
    """Build category name → id and id → name mappings."""
    cat_to_id = {c: i for i, c in enumerate(categories)}
    id_to_cat = {i: c for i, c in enumerate(categories)}
    return cat_to_id, id_to_cat


def get_categories_for_dataset(dataset_name: str) -> list[str]:
    """Return the category list for a given dataset name."""
    mapping = {
        "asqp_rest15": ASQP_CATEGORIES,
        "asqp_rest16": ASQP_CATEGORIES,
        "acos_laptop": ACOS_LAPTOP_CATEGORIES,
        "acos_restaurant": ACOS_RESTAURANT_CATEGORIES,
    }
    if dataset_name not in mapping:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(mapping.keys())}"
        )
    return mapping[dataset_name]


# ── Span finding ────────────────────────────────────────────────────────────

def find_span_in_text(text_tokens: list[str], target: str) -> tuple[int, int]:
    """Find the token-level start and end indices of target in text_tokens.

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

    # Fallback: substring matching
    text_joined = " ".join(text_lower)
    target_joined = " ".join(target_tokens)
    idx = text_joined.find(target_joined)
    if idx >= 0:
        char_count = 0
        start_tok = -1
        for ti, tok in enumerate(text_lower):
            if char_count == idx:
                start_tok = ti
            char_count += len(tok) + 1
        if start_tok >= 0:
            return (start_tok, start_tok + len(target_tokens) - 1)

    return (-1, -1)


# ── Format parsers ──────────────────────────────────────────────────────────

def parse_asqp_tuple_line(line: str) -> tuple[str, list[tuple]]:
    """Parse ASQP tuple format: sentence####(a,c,s,o);(a,c,s,o);..."""
    parts = line.strip().split("####")
    if len(parts) != 2:
        raise ValueError(f"Invalid ASQP line: {line}")

    text = parts[0].strip()
    quads_str = parts[1].strip()

    raw_quads = []
    pattern = r'\(([^)]+)\)'
    for match in re.findall(pattern, quads_str):
        fields = [f.strip() for f in match.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Invalid quad format: ({match})")
        raw_quads.append(tuple(fields))

    return text, raw_quads


def parse_asqp_list_line(line: str) -> tuple[str, list[tuple]]:
    """Parse ASQP list format: sentence####[['a','c','s','o'], ...]"""
    parts = line.strip().split("####")
    if len(parts) != 2:
        raise ValueError(f"Invalid ASQP line: {line}")

    text = parts[0].strip()
    quads_raw = ast.literal_eval(parts[1].strip())

    raw_quads = []
    for q in quads_raw:
        if len(q) != 4:
            raise ValueError(f"Invalid quad: {q}")
        raw_quads.append(tuple(q))  # (aspect, category, sentiment, opinion)

    return text, raw_quads


def parse_acos_line(line: str) -> tuple[str, list[dict]]:
    """Parse ACOS TSV format: sentence\\tasp_s,asp_e CAT sent opn_s,opn_e\\t...

    Returns:
        text: sentence string
        raw_quads: list of dicts with asp_span, opn_span, category, sentiment_id
    """
    parts = line.strip().split("\t")
    if len(parts) < 2:
        raise ValueError(f"Invalid ACOS line: {line}")

    text = parts[0].strip()
    raw_quads = []

    for quad_str in parts[1:]:
        quad_str = quad_str.strip()
        if not quad_str:
            continue
        fields = quad_str.split()
        if len(fields) != 4:
            raise ValueError(f"Invalid ACOS quad: {quad_str}")

        asp_span_str, category, sentiment_str, opn_span_str = fields
        asp_s, asp_e = [int(x) for x in asp_span_str.split(",")]
        opn_s, opn_e = [int(x) for x in opn_span_str.split(",")]
        sentiment_id = int(sentiment_str)

        raw_quads.append({
            "asp_span": (asp_s, asp_e),
            "opn_span": (opn_s, opn_e),
            "category": category,
            "sentiment_id": sentiment_id,
        })

    return text, raw_quads


# ── Unified loaders ─────────────────────────────────────────────────────────

def _detect_format(filepath: str) -> str:
    """Auto-detect data file format from first line."""
    with open(filepath, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if "\t" in line:
        # Check if it's ACOS (tab-separated with span indices)
        parts = line.split("\t")
        if len(parts) >= 2 and "," in parts[1].split()[0]:
            return "acos"
    if "####" in line:
        after = line.split("####")[1].strip()
        if after.startswith("["):
            return "asqp_list"
        elif after.startswith("("):
            return "asqp_tuple"
    raise ValueError(f"Cannot detect format of {filepath}: {line[:100]}")


def _build_example_from_text_quads(
    text: str, raw_quads: list[tuple], line_idx: int,
    file_stem: str, categories: list[str]
) -> Example:
    """Build Example from text + list of (aspect_text, category, sentiment, opinion_text)."""
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
        # Aspect
        asp_start, asp_end = find_span_in_text(tokens, asp_text)
        if asp_text.upper() == "NULL":
            aspect = Span.null("aspect")
        elif asp_start >= 0:
            aspect = Span(start=asp_start, end=asp_end, text=asp_text)
        else:
            continue

        # Opinion
        opn_start, opn_end = find_span_in_text(tokens, opn_text)
        if opn_text.upper() == "NULL":
            opinion = Span.null("opinion")
        elif opn_start >= 0:
            opinion = Span(start=opn_start, end=opn_end, text=opn_text)
        else:
            continue

        sent_normalized = SENTIMENT_MAP.get(sentiment.lower(), "NEU")
        quads.append(Quad(
            aspect=aspect, opinion=opinion,
            category=category, sentiment=sent_normalized,
        ))

    return Example(
        id=f"{file_stem}_{line_idx}",
        text=text, tokens=tokens,
        token_offsets=token_offsets, quads=quads,
    )


def load_asqp_file(filepath: str, categories: list[str] = None) -> list[Example]:
    """Load an ASQP format file (tuple or list) into Example objects."""
    if categories is None:
        categories = ASQP_CATEGORIES

    path = Path(filepath)
    fmt = _detect_format(filepath)

    parse_fn = parse_asqp_tuple_line if fmt == "asqp_tuple" else parse_asqp_list_line

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                text, raw_quads = parse_fn(line)
            except ValueError as e:
                print(f"Warning: skipping line {line_idx}: {e}")
                continue

            ex = _build_example_from_text_quads(
                text, raw_quads, line_idx, path.stem, categories
            )
            if ex is not None:
                examples.append(ex)

    return examples


def load_acos_file(filepath: str, categories: list[str] = None) -> list[Example]:
    """Load an ACOS TSV file into Example objects."""
    path = Path(filepath)
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                text, raw_quads = parse_acos_line(line)
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
            for rq in raw_quads:
                asp_s, asp_e = rq["asp_span"]
                opn_s, opn_e = rq["opn_span"]

                if asp_s == -1 and asp_e == -1:
                    aspect = Span.null("aspect")
                else:
                    asp_text = " ".join(tokens[asp_s:asp_e])
                    aspect = Span(start=asp_s, end=asp_e - 1, text=asp_text)

                if opn_s == -1 and opn_e == -1:
                    opinion = Span.null("opinion")
                else:
                    opn_text = " ".join(tokens[opn_s:opn_e])
                    opinion = Span(start=opn_s, end=opn_e - 1, text=opn_text)

                sentiment = ACOS_SENTIMENT_MAP.get(rq["sentiment_id"], "NEU")

                quads.append(Quad(
                    aspect=aspect, opinion=opinion,
                    category=rq["category"], sentiment=sentiment,
                ))

            examples.append(Example(
                id=f"{path.stem}_{line_idx}",
                text=text, tokens=tokens,
                token_offsets=token_offsets, quads=quads,
            ))

    return examples


def load_data(filepath: str, data_format: str = "auto",
              categories: list[str] = None) -> list[Example]:
    """Unified data loader. Auto-detects format if not specified.

    Args:
        filepath: path to data file
        data_format: "auto", "asqp_tuple", "asqp_list", "acos", "asqp_txt"
        categories: valid category list

    Returns:
        list of Example objects
    """
    if data_format == "auto":
        data_format = _detect_format(filepath)

    if data_format in ("asqp_tuple", "asqp_list", "asqp_txt"):
        return load_asqp_file(filepath, categories)
    elif data_format == "acos":
        return load_acos_file(filepath, categories)
    else:
        raise ValueError(f"Unknown format: {data_format}")
