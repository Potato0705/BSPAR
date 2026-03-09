"""Run simple reproducible baselines for BSPAR ACOS-style datasets.

Current baseline options:
1) copy: exact-text retrieval from train set, else predict empty.
2) copy_majority: exact-text retrieval, else predict one majority implicit quad.
3) nn_jaccard: nearest-neighbor retrieval by token-set Jaccard.
4) empty: always predict empty.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bspar.data.schema import Quad, Span
from bspar.evaluation.metrics import category_accuracy, pair_f1, quad_f1, span_f1


@dataclass
class Sample:
    text: str
    tokens: list[str]
    quads: list[Quad]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BSPAR quick baselines.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("bspar/data/Restaurant-ACOS/rest16_quad_train.tsv"),
        help="Path to training tsv.",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("bspar/data/Restaurant-ACOS/rest16_quad_dev.tsv"),
        help="Path to dev tsv.",
    )
    parser.add_argument(
        "--method",
        choices=["copy", "copy_majority", "nn_jaccard", "empty"],
        default="copy",
        help="Baseline method.",
    )
    return parser.parse_args()


def parse_span(span_token: str, tokens: list[str], role: str) -> Span:
    start_raw, end_raw = span_token.split(",")
    start = int(start_raw)
    end_exclusive = int(end_raw)

    if start == -1 and end_exclusive == -1:
        return Span.null(role)

    # ACOS tsv uses [start, end) token indexing.
    end_inclusive = end_exclusive - 1
    if start < 0 or end_exclusive > len(tokens) or start >= end_exclusive:
        # Keep malformed spans recoverable while preserving indices.
        return Span(start=start, end=end_inclusive, text="")

    text = " ".join(tokens[start:end_exclusive])
    return Span(start=start, end=end_inclusive, text=text)


def load_acos_tsv(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            text = parts[0]
            tokens = text.split()
            quads: list[Quad] = []

            for field in parts[1:]:
                field = field.strip()
                if not field:
                    continue
                items = field.split()
                if len(items) != 4:
                    continue

                asp_tok, category, sentiment_id, opn_tok = items
                aspect = parse_span(asp_tok, tokens, "aspect")
                opinion = parse_span(opn_tok, tokens, "opinion")
                quads.append(
                    Quad(
                        aspect=aspect,
                        opinion=opinion,
                        category=category,
                        sentiment=sentiment_id,
                    )
                )

            samples.append(Sample(text=text, tokens=tokens, quads=quads))
    return samples


def quad_key(quad: Quad) -> tuple[int, int, int, int, str, str | None]:
    return (
        quad.aspect.start,
        quad.aspect.end,
        quad.opinion.start,
        quad.opinion.end,
        quad.category,
        quad.sentiment,
    )


def canonicalize_quads(quads: list[Quad]) -> tuple[tuple[int, int, int, int, str, str | None], ...]:
    return tuple(sorted(quad_key(q) for q in quads))


def build_copy_index(train_samples: list[Sample]) -> dict[str, list[Quad]]:
    by_text: dict[str, list[list[Quad]]] = defaultdict(list)
    for sample in train_samples:
        by_text[sample.text].append(sample.quads)

    index: dict[str, list[Quad]] = {}
    for text, quad_lists in by_text.items():
        voted = Counter(canonicalize_quads(qs) for qs in quad_lists).most_common(1)[0][0]
        best_quads = [
            Quad(
                aspect=Span(start=k[0], end=k[1], text="", is_null=(k[0] == -1)),
                opinion=Span(start=k[2], end=k[3], text="", is_null=(k[2] == -1)),
                category=k[4],
                sentiment=k[5],
            )
            for k in voted
        ]
        index[text] = best_quads
    return index


def most_common_implicit_quad(train_samples: list[Sample]) -> Quad:
    counter: Counter[tuple[str, str | None]] = Counter()
    for sample in train_samples:
        for q in sample.quads:
            counter[(q.category, q.sentiment)] += 1

    (category, sentiment), _ = counter.most_common(1)[0]
    return Quad(
        aspect=Span.null("aspect"),
        opinion=Span.null("opinion"),
        category=category,
        sentiment=sentiment,
    )


def predict(
    train_samples: list[Sample],
    dev_samples: list[Sample],
    method: str,
    copy_index: dict[str, list[Quad]],
    fallback_quad: Quad,
) -> tuple[list[list[Quad]], int]:
    predictions: list[list[Quad]] = []
    copied = 0

    for sample in dev_samples:
        if method != "empty" and sample.text in copy_index:
            pred = copy_index[sample.text]
            copied += 1
        elif method == "nn_jaccard":
            pred = nearest_neighbor_quads(train_samples, sample)
        elif method == "copy_majority":
            pred = [fallback_quad]
        else:
            pred = []
        predictions.append(pred)

    return predictions, copied


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def nearest_neighbor_quads(train_samples: list[Sample], dev_sample: Sample) -> list[Quad]:
    dev_set = set(dev_sample.tokens)
    best_score = -1.0
    best_quads: list[Quad] = []
    for sample in train_samples:
        score = jaccard(dev_set, set(sample.tokens))
        if score > best_score:
            best_score = score
            best_quads = sample.quads
    return best_quads


def print_metrics(name: str, predictions: list[list[Quad]], golds: list[list[Quad]]) -> None:
    quad = quad_f1(predictions, golds, match_affective=True)
    pair = pair_f1(predictions, golds)
    asp = span_f1(predictions, golds, role="aspect")
    opn = span_f1(predictions, golds, role="opinion")
    cat = category_accuracy(predictions, golds)

    print(f"\n=== {name} ===")
    print(
        f"Quad  P/R/F1: {quad['precision']:.4f} / {quad['recall']:.4f} / {quad['f1']:.4f} "
        f"(tp={quad['tp']}, pred={quad['total_pred']}, gold={quad['total_gold']})"
    )
    print(f"Pair  P/R/F1: {pair['precision']:.4f} / {pair['recall']:.4f} / {pair['f1']:.4f}")
    print(f"Asp   P/R/F1: {asp['precision']:.4f} / {asp['recall']:.4f} / {asp['f1']:.4f}")
    print(f"Opn   P/R/F1: {opn['precision']:.4f} / {opn['recall']:.4f} / {opn['f1']:.4f}")
    print(f"Cat   Acc:    {cat['accuracy']:.4f} (correct={cat['correct']}, total={cat['total']})")


def main() -> None:
    args = parse_args()
    train_samples = load_acos_tsv(args.train)
    dev_samples = load_acos_tsv(args.dev)
    golds = [s.quads for s in dev_samples]

    copy_index = build_copy_index(train_samples)
    fallback = most_common_implicit_quad(train_samples)
    preds, copied = predict(train_samples, dev_samples, args.method, copy_index, fallback)

    print(f"Train samples: {len(train_samples)}")
    print(f"Dev samples:   {len(dev_samples)}")
    print(f"Method:        {args.method}")
    print(f"Exact copied:  {copied} / {len(dev_samples)} ({copied / max(len(dev_samples), 1):.2%})")
    print(f"Fallback quad: ({fallback.category}, {fallback.sentiment})")
    print_metrics(args.method, preds, golds)


if __name__ == "__main__":
    main()
