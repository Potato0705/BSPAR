"""Shared utilities for the classical trainable BSPAR baseline."""

from __future__ import annotations

import ast
import json
import pickle
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bspar.data.schema import Quad, Span  # noqa: E402
from bspar.evaluation.metrics import category_accuracy, pair_f1, quad_f1, span_f1  # noqa: E402


@dataclass
class GoldQuad:
    asp: tuple[int, int]  # inclusive, (-1,-1) for NULL
    opn: tuple[int, int]  # inclusive, (-1,-1) for NULL
    category: str
    sentiment: str


@dataclass
class Sample:
    text: str
    tokens: list[str]
    quads: list[GoldQuad]


@dataclass
class LoadStats:
    samples: int = 0
    quads_raw: int = 0
    quads_kept: int = 0
    dropped_unmatched_spans: int = 0
    format_name: str = ""


def parse_span_exclusive_to_inclusive(raw: str) -> tuple[int, int]:
    start_raw, end_raw = raw.split(",")
    start = int(start_raw)
    end_exclusive = int(end_raw)
    if start == -1 and end_exclusive == -1:
        return (-1, -1)
    return (start, end_exclusive - 1)


def span_text(tokens: list[str], span: tuple[int, int]) -> str:
    if span == (-1, -1):
        return "NULL"
    s, e = span
    if s < 0 or e >= len(tokens) or s > e:
        return ""
    return " ".join(tokens[s : e + 1])


def find_phrase_span(tokens: list[str], phrase: str) -> tuple[int, int] | None:
    phrase = phrase.strip()
    if phrase.upper() == "NULL":
        return (-1, -1)

    phrase_tokens = phrase.split()
    if not phrase_tokens:
        return None

    tokens_lower = [t.lower() for t in tokens]
    phrase_lower = [t.lower() for t in phrase_tokens]
    m = len(phrase_lower)
    n = len(tokens_lower)
    if m > n:
        return None

    for i in range(n - m + 1):
        if tokens_lower[i : i + m] == phrase_lower:
            return (i, i + m - 1)
    return None


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "####" in line:
                return "asqp_txt"
            return "acos_tsv"
    return "unknown"


def load_acos_tsv(path: Path) -> tuple[list[Sample], LoadStats]:
    stats = LoadStats(format_name="acos_tsv")
    samples: list[Sample] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            text = parts[0].strip()
            tokens = text.split()
            quads: list[GoldQuad] = []

            for field in parts[1:]:
                field = field.strip()
                if not field:
                    continue
                items = field.split()
                if len(items) != 4:
                    continue
                stats.quads_raw += 1
                asp_raw, category, sentiment, opn_raw = items
                asp = parse_span_exclusive_to_inclusive(asp_raw)
                opn = parse_span_exclusive_to_inclusive(opn_raw)
                quads.append(
                    GoldQuad(
                        asp=asp,
                        opn=opn,
                        category=category.strip(),
                        sentiment=sentiment.strip(),
                    )
                )
                stats.quads_kept += 1

            samples.append(Sample(text=text, tokens=tokens, quads=quads))
            stats.samples += 1

    return samples, stats


def load_asqp_txt(path: Path) -> tuple[list[Sample], LoadStats]:
    stats = LoadStats(format_name="asqp_txt")
    samples: list[Sample] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "####" not in line:
                continue
            text, rhs = line.split("####", 1)
            text = text.strip()
            tokens = text.split()

            try:
                annotations = ast.literal_eval(rhs.strip())
            except (ValueError, SyntaxError):
                annotations = []

            quads: list[GoldQuad] = []
            for item in annotations:
                if not isinstance(item, list) or len(item) != 4:
                    continue
                aspect_txt, category, sentiment, opinion_txt = item
                stats.quads_raw += 1
                asp = find_phrase_span(tokens, str(aspect_txt))
                opn = find_phrase_span(tokens, str(opinion_txt))

                if asp is None or opn is None:
                    stats.dropped_unmatched_spans += 1
                    continue

                quads.append(
                    GoldQuad(
                        asp=asp,
                        opn=opn,
                        category=str(category).strip(),
                        sentiment=str(sentiment).strip().lower(),
                    )
                )
                stats.quads_kept += 1

            samples.append(Sample(text=text, tokens=tokens, quads=quads))
            stats.samples += 1

    return samples, stats


def load_dataset(path: Path) -> tuple[list[Sample], LoadStats]:
    fmt = detect_format(path)
    if fmt == "acos_tsv":
        return load_acos_tsv(path)
    if fmt == "asqp_txt":
        return load_asqp_txt(path)
    raise ValueError(f"Unsupported or empty dataset format: {path}")


def is_punctuation_token(token: str) -> bool:
    return all(ch in string.punctuation for ch in token)


def build_phrase_lexicons(samples: list[Sample]) -> tuple[set[str], set[str]]:
    asp_lex: set[str] = set()
    opn_lex: set[str] = set()
    for sample in samples:
        tokens_lower = [t.lower() for t in sample.tokens]
        for q in sample.quads:
            if q.asp != (-1, -1):
                s, e = q.asp
                asp_lex.add(" ".join(tokens_lower[s : e + 1]))
            if q.opn != (-1, -1):
                s, e = q.opn
                opn_lex.add(" ".join(tokens_lower[s : e + 1]))
    return asp_lex, opn_lex


def extract_candidate_spans(
    tokens: list[str],
    phrase_lexicon: set[str],
    max_span_len: int,
) -> list[tuple[int, int]]:
    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)
    candidates: set[tuple[int, int]] = set()

    for i in range(n):
        for length in range(1, max_span_len + 1):
            j = i + length
            if j > n:
                break
            phrase = " ".join(tokens_lower[i:j])
            if phrase in phrase_lexicon:
                candidates.add((i, j - 1))

    if not candidates:
        for i, token in enumerate(tokens):
            if not is_punctuation_token(token):
                candidates.add((i, i))

    candidates.add((-1, -1))
    return sorted(candidates)


def pair_distance_bucket(asp: tuple[int, int], opn: tuple[int, int]) -> str:
    if asp == (-1, -1) or opn == (-1, -1):
        return "NULL"
    a_s, a_e = asp
    o_s, o_e = opn
    if a_e < o_s:
        dist = o_s - a_e - 1
    elif o_e < a_s:
        dist = a_s - o_e - 1
    else:
        dist = 0
    if dist == 0:
        return "0"
    if dist <= 2:
        return "1-2"
    if dist <= 5:
        return "3-5"
    return "6+"


def pair_order(asp: tuple[int, int], opn: tuple[int, int]) -> str:
    if asp == (-1, -1) or opn == (-1, -1):
        return "NULL"
    return "A_FIRST" if asp[0] <= opn[0] else "O_FIRST"


def build_pair_feature_text(
    sent_text: str,
    tokens: list[str],
    asp: tuple[int, int],
    opn: tuple[int, int],
) -> str:
    asp_txt = span_text(tokens, asp).lower()
    opn_txt = span_text(tokens, opn).lower()
    dist = pair_distance_bucket(asp, opn)
    order = pair_order(asp, opn)
    asp_len = 0 if asp == (-1, -1) else asp[1] - asp[0] + 1
    opn_len = 0 if opn == (-1, -1) else opn[1] - opn[0] + 1
    return (
        f"{sent_text.lower()} [ASP] {asp_txt} [OPN] {opn_txt} "
        f"[DIST] {dist} [ORDER] {order} [ALEN] {asp_len} [OLEN] {opn_len}"
    )


def class_vocab(samples: list[Sample]) -> list[tuple[str, str]]:
    labels: set[tuple[str, str]] = set()
    for sample in samples:
        for q in sample.quads:
            labels.add((q.category, q.sentiment))
    return sorted(labels)


def sample_gold_pair_labels(
    quads: list[GoldQuad], class_to_idx: dict[tuple[str, str], int]
) -> dict[tuple[int, int, int, int], set[int]]:
    mapping: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    for q in quads:
        key = (q.asp[0], q.asp[1], q.opn[0], q.opn[1])
        mapping[key].add(class_to_idx[(q.category, q.sentiment)])
    return mapping


def build_train_instances(
    samples: list[Sample],
    asp_lex: set[str],
    opn_lex: set[str],
    classes: list[tuple[str, str]],
    max_span_len: int,
) -> tuple[list[str], np.ndarray]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    feats: list[str] = []
    y_rows: list[np.ndarray] = []

    for sample in samples:
        asp_cands = extract_candidate_spans(sample.tokens, asp_lex, max_span_len)
        opn_cands = extract_candidate_spans(sample.tokens, opn_lex, max_span_len)
        gold_map = sample_gold_pair_labels(sample.quads, class_to_idx)

        for asp in asp_cands:
            for opn in opn_cands:
                if asp == (-1, -1) and opn == (-1, -1):
                    continue
                if asp != (-1, -1) and opn != (-1, -1) and asp == opn:
                    continue

                feats.append(build_pair_feature_text(sample.text, sample.tokens, asp, opn))
                row = np.zeros(len(classes), dtype=np.int8)
                key = (asp[0], asp[1], opn[0], opn[1])
                for class_idx in gold_map.get(key, set()):
                    row[class_idx] = 1
                y_rows.append(row)

    if not y_rows:
        return feats, np.zeros((0, len(classes)), dtype=np.int8)
    return feats, np.vstack(y_rows)


def train_model(
    train_features: list[str],
    train_targets: np.ndarray,
    max_iter: int,
) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_features)

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=max_iter,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )
    )
    clf.fit(x_train, train_targets)
    return vectorizer, clf


def decode_samples(
    samples: list[Sample],
    vectorizer: TfidfVectorizer,
    clf: OneVsRestClassifier,
    asp_lex: set[str],
    opn_lex: set[str],
    classes: list[tuple[str, str]],
    max_span_len: int,
    threshold: float,
    top_n_per_sent: int,
) -> list[list[Quad]]:
    all_preds: list[list[Quad]] = []

    for sample in samples:
        asp_cands = extract_candidate_spans(sample.tokens, asp_lex, max_span_len)
        opn_cands = extract_candidate_spans(sample.tokens, opn_lex, max_span_len)

        pair_list: list[tuple[tuple[int, int], tuple[int, int]]] = []
        pair_feats: list[str] = []

        for asp in asp_cands:
            for opn in opn_cands:
                if asp == (-1, -1) and opn == (-1, -1):
                    continue
                if asp != (-1, -1) and opn != (-1, -1) and asp == opn:
                    continue
                pair_list.append((asp, opn))
                pair_feats.append(build_pair_feature_text(sample.text, sample.tokens, asp, opn))

        if not pair_feats:
            all_preds.append([])
            continue

        x = vectorizer.transform(pair_feats)
        proba = clf.predict_proba(x)

        scored: dict[tuple[int, int, int, int, str, str], float] = {}
        max_idx = np.unravel_index(np.argmax(proba), proba.shape)
        fallback_pair = pair_list[max_idx[0]]
        fallback_cls = classes[max_idx[1]]
        fallback_key = (
            fallback_pair[0][0],
            fallback_pair[0][1],
            fallback_pair[1][0],
            fallback_pair[1][1],
            fallback_cls[0],
            fallback_cls[1],
        )
        scored[fallback_key] = float(proba[max_idx])

        for pair_idx, (asp, opn) in enumerate(pair_list):
            for class_idx, (category, sentiment) in enumerate(classes):
                p = float(proba[pair_idx, class_idx])
                if p < threshold:
                    continue
                key = (asp[0], asp[1], opn[0], opn[1], category, sentiment)
                if key not in scored or p > scored[key]:
                    scored[key] = p

        sorted_items = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_n_per_sent]
        sentence_preds: list[Quad] = []
        for key, _ in sorted_items:
            a_s, a_e, o_s, o_e, category, sentiment = key

            asp_span = (
                Span.null("aspect")
                if (a_s, a_e) == (-1, -1)
                else Span(start=a_s, end=a_e, text=span_text(sample.tokens, (a_s, a_e)), is_null=False)
            )
            opn_span = (
                Span.null("opinion")
                if (o_s, o_e) == (-1, -1)
                else Span(start=o_s, end=o_e, text=span_text(sample.tokens, (o_s, o_e)), is_null=False)
            )

            sentence_preds.append(
                Quad(
                    aspect=asp_span,
                    opinion=opn_span,
                    category=category,
                    sentiment=sentiment,
                )
            )

        all_preds.append(sentence_preds)

    return all_preds


def golds_to_eval_quads(samples: list[Sample]) -> list[list[Quad]]:
    result: list[list[Quad]] = []
    for sample in samples:
        quads: list[Quad] = []
        for q in sample.quads:
            asp = (
                Span.null("aspect")
                if q.asp == (-1, -1)
                else Span(start=q.asp[0], end=q.asp[1], text=span_text(sample.tokens, q.asp), is_null=False)
            )
            opn = (
                Span.null("opinion")
                if q.opn == (-1, -1)
                else Span(start=q.opn[0], end=q.opn[1], text=span_text(sample.tokens, q.opn), is_null=False)
            )
            quads.append(Quad(aspect=asp, opinion=opn, category=q.category, sentiment=q.sentiment))
        result.append(quads)
    return result


def compute_metrics(predictions: list[list[Quad]], golds: list[list[Quad]]) -> dict[str, Any]:
    q = quad_f1(predictions, golds, match_affective=True)
    p = pair_f1(predictions, golds)
    a = span_f1(predictions, golds, role="aspect")
    o = span_f1(predictions, golds, role="opinion")
    c = category_accuracy(predictions, golds)
    return {
        "quad": q,
        "pair": p,
        "aspect": a,
        "opinion": o,
        "category": c,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    q = metrics["quad"]
    p = metrics["pair"]
    a = metrics["aspect"]
    o = metrics["opinion"]
    c = metrics["category"]
    print("== Metrics ==")
    print(
        f"Quad  P/R/F1: {q['precision']:.4f} / {q['recall']:.4f} / {q['f1']:.4f} "
        f"(tp={q['tp']}, pred={q['total_pred']}, gold={q['total_gold']})"
    )
    print(f"Pair  P/R/F1: {p['precision']:.4f} / {p['recall']:.4f} / {p['f1']:.4f}")
    print(f"Asp   P/R/F1: {a['precision']:.4f} / {a['recall']:.4f} / {a['f1']:.4f}")
    print(f"Opn   P/R/F1: {o['precision']:.4f} / {o['recall']:.4f} / {o['f1']:.4f}")
    print(f"Cat   Acc:    {c['accuracy']:.4f} (correct={c['correct']}, total={c['total']})")


def print_load_stats(name: str, stats: LoadStats) -> None:
    print(f"[{name}] format={stats.format_name}, samples={stats.samples}, quads={stats.quads_kept}/{stats.quads_raw}")
    if stats.dropped_unmatched_spans > 0:
        print(f"[{name}] dropped unmatched explicit spans: {stats.dropped_unmatched_spans}")


def save_model_artifacts(
    save_dir: Path,
    vectorizer: TfidfVectorizer,
    clf: OneVsRestClassifier,
    classes: list[tuple[str, str]],
    asp_lex: set[str],
    opn_lex: set[str],
    max_span_len: int,
    threshold: float,
    top_n_per_sent: int,
    extra_config: dict[str, Any] | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": classes,
        "aspect_lexicon": sorted(asp_lex),
        "opinion_lexicon": sorted(opn_lex),
        "max_span_len": max_span_len,
        "threshold": threshold,
        "top_n_per_sent": top_n_per_sent,
    }
    if extra_config:
        payload["extra_config"] = extra_config

    with (save_dir / "model.pkl").open("wb") as f:
        pickle.dump(payload, f)

    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "max_span_len": max_span_len,
                "threshold": threshold,
                "top_n_per_sent": top_n_per_sent,
                "num_classes": len(classes),
                "num_aspect_lexicon": len(asp_lex),
                "num_opinion_lexicon": len(opn_lex),
                "extra_config": extra_config or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_model_artifacts(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "model.pkl").open("rb") as f:
        return pickle.load(f)


def quads_to_jsonable(quads: list[Quad]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in quads:
        out.append(
            {
                "aspect": {
                    "start": q.aspect.start,
                    "end": q.aspect.end,
                    "text": q.aspect.text,
                    "is_null": q.aspect.is_null,
                },
                "opinion": {
                    "start": q.opinion.start,
                    "end": q.opinion.end,
                    "text": q.opinion.text,
                    "is_null": q.opinion.is_null,
                },
                "category": q.category,
                "sentiment": q.sentiment,
            }
        )
    return out
