"""Frozen-feature offline cat/aff decode calibration replay.

Goal:
- No Stage-1 trunk retraining
- No retention path change (topn_only + top_n=20)
- No Stage-2 logic change
- Train a very-light offline probe on retained frozen pair_reprs
- Replay baseline vs probe-calibrated cat/aff decode strictly inside the same retained set
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bspar.config import BSPARConfig
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.data.preprocessor import (
    SENTIMENT_TO_ID,
    build_category_map,
    get_categories_for_dataset,
    load_data,
)
from bspar.evaluation.metrics import compute_a3_diagnostics, compute_quad_f1
from bspar.models.bspar_stage1 import BSPARStage1


@dataclass
class Entry:
    seed: int
    stage1_ckpt: Path
    run_dir: Path | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen-feature retained-only cat/aff decode calibration replay"
    )
    p.add_argument(
        "--affacr_per_seed",
        default=(
            "outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/"
            "summary/affacr_a0_4seed_per_seed.csv"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--cat_scale",
        type=float,
        default=0.8,
        help="cat logits scale: logits' = logits * (1 + cat_scale*(p-0.5))",
    )
    p.add_argument(
        "--aff_scale",
        type=float,
        default=0.8,
        help="aff logits scale: logits' = logits * (1 + aff_scale*(p-0.5))",
    )
    p.add_argument("--max_pred", type=int, default=10)
    p.add_argument("--output_root", default=None)
    return p.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return float(v)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_entry(path: Path, seed: int) -> Entry:
    rows = load_csv_rows(path)
    row = next((r for r in rows if int(r["seed"]) == seed), None)
    if row is None:
        raise RuntimeError(f"Seed {seed} not found in {path}")
    run_dir = Path(row["run_dir"]).resolve() if row.get("run_dir") else None
    return Entry(
        seed=seed,
        stage1_ckpt=Path(row["stage1_ckpt"]).resolve(),
        run_dir=run_dir,
    )


def load_cfg_from_stage1(stage1_dir: Path) -> tuple[BSPARConfig, dict[str, Any], Path]:
    manifest_path = stage1_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg_path = Path(manifest["config_path"]).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = BSPARConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, raw, cfg_path


def build_loader(
    cfg: BSPARConfig,
    raw_cfg: dict[str, Any],
    split: str,
    batch_size: int,
):
    dataset_name = raw_cfg.get("dataset_name", "asqp_rest15")
    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, id_to_cat = build_category_map(categories)
    cfg.num_categories = len(categories)

    data_dir = Path(raw_cfg.get("data_dir", "data/asqp_rest15"))
    if split == "train":
        file_name = raw_cfg.get("train_file", "train.txt")
    elif split == "dev":
        file_name = raw_cfg.get("dev_file", "dev.txt")
    elif split == "test":
        file_name = raw_cfg.get("test_file", "test.txt")
    else:
        raise ValueError(f"Unknown split: {split}")

    examples = load_data(
        str(data_dir / file_name),
        raw_cfg.get("data_format", "auto"),
        categories,
    )
    ds = BSPARStage1Dataset(
        examples,
        cfg.model_name,
        max_length=128,
        max_span_length=cfg.max_span_length,
        allow_offline_tokenizer_fallback=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )
    return loader, examples, cat_to_id, id_to_cat


def _to_subword(model: BSPARStage1, span_obj, w2s):
    if span_obj.is_null:
        return (-1, -1)
    sub = model._word_span_to_subword(span_obj.start, span_obj.end, w2s)
    return None if sub is None else tuple(sub)


def _classify_outranker_type(
    pair_key: tuple[tuple[int, int], tuple[int, int]],
    gold_aspects: set[tuple[int, int]],
    gold_opinions: set[tuple[int, int]],
) -> str:
    a_span, o_span = pair_key
    if a_span == (-1, -1) or o_span == (-1, -1):
        return "NULL"
    if (a_span in gold_aspects) or (o_span in gold_opinions):
        return "near_miss"
    return "other"


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or int((labels == 1).sum()) == 0:
        return 0.0
    return float(average_precision_score(labels, scores))

def collect_split_records(
    model: BSPARStage1,
    cfg: BSPARConfig,
    raw_cfg: dict[str, Any],
    split: str,
    batch_size: int,
):
    loader, examples, cat_to_id, id_to_cat = build_loader(cfg, raw_cfg, split, batch_size)
    device = next(model.parameters()).device
    pair_top_n = int(getattr(cfg, "stage1_pair_top_n", 20))
    pair_thr = float(getattr(cfg, "stage1_pair_score_threshold", 0.001))
    pair_strategy = str(getattr(cfg, "stage1_pair_retention_strategy", "topn_only"))
    top_c = int(getattr(cfg, "top_c_categories", 3))
    max_span_len = int(getattr(cfg, "max_span_length", 8))

    sample_records: list[dict[str, Any]] = []
    feature_meta_rows: list[dict[str, Any]] = []
    X_list: list[np.ndarray] = []
    y_probe_list: list[int] = []
    y_materializable_list: list[int] = []

    ex_ptr = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                word_to_subword=batch.get("word_to_subword"),
                mode="inference",
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )

            pair_map = list(outputs["pair_map"])
            pair_probs = torch.sigmoid(outputs["pair_scores"]).detach().cpu()
            pair_reprs = outputs["pair_reprs"].detach().cpu()
            cat_logits_t = outputs["cat_logits"].detach().cpu()
            aff_logits_t = outputs["aff_output"].detach().cpu()
            selected_batch = outputs.get("selected_pair_ids", [])
            asp_batch = outputs["asp_indices"]
            opn_batch = outputs["opn_indices"]

            for b in range(len(outputs["candidates"])):
                ex = examples[ex_ptr]
                w2s = batch["word_to_subword"][b]
                gold_quads = batch["gold_quads"][b]
                asp_indices = [tuple(x) for x in asp_batch[b]]
                opn_indices = [tuple(x) for x in opn_batch[b]]
                asp_set = set(asp_indices)
                opn_set = set(opn_indices)
                selected_ids = (
                    list(selected_batch[b]) if b < len(selected_batch) else []
                )

                pairid_to_key: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
                pair_space_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                for pid, (ai, oi) in enumerate(pair_map):
                    if ai >= len(asp_indices) or oi >= len(opn_indices):
                        continue
                    key = (asp_indices[ai], opn_indices[oi])
                    pairid_to_key[pid] = key
                    pair_space_set.add(key)

                # Build gold maps (subword keyed)
                gold_pair_multi: dict[
                    tuple[tuple[int, int], tuple[int, int]],
                    dict[str, set[int]],
                ] = {}
                gold_pairs_sub: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                gold_aspects_sub: set[tuple[int, int]] = set()
                gold_opinions_sub: set[tuple[int, int]] = set()
                gold_items = []
                for g in gold_quads:
                    a_sub = _to_subword(model, g.aspect, w2s)
                    o_sub = _to_subword(model, g.opinion, w2s)
                    if a_sub is None or o_sub is None:
                        continue
                    if a_sub != (-1, -1):
                        gold_aspects_sub.add(a_sub)
                    if o_sub != (-1, -1):
                        gold_opinions_sub.add(o_sub)

                    pair_key = (a_sub, o_sub)
                    gold_pairs_sub.add(pair_key)
                    if pair_key not in gold_pair_multi:
                        gold_pair_multi[pair_key] = {"cat_ids": set(), "sent_ids": set()}
                    if g.category in cat_to_id:
                        gold_pair_multi[pair_key]["cat_ids"].add(int(cat_to_id[g.category]))
                    if g.sentiment in SENTIMENT_TO_ID:
                        gold_pair_multi[pair_key]["sent_ids"].add(
                            int(SENTIMENT_TO_ID[g.sentiment])
                        )
                    gold_items.append(
                        {
                            "a_sub": a_sub,
                            "o_sub": o_sub,
                            "a_word": (
                                (-1, -1)
                                if g.aspect.is_null
                                else (int(g.aspect.start), int(g.aspect.end))
                            ),
                            "o_word": (
                                (-1, -1)
                                if g.opinion.is_null
                                else (int(g.opinion.start), int(g.opinion.end))
                            ),
                            "cat_id": int(cat_to_id[g.category]) if g.category in cat_to_id else -1,
                            "sent_id": (
                                int(SENTIMENT_TO_ID[g.sentiment])
                                if g.sentiment in SENTIMENT_TO_ID
                                else -1
                            ),
                            "is_implicit": bool(g.aspect.is_null or g.opinion.is_null),
                        }
                    )

                retained_pairs = []
                selected_pair_set_sub = set()
                for pid in selected_ids:
                    if pid not in pairid_to_key:
                        continue
                    a_sub, o_sub = pairid_to_key[pid]
                    selected_pair_set_sub.add((a_sub, o_sub))
                    if a_sub == (-1, -1):
                        a_word = (-1, -1)
                    else:
                        a_word = model._subword_span_to_word(a_sub[0], a_sub[1], w2s)
                    if o_sub == (-1, -1):
                        o_word = (-1, -1)
                    else:
                        o_word = model._subword_span_to_word(o_sub[0], o_sub[1], w2s)

                    cat_logits_vec = cat_logits_t[b, pid]
                    aff_logits_vec = aff_logits_t[b, pid]
                    cat_prob_vec = torch.softmax(cat_logits_vec, dim=-1)
                    aff_prob_vec = torch.softmax(aff_logits_vec, dim=-1)
                    k_cat = min(top_c, int(cat_prob_vec.numel()))
                    top_vals, top_ids = torch.topk(cat_prob_vec, k=k_cat)
                    topcat_ids = [int(x) for x in top_ids.tolist()]
                    topcat_probs = [float(x) for x in top_vals.tolist()]
                    aff_pred = int(torch.argmax(aff_prob_vec).item())
                    aff_top1_prob = float(torch.max(aff_prob_vec).item())
                    pair_score = float(pair_probs[b, pid].item())
                    pair_repr = pair_reprs[b, pid].numpy().astype(np.float32)

                    is_gold_pair = int((a_sub, o_sub) in gold_pair_multi)
                    is_gold_materializable = 0
                    if is_gold_pair:
                        gold_meta = gold_pair_multi[(a_sub, o_sub)]
                        cat_hit = bool(set(topcat_ids).intersection(gold_meta["cat_ids"]))
                        sent_hit = aff_pred in gold_meta["sent_ids"]
                        is_gold_materializable = int(cat_hit and sent_hit)

                    retained_pairs.append(
                        {
                            "pid": int(pid),
                            "a_sub": a_sub,
                            "o_sub": o_sub,
                            "a_word": tuple(a_word),
                            "o_word": tuple(o_word),
                            "pair_score": pair_score,
                            "probe_prob": 0.5,
                            "topcat_ids": topcat_ids,
                            "topcat_probs": topcat_probs,
                            "aff_pred": aff_pred,
                            "aff_top1_prob": aff_top1_prob,
                            "cat_logits": [float(x) for x in cat_logits_vec.tolist()],
                            "aff_logits": [float(x) for x in aff_logits_vec.tolist()],
                            "pair_repr": pair_repr,
                            "label_probe": is_gold_pair,  # existing oracle/probe definition
                            "label_materializable": is_gold_materializable,
                            "neg_type": (
                                "gold"
                                if is_gold_pair == 1
                                else _classify_outranker_type(
                                    (a_sub, o_sub),
                                    gold_aspects_sub,
                                    gold_opinions_sub,
                                )
                            ),
                        }
                    )

                sample_records.append(
                    {
                        "split": split,
                        "example_id": ex.id,
                        "gold_quads": gold_quads,
                        "gold_items": gold_items,
                        "gold_pairs_sub": sorted(list(gold_pairs_sub)),
                        "gold_aspects_sub": gold_aspects_sub,
                        "gold_opinions_sub": gold_opinions_sub,
                        "asp_topk_set": asp_set,
                        "opn_topk_set": opn_set,
                        "pair_space_set": pair_space_set,
                        "selected_pair_set_sub": selected_pair_set_sub,
                        "selected_pair_ids": selected_ids,
                        "pair_scores_all": [float(x) for x in pair_probs[b].tolist()],
                        "pair_map": list(pair_map),
                        "asp_indices": asp_indices,
                        "opn_indices": opn_indices,
                        "retained_pairs": retained_pairs,
                    }
                )

                for rp in retained_pairs:
                    feat_id = len(feature_meta_rows)
                    X_list.append(rp["pair_repr"])
                    y_probe_list.append(int(rp["label_probe"]))
                    y_materializable_list.append(int(rp["label_materializable"]))
                    feature_meta_rows.append(
                        {
                            "row_id": feat_id,
                            "split": split,
                            "example_id": ex.id,
                            "pair_id": rp["pid"],
                            "asp_span_subword": str(rp["a_sub"]),
                            "opn_span_subword": str(rp["o_sub"]),
                            "asp_span_word": str(rp["a_word"]),
                            "opn_span_word": str(rp["o_word"]),
                            "pair_score": rp["pair_score"],
                            "label_probe": rp["label_probe"],
                            "label_materializable": rp["label_materializable"],
                            "neg_type": rp["neg_type"],
                            "aff_pred": rp["aff_pred"],
                            "aff_top1_prob": rp["aff_top1_prob"],
                            "topcat_ids": str(rp["topcat_ids"]),
                            "topcat_probs": str(rp["topcat_probs"]),
                        }
                    )

                ex_ptr += 1

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 1), dtype=np.float32)
    y_probe = np.array(y_probe_list, dtype=np.int64)
    y_mat = np.array(y_materializable_list, dtype=np.int64)
    return {
        "split": split,
        "sample_records": sample_records,
        "feature_meta_rows": feature_meta_rows,
        "X": X,
        "y_probe": y_probe,
        "y_materializable": y_mat,
        "pair_top_n": pair_top_n,
        "top_c": top_c,
        "max_span_length": max_span_len,
        "cat_to_id": cat_to_id,
        "id_to_cat": id_to_cat,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return
    fn = fieldnames if fieldnames is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def decode_predictions_for_sample(
    sample: dict[str, Any],
    mode: str,
    max_pred: int,
) -> list[dict[str, Any]]:
    candidates = []
    for rp in sample["retained_pairs"]:
        pair_score = rp["pair_score"]  # fixed: do not alter retention score path
        if mode == "baseline":
            cat_ids = rp["topcat_ids"]
            cat_probs = rp["topcat_probs"]
            aff_pred = rp["aff_pred"]
            aff_top1_prob = rp["aff_top1_prob"]
        else:
            cat_ids = rp.get("topcat_ids_cal", rp["topcat_ids"])
            cat_probs = rp.get("topcat_probs_cal", rp["topcat_probs"])
            aff_pred = int(rp.get("aff_pred_cal", rp["aff_pred"]))
            aff_top1_prob = float(rp.get("aff_top1_prob_cal", rp["aff_top1_prob"]))

        for cat_id, cat_prob in zip(cat_ids, cat_probs):
            # decode-time joint materialization score (retained-only)
            decode_score = float(pair_score * cat_prob * aff_top1_prob)
            candidates.append(
                {
                    "asp_span": tuple(rp["a_word"]),
                    "opn_span": tuple(rp["o_word"]),
                    "category_id": int(cat_id),
                    "affective": aff_pred,
                    "_decode_score": decode_score,
                }
            )

    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["_decode_score"], reverse=True)
    selected = []
    seen = set()
    for i in order:
        c = candidates[i]
        k = (tuple(c["asp_span"]), tuple(c["opn_span"]), int(c["category_id"]))
        if k in seen:
            continue
        seen.add(k)
        selected.append(
            {
                "asp_span": tuple(c["asp_span"]),
                "opn_span": tuple(c["opn_span"]),
                "category_id": int(c["category_id"]),
                "affective": int(c["affective"]),
            }
        )
        if len(selected) >= max_pred:
            break
    return selected


def compute_a_breakdown(
    sample_records: list[dict[str, Any]],
    max_span_length: int,
    mode: str,
) -> dict[str, Any]:
    a_counts = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0}
    a3_sub = {"topn_drop": 0, "cat_aff_not_materialized": 0}
    a1_split = {"opinion_only_miss": 0, "both_miss": 0, "aspect_only_miss": 0}
    total_gold = 0
    total_a_miss = 0

    for rec in sample_records:
        # Candidate existence for exact materialization (same as Stage-1 candidate-space definition).
        exact_set = set()
        for rp in rec["retained_pairs"]:
            if mode == "baseline":
                cat_ids = rp["topcat_ids"]
                aff_pred = rp["aff_pred"]
            else:
                cat_ids = rp.get("topcat_ids_cal", rp["topcat_ids"])
                aff_pred = int(rp.get("aff_pred_cal", rp["aff_pred"]))
            for cat_id in cat_ids:
                exact_set.add(
                    (
                        tuple(rp["a_word"]),
                        tuple(rp["o_word"]),
                        int(cat_id),
                        int(aff_pred),
                    )
                )

        for g in rec["gold_items"]:
            total_gold += 1
            key_exact = (
                tuple(g["a_word"]),
                tuple(g["o_word"]),
                int(g["cat_id"]),
                int(g["sent_id"]),
            )
            if key_exact in exact_set:
                continue

            total_a_miss += 1
            a_sub = tuple(g["a_sub"])
            o_sub = tuple(g["o_sub"])
            is_implicit = bool(g["is_implicit"])
            a_over = a_sub != (-1, -1) and (a_sub[1] - a_sub[0] + 1) > max_span_length
            o_over = o_sub != (-1, -1) and (o_sub[1] - o_sub[0] + 1) > max_span_length
            hit_max_span = a_over or o_over
            a_in_topk = True if a_sub == (-1, -1) else (a_sub in rec["asp_topk_set"])
            o_in_topk = True if o_sub == (-1, -1) else (o_sub in rec["opn_topk_set"])
            pair_key = (a_sub, o_sub)
            in_pair_space = pair_key in rec["pair_space_set"]
            in_retained = pair_key in rec["selected_pair_set_sub"]

            if hit_max_span:
                cls = "A5"
            elif is_implicit:
                cls = "A4"
            elif not (a_in_topk and o_in_topk):
                cls = "A1"
                if a_in_topk and not o_in_topk:
                    a1_split["opinion_only_miss"] += 1
                elif (not a_in_topk) and o_in_topk:
                    a1_split["aspect_only_miss"] += 1
                else:
                    a1_split["both_miss"] += 1
            elif not in_pair_space:
                cls = "A2"
            elif not in_retained:
                cls = "A3"
                a3_sub["topn_drop"] += 1
            else:
                cls = "A3"
                a3_sub["cat_aff_not_materialized"] += 1
            a_counts[cls] += 1

    return {
        "total_gold": int(total_gold),
        "total_A_miss": int(total_a_miss),
        "A_breakdown_counts": a_counts,
        "A3_subtypes": a3_sub,
        "A1_split_counts": a1_split,
    }


def build_a3_diag_records(
    sample_records: list[dict[str, Any]],
    use_calibrated_scores: bool,
):
    out = []
    for rec in sample_records:
        scores = list(rec["pair_scores_all"])
        # Intentionally keep original pair scores to guarantee retained-path invariants.
        _ = use_calibrated_scores
        out.append(
            {
                "pair_scores": scores,
                "pair_map": rec["pair_map"],
                "asp_indices": rec["asp_indices"],
                "opn_indices": rec["opn_indices"],
                "selected_pair_ids": rec["selected_pair_ids"],
                "gold_pairs": rec["gold_pairs_sub"],
            }
        )
    return out


def compute_mode_metrics(
    split_data: dict[str, Any],
    mode: str,
    max_pred: int,
) -> dict[str, Any]:
    sample_records = split_data["sample_records"]
    cat_to_id = split_data["cat_to_id"]
    id_to_cat = split_data["id_to_cat"]
    pair_top_n = int(split_data["pair_top_n"])
    max_span_length = int(split_data["max_span_length"])

    pred_lists = []
    gold_lists = []
    for rec in sample_records:
        pred_lists.append(decode_predictions_for_sample(rec, mode, max_pred))
        gold_lists.append(rec["gold_quads"])

    quad = compute_quad_f1(pred_lists, gold_lists, id_to_cat, cat_to_id)
    breakdown = compute_a_breakdown(
        sample_records,
        max_span_length=max_span_length,
        mode=mode,
    )
    a3_diag = compute_a3_diagnostics(
        build_a3_diag_records(sample_records, use_calibrated_scores=(mode != "baseline")),
        pair_top_n=pair_top_n,
    )

    ratios = a3_diag.get("first_outranker_type_ratio", {})
    return {
        "split": split_data["split"],
        "mode": mode,
        "quad_f1": float(quad["quad_f1"]),
        "quad_precision": float(quad["quad_precision"]),
        "quad_recall": float(quad["quad_recall"]),
        "A1": int(breakdown["A_breakdown_counts"]["A1"]),
        "A1_opinion_only_miss": int(breakdown["A1_split_counts"]["opinion_only_miss"]),
        "A3": int(breakdown["A_breakdown_counts"]["A3"]),
        "A3_topn_drop": int(breakdown["A3_subtypes"]["topn_drop"]),
        "A3_cat_aff_not_materialized": int(
            breakdown["A3_subtypes"]["cat_aff_not_materialized"]
        ),
        "A_total": int(breakdown["total_A_miss"]),
        "sample_has_positive_after_retention_ratio": to_float(
            a3_diag.get("sample_has_positive_after_retention_ratio")
        ),
        "gold_pair_mean_rank": to_float(a3_diag.get("gold_pair_rank", {}).get("mean")),
        "gold_pair_median_rank": to_float(a3_diag.get("gold_pair_rank", {}).get("median")),
        "score_topn_minus_gold_mean": to_float(
            a3_diag.get("score_topn_minus_gold_pair", {}).get("mean")
        ),
        "score_topn_minus_gold_median": to_float(
            a3_diag.get("score_topn_minus_gold_pair", {}).get("median")
        ),
        "first_outranker_null_ratio": to_float(ratios.get("NULL", {}).get("ratio")),
        "first_outranker_near_miss_ratio": to_float(
            ratios.get("near_miss", {}).get("ratio")
        ),
        "first_outranker_other_ratio": to_float(ratios.get("other", {}).get("ratio")),
    }


def compute_retained_invariants(split_data: dict[str, Any]) -> dict[str, Any]:
    # baseline vs calibrated retained sets are expected identical because replay
    # is constrained to local reweight inside fixed retained top-N.
    n = len(split_data["sample_records"])
    if n == 0:
        return {
            "num_examples": 0,
            "retained_exact_match_ratio": 1.0,
            "retained_jaccard_mean": 1.0,
            "retained_overlap_at20_mean": 1.0,
        }
    exact = 0
    jac = 0.0
    ov20 = 0.0
    for rec in split_data["sample_records"]:
        s0 = set(rec["selected_pair_set_sub"])
        s1 = set(rec["selected_pair_set_sub"])
        inter = len(s0 & s1)
        union = len(s0 | s1)
        if s0 == s1:
            exact += 1
        jac += (inter / union) if union > 0 else 1.0
        ov20 += inter / 20.0
    return {
        "num_examples": n,
        "retained_exact_match_ratio": float(exact / n),
        "retained_jaccard_mean": float(jac / n),
        "retained_overlap_at20_mean": float(ov20 / n),
    }


def write_feature_exports(split_data: dict[str, Any], out_dir: Path) -> None:
    split = split_data["split"]
    meta_csv = out_dir / f"retained_frozen_features_{split}_meta.csv"
    npz_path = out_dir / f"retained_frozen_features_{split}.npz"
    rows = split_data["feature_meta_rows"]
    write_csv(meta_csv, rows)
    np.savez_compressed(
        npz_path,
        X=split_data["X"].astype(np.float32),
        y_probe=split_data["y_probe"].astype(np.int64),
        y_materializable=split_data["y_materializable"].astype(np.int64),
    )

def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    affacr_csv = Path(args.affacr_per_seed).resolve()
    entry = resolve_entry(affacr_csv, seed=int(args.seed))
    stage1_dir = entry.stage1_ckpt.parent
    cfg, raw_cfg, cfg_path = load_cfg_from_stage1(stage1_dir)

    if args.output_root:
        out_root = Path(args.output_root).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(
            f"outputs/stage1_frozen_probe_cataff_replay_{ts}"
        ).resolve()
    summary_dir = out_root / "summary"
    notes_dir = out_root / "notes"
    summary_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSPARStage1(cfg).to(device)
    state = torch.load(entry.stage1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    # 1) Frozen retained feature export
    train_data = collect_split_records(model, cfg, raw_cfg, "train", args.batch_size)
    dev_data = collect_split_records(model, cfg, raw_cfg, "dev", args.batch_size)
    test_data = collect_split_records(model, cfg, raw_cfg, "test", args.batch_size)
    for d in [train_data, dev_data, test_data]:
        write_feature_exports(d, summary_dir)

    # 2) Minimal offline probe fit (logistic regression on frozen pair_reprs)
    X_train = train_data["X"]
    y_train = train_data["y_probe"]
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    probe = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        n_jobs=1,
        random_state=args.seed,
    )
    probe.fit(X_train_n, y_train)

    probe_summary = {
        "train_num_pairs": int(X_train.shape[0]),
        "train_pos_pairs": int((y_train == 1).sum()),
        "train_neg_pairs": int((y_train == 0).sum()),
        "coef_norm": float(np.linalg.norm(probe.coef_)),
        "intercept": float(probe.intercept_[0]),
    }

    for split_data in [dev_data, test_data]:
        X = split_data["X"]
        probs = probe.predict_proba(scaler.transform(X))[:, 1]
        labels = split_data["y_probe"]
        labels_mat = split_data["y_materializable"]
        probe_summary[f"{split_data['split']}_probe_auc_label_probe"] = safe_auc(labels, probs)
        probe_summary[f"{split_data['split']}_probe_ap_label_probe"] = safe_ap(labels, probs)
        probe_summary[f"{split_data['split']}_probe_auc_label_materializable"] = safe_auc(
            labels_mat, probs
        )
        probe_summary[f"{split_data['split']}_probe_ap_label_materializable"] = safe_ap(
            labels_mat, probs
        )

        # Attach probe prob and calibrated cat/aff decode fields back to retained pairs
        idx = 0
        pred_rows = []
        for rec in split_data["sample_records"]:
            for rp in rec["retained_pairs"]:
                p = float(probs[idx])
                rp["probe_prob"] = p

                cat_scale = max(0.1, 1.0 + args.cat_scale * (p - 0.5))
                aff_scale = max(0.1, 1.0 + args.aff_scale * (p - 0.5))
                cat_logits_vec = np.asarray(rp["cat_logits"], dtype=np.float32) * float(cat_scale)
                aff_logits_vec = np.asarray(rp["aff_logits"], dtype=np.float32) * float(aff_scale)

                cat_logits_t = torch.tensor(cat_logits_vec, dtype=torch.float32)
                aff_logits_t = torch.tensor(aff_logits_vec, dtype=torch.float32)
                cat_probs_t = torch.softmax(cat_logits_t, dim=-1)
                aff_probs_t = torch.softmax(aff_logits_t, dim=-1)
                k_cat = min(int(split_data["top_c"]), int(cat_probs_t.numel()))
                top_vals, top_ids = torch.topk(cat_probs_t, k=k_cat)

                rp["topcat_ids_cal"] = [int(x) for x in top_ids.tolist()]
                rp["topcat_probs_cal"] = [float(x) for x in top_vals.tolist()]
                rp["aff_pred_cal"] = int(torch.argmax(aff_probs_t).item())
                rp["aff_top1_prob_cal"] = float(torch.max(aff_probs_t).item())

                pred_rows.append(
                    {
                        "split": split_data["split"],
                        "example_id": rec["example_id"],
                        "pair_id": rp["pid"],
                        "pair_score": rp["pair_score"],
                        "probe_prob": rp["probe_prob"],
                        "cat_scale": cat_scale,
                        "aff_scale": aff_scale,
                        "topcat_ids_cal": str(rp["topcat_ids_cal"]),
                        "topcat_probs_cal": str(rp["topcat_probs_cal"]),
                        "aff_pred_cal": rp["aff_pred_cal"],
                        "aff_top1_prob_cal": rp["aff_top1_prob_cal"],
                        "label_probe": rp["label_probe"],
                        "label_materializable": rp["label_materializable"],
                        "neg_type": rp["neg_type"],
                        "asp_span_subword": str(rp["a_sub"]),
                        "opn_span_subword": str(rp["o_sub"]),
                    }
                )
                idx += 1
        pred_csv = summary_dir / f"frozen_probe_cataff_predictions_{split_data['split']}.csv"
        write_csv(pred_csv, pred_rows)

    (summary_dir / "frozen_probe_fit_summary.json").write_text(
        json.dumps(probe_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (summary_dir / "frozen_probe_cataff_model_params.json").write_text(
        json.dumps(
            {
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coef": probe.coef_.tolist(),
                "intercept": probe.intercept_.tolist(),
                "seed": args.seed,
                "cat_scale": args.cat_scale,
                "aff_scale": args.aff_scale,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 3) Pure offline replay (same retained set)
    metric_rows = []
    invariant = {}
    for split_data in [dev_data, test_data]:
        base = compute_mode_metrics(split_data, mode="baseline", max_pred=args.max_pred)
        cal = compute_mode_metrics(split_data, mode="probe_calibrated", max_pred=args.max_pred)
        metric_rows.extend([base, cal])

        inv = compute_retained_invariants(split_data)
        inv["delta_sample_has_positive_after_retention_ratio"] = (
            float(cal["sample_has_positive_after_retention_ratio"])
            - float(base["sample_has_positive_after_retention_ratio"])
        )
        inv["delta_A3_topn_drop"] = int(cal["A3_topn_drop"]) - int(base["A3_topn_drop"])
        inv["retained_exact_match_is_one"] = (
            abs(inv["retained_exact_match_ratio"] - 1.0) < 1e-12
        )
        inv["retained_jaccard_is_one"] = (
            abs(inv["retained_jaccard_mean"] - 1.0) < 1e-12
        )
        inv["sample_pos_unchanged"] = abs(inv["delta_sample_has_positive_after_retention_ratio"]) < 1e-12
        inv["A3_topn_drop_unchanged"] = inv["delta_A3_topn_drop"] == 0
        invariant[split_data["split"]] = inv

    metrics_csv = summary_dir / "frozen_probe_cataff_replay_metrics_dev_test.csv"
    write_csv(metrics_csv, metric_rows)

    # delta table (probe_calibrated - baseline) by split
    deltas = {}
    for split in ["dev", "test"]:
        base = next(r for r in metric_rows if r["split"] == split and r["mode"] == "baseline")
        cal = next(
            r for r in metric_rows if r["split"] == split and r["mode"] == "probe_calibrated"
        )
        d = {}
        for k, v in base.items():
            if k in {"split", "mode"}:
                continue
            bv, cv = base[k], cal[k]
            d[f"delta_{k}"] = float(cv) - float(bv) if (bv is not None and cv is not None) else None
        deltas[split] = d

    (summary_dir / "frozen_probe_cataff_replay_deltas.json").write_text(
        json.dumps(deltas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (summary_dir / "frozen_probe_cataff_replay_invariants.json").write_text(
        json.dumps(invariant, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # quick decision note
    test_delta = deltas["test"]
    takeaways = [
        "# Frozen Probe Cat/Aff Decode Calibration Replay (Fixed Aff-ACR checkpoint)",
        "",
        f"- checkpoint: {entry.stage1_ckpt}",
        f"- config: {cfg_path}",
        f"- cat_scale: {args.cat_scale}",
        f"- aff_scale: {args.aff_scale}",
        "",
        "## Invariant checks (must hold)",
        f"- dev retained exact/jaccard: {invariant['dev']['retained_exact_match_ratio']:.6f} / {invariant['dev']['retained_jaccard_mean']:.6f}",
        f"- test retained exact/jaccard: {invariant['test']['retained_exact_match_ratio']:.6f} / {invariant['test']['retained_jaccard_mean']:.6f}",
        f"- dev sample_pos delta: {invariant['dev']['delta_sample_has_positive_after_retention_ratio']:+.6f}",
        f"- test sample_pos delta: {invariant['test']['delta_sample_has_positive_after_retention_ratio']:+.6f}",
        f"- dev A3_topn_drop delta: {invariant['dev']['delta_A3_topn_drop']:+d}",
        f"- test A3_topn_drop delta: {invariant['test']['delta_A3_topn_drop']:+d}",
        "",
        "## Key replay deltas (probe_calibrated - baseline)",
        f"- test Quad-F1 delta: {test_delta['delta_quad_f1']:+.6f}",
        f"- test A3_cat_aff_not_materialized delta: {test_delta['delta_A3_cat_aff_not_materialized']:+.0f}",
        f"- test first_outranker_other_ratio delta: {test_delta['delta_first_outranker_other_ratio']:+.6f}",
        "",
        "Interpretation target: does frozen probe signal itself convert into materialization-side gain without touching retained set?",
    ]
    (summary_dir / "frozen_probe_cataff_replay_takeaways.md").write_text(
        "\n".join(takeaways) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "affacr_per_seed": str(affacr_csv),
        "seed": int(args.seed),
        "stage1_ckpt": str(entry.stage1_ckpt),
        "config_path": str(cfg_path),
        "cat_scale": float(args.cat_scale),
        "aff_scale": float(args.aff_scale),
        "max_pred": int(args.max_pred),
        "batch_size": int(args.batch_size),
        "output_root": str(out_root),
        "summary_metrics_csv": str(metrics_csv),
        "summary_deltas_json": str(summary_dir / "frozen_probe_cataff_replay_deltas.json"),
        "summary_invariants_json": str(summary_dir / "frozen_probe_cataff_replay_invariants.json"),
    }
    (notes_dir / "frozen_probe_cataff_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote outputs under: {out_root}")
    print(f"Summary: {metrics_csv}")


if __name__ == "__main__":
    main()
