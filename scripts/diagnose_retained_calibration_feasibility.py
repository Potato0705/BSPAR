"""Retained-only calibration feasibility diagnosis (read-only replay).

Goal:
- Re-run Stage-1 inference on test split using existing checkpoints.
- Extract retained top-N (top-20) pair-level signals without changing model logic.
- Measure whether current retained signals can separate gold-like pairs from
  near-miss / category-confused hard negatives.

Outputs:
- summary/retained_calibration_pair_features.csv
- summary/retained_calibration_separability_per_seed.csv
- summary/retained_calibration_separability_group.json
- summary/retained_calibration_takeaways.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

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
from bspar.models.bspar_stage1 import BSPARStage1


@dataclass
class RunEntry:
    track: str
    seed: int
    stage1_ckpt: Path
    run_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose retained-only calibration feasibility on existing runs"
    )
    parser.add_argument(
        "--affacr_per_seed",
        default=(
            "outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/"
            "summary/affacr_a0_4seed_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--cbr_per_seed",
        default=(
            "outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_20260318_082343/"
            "summary/cbrv1_a0_4seed_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Default: outputs/stage1_retained_calibration_feasibility_<timestamp>"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--tracks",
        default="affacr_only,cbr_v1",
        help="Comma separated subset of tracks to run",
    )
    return parser.parse_args()


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return float(v)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_entries(affacr_csv: Path, cbr_csv: Path, track_filter: set[str]) -> list[RunEntry]:
    entries: list[RunEntry] = []
    if "affacr_only" in track_filter:
        for r in load_csv_rows(affacr_csv):
            entries.append(
                RunEntry(
                    track="affacr_only",
                    seed=int(r["seed"]),
                    stage1_ckpt=Path(r["stage1_ckpt"]).resolve(),
                    run_dir=Path(r["run_dir"]).resolve() if r.get("run_dir") else None,
                )
            )
    if "cbr_v1" in track_filter:
        for r in load_csv_rows(cbr_csv):
            entries.append(
                RunEntry(
                    track="cbr_v1",
                    seed=int(r["seed"]),
                    stage1_ckpt=Path(r["stage1_ckpt"]).resolve(),
                    run_dir=Path(r["run_dir"]).resolve() if r.get("run_dir") else None,
                )
            )
    entries.sort(key=lambda x: (x.track, x.seed))
    return entries


def load_config_from_stage1_dir(stage1_dir: Path) -> tuple[BSPARConfig, dict[str, Any], Path]:
    manifest_path = stage1_dir / "manifest.json"
    config_path: Path | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg_raw = manifest.get("config_path")
        if cfg_raw:
            p = Path(cfg_raw)
            if p.exists():
                config_path = p.resolve()
    if config_path is None:
        raise FileNotFoundError(
            f"Cannot resolve config_path from {manifest_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = BSPARConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, raw, config_path


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * q)
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "q50": 0.0, "q90": 0.0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "q50": quantile(values, 0.5),
        "q90": quantile(values, 0.9),
    }


def roc_auc_binary(scores: list[float], labels: list[int]) -> float:
    pairs = list(zip(scores, labels))
    n = len(pairs)
    if n == 0:
        return 0.0
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    pairs.sort(key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = 0.0
    for r, (_, y) in zip(ranks, pairs):
        if y == 1:
            rank_sum_pos += r
    u = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def _word_to_sub(model: BSPARStage1, span_obj, w2s):
    if span_obj.is_null:
        return (-1, -1)
    sub = model._word_span_to_subword(span_obj.start, span_obj.end, w2s)
    return None if sub is None else tuple(sub)


def _build_gold_maps(
    model: BSPARStage1,
    gold_quads,
    w2s,
    cat_to_id: dict[str, int],
) -> tuple[dict[tuple[tuple[int, int], tuple[int, int]], dict[str, set[int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    gold_pair_multi: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, set[int]]] = {}
    gold_aspects: set[tuple[int, int]] = set()
    gold_opinions: set[tuple[int, int]] = set()
    for q in gold_quads:
        a_sub = _word_to_sub(model, q.aspect, w2s)
        o_sub = _word_to_sub(model, q.opinion, w2s)
        if a_sub is None or o_sub is None:
            continue
        if a_sub != (-1, -1):
            gold_aspects.add(a_sub)
        if o_sub != (-1, -1):
            gold_opinions.add(o_sub)
        key = (a_sub, o_sub)
        if key not in gold_pair_multi:
            gold_pair_multi[key] = {"cat_ids": set(), "sent_ids": set()}
        if q.category in cat_to_id:
            gold_pair_multi[key]["cat_ids"].add(int(cat_to_id[q.category]))
        if q.sentiment in SENTIMENT_TO_ID:
            gold_pair_multi[key]["sent_ids"].add(int(SENTIMENT_TO_ID[q.sentiment]))
    return gold_pair_multi, gold_aspects, gold_opinions


def classify_negative(
    a_span: tuple[int, int],
    o_span: tuple[int, int],
    gold_aspects: set[tuple[int, int]],
    gold_opinions: set[tuple[int, int]],
) -> str:
    if a_span == (-1, -1) or o_span == (-1, -1):
        return "null_related"
    a_gold = a_span in gold_aspects
    o_gold = o_span in gold_opinions
    if a_gold and o_gold:
        return "category_confused"
    if a_gold or o_gold:
        return "near_miss"
    return "other"


def analyze_run(
    entry: RunEntry,
    batch_size_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage1_dir = entry.stage1_ckpt.parent
    cfg, raw_cfg, cfg_path = load_config_from_stage1_dir(stage1_dir)
    dataset_name = raw_cfg.get("dataset_name", "asqp_rest15")
    data_format = raw_cfg.get("data_format", "auto")
    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, _ = build_category_map(categories)
    cfg.num_categories = len(categories)

    data_dir = raw_cfg.get("data_dir", "data/asqp_rest15")
    test_file = Path(data_dir) / raw_cfg.get("test_file", "test.txt")
    test_examples = load_data(str(test_file), data_format, categories)

    ds = BSPARStage1Dataset(
        test_examples,
        cfg.model_name,
        max_length=128,
        max_span_length=cfg.max_span_length,
    )
    loader = DataLoader(
        ds,
        batch_size=(batch_size_override if batch_size_override else cfg.batch_size),
        shuffle=False,
        collate_fn=collate_stage1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSPARStage1(cfg).to(device)
    state = torch.load(entry.stage1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    pair_thr = getattr(cfg, "stage1_pair_score_threshold", 0.001)
    pair_strategy = getattr(cfg, "stage1_pair_retention_strategy", "topn_only")
    pair_top_n = int(getattr(cfg, "stage1_pair_top_n", 20))
    top_c = int(getattr(cfg, "top_c_categories", 3))

    rows: list[dict[str, Any]] = []
    ex_idx = 0
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
            pair_map = outputs["pair_map"]
            pair_scores = torch.sigmoid(outputs["pair_scores"])
            cat_logits = outputs["cat_logits"]
            aff_logits = outputs["aff_output"]
            selected_ids_batch = outputs.get("selected_pair_ids", [])
            asp_indices_batch = outputs["asp_indices"]
            opn_indices_batch = outputs["opn_indices"]

            for b in range(len(outputs["candidates"])):
                ex = test_examples[ex_idx]
                w2s = batch["word_to_subword"][b]
                gold_quads = batch["gold_quads"][b]
                gold_pair_multi, gold_aspects, gold_opinions = _build_gold_maps(
                    model, gold_quads, w2s, cat_to_id
                )
                selected_ids = selected_ids_batch[b] if b < len(selected_ids_batch) else []
                asp_indices = asp_indices_batch[b]
                opn_indices = opn_indices_batch[b]

                # same-aspect competition stats within retained set
                by_aspect: dict[int, list[int]] = {}
                for pid in selected_ids:
                    ai, _ = pair_map[pid]
                    by_aspect.setdefault(int(ai), []).append(int(pid))
                for ai in list(by_aspect.keys()):
                    by_aspect[ai] = sorted(
                        by_aspect[ai],
                        key=lambda pid: float(pair_scores[b, pid].item()),
                        reverse=True,
                    )

                for pid in selected_ids:
                    ai, oi = pair_map[pid]
                    a_span = tuple(asp_indices[ai])
                    o_span = tuple(opn_indices[oi])
                    pair_key = (a_span, o_span)
                    pscore = float(pair_scores[b, pid].item())

                    cat_logit = cat_logits[b, pid]
                    cat_prob = torch.softmax(cat_logit, dim=-1)
                    k_cat = min(2, cat_prob.numel())
                    cat_top2_prob = torch.topk(cat_prob, k=k_cat).values
                    cat_top2_logit = torch.topk(cat_logit, k=k_cat).values
                    cat_margin_prob = float(
                        cat_top2_prob[0] - (cat_top2_prob[1] if k_cat > 1 else 0.0)
                    )
                    cat_margin_logit = float(
                        cat_top2_logit[0] - (cat_top2_logit[1] if k_cat > 1 else 0.0)
                    )
                    cat_top1_prob = float(cat_top2_prob[0].item())
                    cat_topk_ids = torch.topk(
                        cat_prob, k=min(top_c, cat_prob.numel())
                    ).indices.tolist()

                    aff_logit = aff_logits[b, pid]
                    aff_prob = torch.softmax(aff_logit, dim=-1)
                    k_aff = min(2, aff_prob.numel())
                    aff_top2_prob = torch.topk(aff_prob, k=k_aff).values
                    aff_top2_logit = torch.topk(aff_logit, k=k_aff).values
                    aff_margin_prob = float(
                        aff_top2_prob[0] - (aff_top2_prob[1] if k_aff > 1 else 0.0)
                    )
                    aff_margin_logit = float(
                        aff_top2_logit[0] - (aff_top2_logit[1] if k_aff > 1 else 0.0)
                    )
                    aff_top1_prob = float(aff_top2_prob[0].item())
                    aff_pred = int(torch.argmax(aff_logit).item())

                    is_gold_pair = pair_key in gold_pair_multi
                    gold_cat_hit_topc = False
                    gold_aff_hit_top1 = False
                    materializable = False
                    if is_gold_pair:
                        gold_meta = gold_pair_multi[pair_key]
                        gold_cat_hit_topc = bool(
                            set(int(x) for x in cat_topk_ids).intersection(
                                gold_meta["cat_ids"]
                            )
                        )
                        gold_aff_hit_top1 = aff_pred in gold_meta["sent_ids"]
                        materializable = gold_cat_hit_topc and gold_aff_hit_top1

                    neg_type = (
                        "gold"
                        if is_gold_pair
                        else classify_negative(a_span, o_span, gold_aspects, gold_opinions)
                    )

                    same_aspect = by_aspect.get(int(ai), [])
                    within_rank = (
                        same_aspect.index(int(pid)) + 1 if int(pid) in same_aspect else 0
                    )
                    top_aspect_score = (
                        float(pair_scores[b, same_aspect[0]].item()) if same_aspect else pscore
                    )
                    gap_to_aspect_top = float(top_aspect_score - pscore)

                    rows.append(
                        {
                            "track": entry.track,
                            "seed": entry.seed,
                            "example_index": ex_idx,
                            "example_id": ex.id,
                            "pair_id": int(pid),
                            "aspect_idx": int(ai),
                            "opinion_idx": int(oi),
                            "asp_span_subword": str(a_span),
                            "opn_span_subword": str(o_span),
                            "pair_score": pscore,
                            "cat_top1_prob": cat_top1_prob,
                            "cat_margin_prob": cat_margin_prob,
                            "cat_margin_logit": cat_margin_logit,
                            "aff_top1_prob": aff_top1_prob,
                            "aff_margin_prob": aff_margin_prob,
                            "aff_margin_logit": aff_margin_logit,
                            "joint_confidence_prod": cat_top1_prob * aff_top1_prob,
                            "joint_margin_sum": cat_margin_prob + aff_margin_prob,
                            "same_aspect_retained_count": len(same_aspect),
                            "same_aspect_rank": within_rank,
                            "gap_to_same_aspect_top": gap_to_aspect_top,
                            "is_gold_pair": int(is_gold_pair),
                            "is_gold_materializable": int(materializable),
                            "gold_cat_hit_topc": int(gold_cat_hit_topc),
                            "gold_aff_hit_top1": int(gold_aff_hit_top1),
                            "neg_type": neg_type,
                            "config_path": str(cfg_path),
                            "stage1_ckpt": str(entry.stage1_ckpt),
                        }
                    )

                ex_idx += 1

    # run-level quick summary for logs
    n_rows = len(rows)
    n_gold = sum(r["is_gold_pair"] for r in rows)
    n_near = sum(1 for r in rows if r["neg_type"] == "near_miss")
    n_catconf = sum(1 for r in rows if r["neg_type"] == "category_confused")
    run_summary = {
        "track": entry.track,
        "seed": entry.seed,
        "retained_pairs": n_rows,
        "gold_pairs": n_gold,
        "near_miss_neg": n_near,
        "category_confused_neg": n_catconf,
    }
    return rows, run_summary


def calc_separability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # target: gold-like vs hard negatives (near_miss + category_confused)
    use_rows = [
        r for r in rows
        if r["is_gold_pair"] == 1 or r["neg_type"] in {"near_miss", "category_confused"}
    ]
    labels = [1 if r["is_gold_pair"] == 1 else 0 for r in use_rows]
    feats = {
        "pair_score": [float(r["pair_score"]) for r in use_rows],
        "cat_margin_prob": [float(r["cat_margin_prob"]) for r in use_rows],
        "aff_margin_prob": [float(r["aff_margin_prob"]) for r in use_rows],
        "joint_confidence_prod": [float(r["joint_confidence_prod"]) for r in use_rows],
        "joint_margin_sum": [float(r["joint_margin_sum"]) for r in use_rows],
        # lower same_aspect_rank means better; invert for AUC.
        "neg_same_aspect_rank": [-float(r["same_aspect_rank"]) for r in use_rows],
        "neg_gap_to_same_aspect_top": [-float(r["gap_to_same_aspect_top"]) for r in use_rows],
    }
    aucs = {k: roc_auc_binary(v, labels) for k, v in feats.items()}

    def group_vals(filter_fn, key):
        return [float(r[key]) for r in rows if filter_fn(r)]

    gold_like = lambda r: r["is_gold_pair"] == 1
    near = lambda r: r["neg_type"] == "near_miss"
    catc = lambda r: r["neg_type"] == "category_confused"

    dist = {}
    for k in [
        "pair_score",
        "cat_margin_prob",
        "aff_margin_prob",
        "joint_confidence_prod",
        "joint_margin_sum",
        "gap_to_same_aspect_top",
    ]:
        dist[k] = {
            "gold_like": summarize(group_vals(gold_like, k)),
            "near_miss": summarize(group_vals(near, k)),
            "category_confused": summarize(group_vals(catc, k)),
        }

    n_gold = sum(1 for r in rows if r["is_gold_pair"] == 1)
    n_near = sum(1 for r in rows if r["neg_type"] == "near_miss")
    n_cat = sum(1 for r in rows if r["neg_type"] == "category_confused")
    n_other = sum(1 for r in rows if r["is_gold_pair"] == 0 and r["neg_type"] not in {"near_miss", "category_confused"})
    return {
        "counts": {
            "gold_like_retained": n_gold,
            "near_miss_retained": n_near,
            "category_confused_retained": n_cat,
            "other_retained": n_other,
            "used_for_auc": len(labels),
        },
        "auc_gold_vs_hard_neg": aucs,
        "distributions": dist,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    tracks = set(x.strip() for x in args.tracks.split(",") if x.strip())
    affacr_csv = Path(args.affacr_per_seed).resolve()
    cbr_csv = Path(args.cbr_per_seed).resolve()

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(f"outputs/stage1_retained_calibration_feasibility_{ts}").resolve()
    summary_dir = output_root / "summary"
    notes_dir = output_root / "notes"
    summary_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(affacr_csv, cbr_csv, tracks)
    if not entries:
        raise RuntimeError("No entries selected to analyze.")

    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    sep_rows: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {
        "tracks": {},
        "run_summaries": [],
    }

    for e in entries:
        print(f"[diag] track={e.track} seed={e.seed}")
        rows, run_sum = analyze_run(e, batch_size_override=args.batch_size)
        all_rows.extend(rows)
        run_summaries.append(run_sum)
        sep = calc_separability(rows)

        sep_row = {
            "track": e.track,
            "seed": e.seed,
            **sep["counts"],
        }
        for k, v in sep["auc_gold_vs_hard_neg"].items():
            sep_row[f"auc_{k}"] = v
        sep_rows.append(sep_row)

        group_summary["tracks"].setdefault(e.track, {})
        group_summary["tracks"][e.track][str(e.seed)] = sep

    # CSV: pair-level features
    pair_fields = [
        "track",
        "seed",
        "example_index",
        "example_id",
        "pair_id",
        "aspect_idx",
        "opinion_idx",
        "asp_span_subword",
        "opn_span_subword",
        "pair_score",
        "cat_top1_prob",
        "cat_margin_prob",
        "cat_margin_logit",
        "aff_top1_prob",
        "aff_margin_prob",
        "aff_margin_logit",
        "joint_confidence_prod",
        "joint_margin_sum",
        "same_aspect_retained_count",
        "same_aspect_rank",
        "gap_to_same_aspect_top",
        "is_gold_pair",
        "is_gold_materializable",
        "gold_cat_hit_topc",
        "gold_aff_hit_top1",
        "neg_type",
        "config_path",
        "stage1_ckpt",
    ]
    pair_csv = summary_dir / "retained_calibration_pair_features.csv"
    write_csv(pair_csv, all_rows, pair_fields)

    # CSV: separability per seed
    sep_fields = [
        "track",
        "seed",
        "gold_like_retained",
        "near_miss_retained",
        "category_confused_retained",
        "other_retained",
        "used_for_auc",
        "auc_pair_score",
        "auc_cat_margin_prob",
        "auc_aff_margin_prob",
        "auc_joint_confidence_prod",
        "auc_joint_margin_sum",
        "auc_neg_same_aspect_rank",
        "auc_neg_gap_to_same_aspect_top",
    ]
    sep_csv = summary_dir / "retained_calibration_separability_per_seed.csv"
    write_csv(sep_csv, sep_rows, sep_fields)

    # group summary (track-level avg AUC)
    track_avg = {}
    for tr in sorted(set(r["track"] for r in sep_rows)):
        rs = [r for r in sep_rows if r["track"] == tr]
        avg_auc = {}
        for k in [
            "auc_pair_score",
            "auc_cat_margin_prob",
            "auc_aff_margin_prob",
            "auc_joint_confidence_prod",
            "auc_joint_margin_sum",
            "auc_neg_same_aspect_rank",
            "auc_neg_gap_to_same_aspect_top",
        ]:
            vals = [to_float(r.get(k)) for r in rs if r.get(k) is not None]
            avg_auc[k] = float(sum(vals) / len(vals)) if vals else None
        track_avg[tr] = {
            "num_seeds": len(rs),
            "avg_auc": avg_auc,
        }

    group_summary["run_summaries"] = run_summaries
    group_summary["track_avg"] = track_avg
    group_json = summary_dir / "retained_calibration_separability_group.json"
    group_json.write_text(
        json.dumps(group_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # markdown takeaways
    lines = [
        "# Retained Calibration Feasibility Diagnosis",
        "",
        "## Coverage",
    ]
    for r in run_summaries:
        lines.append(
            f"- {r['track']} seed{r['seed']}: retained={r['retained_pairs']}, "
            f"gold={r['gold_pairs']}, near_miss={r['near_miss_neg']}, "
            f"category_confused={r['category_confused_neg']}"
        )
    lines.append("")
    lines.append("## Track-level AUC (gold-like vs near_miss/category_confused)")
    for tr, node in track_avg.items():
        lines.append(f"- {tr} ({node['num_seeds']} seeds)")
        for k, v in node["avg_auc"].items():
            lines.append(f"  - {k}: {v:.4f}" if v is not None else f"  - {k}: NA")
    lines.append("")
    lines.append("## Heuristic conclusion")
    # simple separability signal: joint/pair AUC >= 0.60
    feasible_tracks = []
    for tr, node in track_avg.items():
        a = node["avg_auc"]
        best = max(
            x for x in [
                a.get("auc_joint_confidence_prod"),
                a.get("auc_joint_margin_sum"),
                a.get("auc_pair_score"),
                a.get("auc_cat_margin_prob"),
                a.get("auc_aff_margin_prob"),
            ] if x is not None
        )
        if best >= 0.60:
            feasible_tracks.append((tr, best))
    if feasible_tracks:
        lines.append(
            "- Retained-only calibration appears feasible: at least one existing "
            "signal reaches usable separability (AUC >= 0.60)."
        )
        for tr, best in feasible_tracks:
            lines.append(f"  - {tr}: best feature AUC={best:.4f}")
    else:
        lines.append(
            "- No strong separability signal found (best AUC < 0.60); "
            "retained-only calibration may not be worthwhile."
        )

    md = summary_dir / "retained_calibration_takeaways.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = {
        "affacr_per_seed": str(affacr_csv),
        "cbr_per_seed": str(cbr_csv),
        "tracks": sorted(list(tracks)),
        "num_entries": len(entries),
        "pair_feature_csv": str(pair_csv),
        "separability_per_seed_csv": str(sep_csv),
        "separability_group_json": str(group_json),
        "takeaways_md": str(md),
    }
    (notes_dir / "diagnosis_manifest.json").write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote: {pair_csv}")
    print(f"Wrote: {sep_csv}")
    print(f"Wrote: {group_json}")
    print(f"Wrote: {md}")


if __name__ == "__main__":
    main()
