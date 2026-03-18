"""Retained-only oracle upper-bound + frozen-probe feasibility diagnosis.

Read-only diagnosis:
- No change to Stage-1/Stage-2 logic.
- Retention path unchanged: topn_only + top_n=20.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
class Entry:
    track: str
    seed: int
    stage1_ckpt: Path
    run_dir: Path | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--affacr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/summary/affacr_a0_4seed_per_seed.csv",
    )
    p.add_argument(
        "--cbr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_20260318_082343/summary/cbrv1_a0_4seed_per_seed.csv",
    )
    p.add_argument("--tracks", default="affacr_only,cbr_v1")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_root", default=None)
    return p.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_entries(affacr_csv: Path, cbr_csv: Path, tracks: set[str]) -> list[Entry]:
    out: list[Entry] = []
    if "affacr_only" in tracks:
        for r in load_csv(affacr_csv):
            out.append(Entry("affacr_only", int(r["seed"]), Path(r["stage1_ckpt"]).resolve(), Path(r["run_dir"]).resolve()))
    if "cbr_v1" in tracks:
        for r in load_csv(cbr_csv):
            out.append(Entry("cbr_v1", int(r["seed"]), Path(r["stage1_ckpt"]).resolve(), Path(r["run_dir"]).resolve()))
    out.sort(key=lambda x: (x.track, x.seed))
    return out


def to_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return int(v)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _word_to_sub(model: BSPARStage1, span_obj, w2s):
    if span_obj.is_null:
        return (-1, -1)
    sub = model._word_span_to_subword(span_obj.start, span_obj.end, w2s)
    return None if sub is None else tuple(sub)


def classify_neg(a_span, o_span, gold_aspects, gold_opinions) -> str:
    if a_span == (-1, -1) or o_span == (-1, -1):
        return "null_related"
    a_gold = a_span in gold_aspects
    o_gold = o_span in gold_opinions
    if a_gold and o_gold:
        return "category_confused"
    if a_gold or o_gold:
        return "near_miss"
    return "other"


def load_cfg(stage1_dir: Path) -> tuple[BSPARConfig, dict[str, Any], Path]:
    manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg_path = Path(manifest["config_path"]).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = BSPARConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, raw, cfg_path


def build_loader(raw_cfg: dict[str, Any], cfg: BSPARConfig, split: str, batch_size: int):
    dataset_name = raw_cfg.get("dataset_name", "asqp_rest15")
    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, _ = build_category_map(categories)
    cfg.num_categories = len(categories)
    data_dir = raw_cfg.get("data_dir", "data/asqp_rest15")
    file_name = raw_cfg["train_file"] if split == "train" else raw_cfg["test_file"]
    examples = load_data(str(Path(data_dir) / file_name), raw_cfg.get("data_format", "auto"), categories)
    ds = BSPARStage1Dataset(examples, cfg.model_name, max_length=128, max_span_length=cfg.max_span_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stage1)
    return loader, examples, cat_to_id


def collect_split(model: BSPARStage1, cfg: BSPARConfig, raw_cfg: dict[str, Any], split: str, batch_size: int):
    loader, examples, cat_to_id = build_loader(raw_cfg, cfg, split, batch_size)
    device = next(model.parameters()).device
    pair_top_n = int(getattr(cfg, "stage1_pair_top_n", 20))
    pair_thr = float(getattr(cfg, "stage1_pair_score_threshold", 0.001))
    pair_strategy = str(getattr(cfg, "stage1_pair_retention_strategy", "topn_only"))

    X_repr, X_stats, y, neg_type = [], [], [], []
    signals = {k: [] for k in ["pair_score", "cat_margin_prob", "aff_margin_prob", "joint_confidence_prod", "joint_margin_sum", "same_aspect_rank_inv", "gap_to_same_aspect_top_inv"]}

    ex_idx = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                word_to_subword=batch["word_to_subword"],
                mode="inference",
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )
            pair_map = outputs["pair_map"]
            pair_probs = torch.sigmoid(outputs["pair_scores"])
            cat_logits = outputs["cat_logits"]
            aff_logits = outputs["aff_output"]
            selected_batch = outputs.get("selected_pair_ids", [])
            asp_indices_batch = outputs["asp_indices"]
            opn_indices_batch = outputs["opn_indices"]

            for b in range(len(outputs["candidates"])):
                gold_quads = batch["gold_quads"][b]
                w2s = batch["word_to_subword"][b]
                asp_indices = asp_indices_batch[b]
                opn_indices = opn_indices_batch[b]
                selected_ids = selected_batch[b] if b < len(selected_batch) else []

                gold_pairs = set()
                gold_aspects = set()
                gold_opinions = set()
                for q in gold_quads:
                    a_sub = _word_to_sub(model, q.aspect, w2s)
                    o_sub = _word_to_sub(model, q.opinion, w2s)
                    if a_sub is None or o_sub is None:
                        continue
                    gold_pairs.add((a_sub, o_sub))
                    if a_sub != (-1, -1):
                        gold_aspects.add(a_sub)
                    if o_sub != (-1, -1):
                        gold_opinions.add(o_sub)

                by_aspect: dict[int, list[int]] = {}
                for pid in selected_ids:
                    ai, _ = pair_map[pid]
                    by_aspect.setdefault(int(ai), []).append(int(pid))
                for ai in by_aspect:
                    by_aspect[ai].sort(key=lambda pid: float(pair_probs[b, pid].item()), reverse=True)

                for pid in selected_ids:
                    ai, oi = pair_map[pid]
                    a_span = tuple(asp_indices[ai]); o_span = tuple(opn_indices[oi])
                    is_gold = 1 if (a_span, o_span) in gold_pairs else 0
                    nt = "gold" if is_gold else classify_neg(a_span, o_span, gold_aspects, gold_opinions)
                    pscore = float(pair_probs[b, pid].item())

                    cp = torch.softmax(cat_logits[b, pid], dim=-1)
                    ap = torch.softmax(aff_logits[b, pid], dim=-1)
                    cat_top2 = torch.topk(cp, k=min(2, cp.numel())).values
                    aff_top2 = torch.topk(ap, k=min(2, ap.numel())).values
                    cat_margin = float(cat_top2[0] - (cat_top2[1] if cat_top2.numel() > 1 else 0.0))
                    aff_margin = float(aff_top2[0] - (aff_top2[1] if aff_top2.numel() > 1 else 0.0))
                    cat_top1 = float(cat_top2[0].item()); aff_top1 = float(aff_top2[0].item())

                    same_aspect = by_aspect.get(int(ai), [])
                    rank = (same_aspect.index(int(pid)) + 1) if int(pid) in same_aspect else 1
                    group_sz = max(1, len(same_aspect))
                    top_score = float(pair_probs[b, same_aspect[0]].item()) if same_aspect else pscore
                    gap_top = float(top_score - pscore)
                    rank_norm = float(rank / group_sz)

                    X_repr.append(outputs["pair_reprs"][b, pid].detach().cpu().numpy().astype(np.float32))
                    X_stats.append(np.array([pscore, cat_margin, aff_margin, cat_top1 * aff_top1, cat_margin + aff_margin, rank_norm, gap_top, group_sz / pair_top_n], dtype=np.float32))
                    y.append(is_gold)
                    neg_type.append(nt)

                    signals["pair_score"].append(pscore)
                    signals["cat_margin_prob"].append(cat_margin)
                    signals["aff_margin_prob"].append(aff_margin)
                    signals["joint_confidence_prod"].append(cat_top1 * aff_top1)
                    signals["joint_margin_sum"].append(cat_margin + aff_margin)
                    signals["same_aspect_rank_inv"].append(-rank_norm)
                    signals["gap_to_same_aspect_top_inv"].append(-gap_top)
                ex_idx += 1

    return {
        "X_repr": np.stack(X_repr, axis=0),
        "X_stats": np.stack(X_stats, axis=0),
        "y": np.array(y, dtype=np.int64),
        "neg_type": np.array(neg_type),
        "signals": {k: np.array(v, dtype=np.float32) for k, v in signals.items()},
    }


def oracle_from_json(stage1_dir: Path) -> dict[str, Any]:
    br = json.loads((stage1_dir / "stage1_a_breakdown_test.json").read_text(encoding="utf-8"))
    a1s = json.loads((stage1_dir / "a1_span_side_split_test.json").read_text(encoding="utf-8"))
    A1 = int(br["A_breakdown_counts"]["A1"])
    A3 = int(br["A_breakdown_counts"]["A3"])
    A3_topn = int(br["A3_subtypes"]["topn_drop"])
    A3_cataff = int(br["A3_subtypes"]["cat_aff_not_materialized"])
    A_total = int(br["total_A_miss"])
    A1_opn = to_int(a1s.get("A1_split_counts", {}).get("opinion_only_miss"))
    return {
        "A1": A1,
        "A1_opinion_only_miss": A1_opn,
        "A3": A3,
        "A3_topn_drop": A3_topn,
        "A3_cat_aff_not_materialized": A3_cataff,
        "A_total": A_total,
        "oracle_fixable_A1_opinion_only_miss": 0,
        "oracle_fixable_A3_cat_aff_not_materialized": A3_cataff,
        "oracle_A1_after": A1,
        "oracle_A3_after": A3 - A3_cataff,
        "oracle_A_total_after": A_total - A3_cataff,
        "oracle_fixable_ratio_over_A_total": float(A3_cataff / A_total) if A_total > 0 else 0.0,
    }


def subtype_auc(score: np.ndarray, y: np.ndarray, neg_type: np.ndarray, subtype: str) -> float:
    mask = (y == 1) | ((y == 0) & (neg_type == subtype))
    return safe_auc(y[mask], score[mask])


def main() -> None:
    args = parse_args()
    tracks = {x.strip() for x in args.tracks.split(",") if x.strip()}
    entries = load_entries(Path(args.affacr_per_seed).resolve(), Path(args.cbr_per_seed).resolve(), tracks)
    if not entries:
        raise RuntimeError("No selected entries")

    if args.output_root:
        out_root = Path(args.output_root).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(f"outputs/stage1_retained_oracle_probe_{ts}").resolve()
    (out_root / "summary").mkdir(parents=True, exist_ok=True)

    oracle_rows, probe_rows = [], []

    for e in entries:
        print(f"[diag] {e.track} seed={e.seed}")
        stage1_dir = e.stage1_ckpt.parent
        cfg, raw_cfg, cfg_path = load_cfg(stage1_dir)
        model = BSPARStage1(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(e.stage1_ckpt, map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()

        oracle = oracle_from_json(stage1_dir)
        oracle_rows.append({"track": e.track, "seed": e.seed, "stage1_ckpt": str(e.stage1_ckpt), **oracle})

        train = collect_split(model, cfg, raw_cfg, "train", args.batch_size)
        test = collect_split(model, cfg, raw_cfg, "test", args.batch_size)

        x1_tr, x1_te = train["X_repr"], test["X_repr"]
        x2_tr = np.concatenate([train["X_repr"], train["X_stats"]], axis=1)
        x2_te = np.concatenate([test["X_repr"], test["X_stats"]], axis=1)
        y_tr, y_te = train["y"], test["y"]
        neg_te = test["neg_type"]

        sc1, sc2 = StandardScaler(), StandardScaler()
        x1_tr_n, x1_te_n = sc1.fit_transform(x1_tr), sc1.transform(x1_te)
        x2_tr_n, x2_te_n = sc2.fit_transform(x2_tr), sc2.transform(x2_te)

        clf1 = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
        clf2 = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
        clf1.fit(x1_tr_n, y_tr); clf2.fit(x2_tr_n, y_tr)
        p1 = clf1.predict_proba(x1_te_n)[:, 1]
        p2 = clf2.predict_proba(x2_te_n)[:, 1]

        sig_auc = {k: safe_auc(y_te, v) for k, v in test["signals"].items()}
        weak_keys = ["cat_margin_prob", "aff_margin_prob", "joint_confidence_prod", "joint_margin_sum", "same_aspect_rank_inv", "gap_to_same_aspect_top_inv"]
        weak_best = max(sig_auc[k] for k in weak_keys)

        probe_rows.append({
            "track": e.track,
            "seed": e.seed,
            "config_path": str(cfg_path),
            "stage1_ckpt": str(e.stage1_ckpt),
            "probe1_auc": safe_auc(y_te, p1),
            "probe1_ap": safe_ap(y_te, p1),
            "probe2_auc": safe_auc(y_te, p2),
            "probe2_ap": safe_ap(y_te, p2),
            "signal_auc_pair_score": sig_auc["pair_score"],
            "signal_auc_cat_margin_prob": sig_auc["cat_margin_prob"],
            "signal_auc_aff_margin_prob": sig_auc["aff_margin_prob"],
            "signal_auc_joint_confidence_prod": sig_auc["joint_confidence_prod"],
            "signal_auc_joint_margin_sum": sig_auc["joint_margin_sum"],
            "signal_auc_same_aspect_rank_inv": sig_auc["same_aspect_rank_inv"],
            "signal_auc_gap_to_same_aspect_top_inv": sig_auc["gap_to_same_aspect_top_inv"],
            "probe2_auc_minus_weak_best": safe_auc(y_te, p2) - weak_best,
            "probe2_auc_minus_pair_score": safe_auc(y_te, p2) - sig_auc["pair_score"],
            "probe2_auc_gold_vs_near_miss": subtype_auc(p2, y_te, neg_te, "near_miss"),
            "probe2_auc_gold_vs_category_confused": subtype_auc(p2, y_te, neg_te, "category_confused"),
            "probe2_auc_gold_vs_null_related": subtype_auc(p2, y_te, neg_te, "null_related"),
            "probe2_auc_gold_vs_other": subtype_auc(p2, y_te, neg_te, "other"),
        })

    oracle_csv = out_root / "summary/retained_oracle_upperbound.csv"
    with oracle_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(oracle_rows[0].keys()))
        w.writeheader(); w.writerows(oracle_rows)

    probe_csv = out_root / "summary/frozen_probe_feasibility_per_seed.csv"
    with probe_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
        w.writeheader(); w.writerows(probe_rows)

    group = {"oracle": {}, "probe": {}}
    for tr in sorted({r["track"] for r in oracle_rows}):
        rs = [r for r in oracle_rows if r["track"] == tr]
        vals = [r["oracle_fixable_ratio_over_A_total"] for r in rs]
        group["oracle"][tr] = {
            "num_seeds": len(rs),
            "mean_fixable_ratio_over_A_total": float(np.mean(vals)),
            "mean_oracle_fixable_A3_cat_aff_not_materialized": float(np.mean([r["oracle_fixable_A3_cat_aff_not_materialized"] for r in rs])),
            "mean_oracle_fixable_A1_opinion_only_miss": float(np.mean([r["oracle_fixable_A1_opinion_only_miss"] for r in rs])),
        }
    for tr in sorted({r["track"] for r in probe_rows}):
        rs = [r for r in probe_rows if r["track"] == tr]
        keys = [k for k in probe_rows[0].keys() if k not in {"track", "seed", "config_path", "stage1_ckpt"}]
        group["probe"][tr] = {"num_seeds": len(rs), "mean": {k: float(np.mean([r[k] for r in rs])) for k in keys}}

    group_json = out_root / "summary/frozen_probe_group_summary.json"
    group_json.write_text(json.dumps(group, indent=2, ensure_ascii=False), encoding="utf-8")

    md = out_root / "summary/retained_oracle_probe_takeaways.md"
    lines = [
        "# Retained-only Oracle + Frozen Probe Takeaways",
        "",
        "## Oracle upper-bound",
    ]
    for tr, v in group["oracle"].items():
        lines.append(f"- {tr}: fixable_ratio_over_A_total={v['mean_fixable_ratio_over_A_total']:.4f}, fixable_A3_cat_aff={v['mean_oracle_fixable_A3_cat_aff_not_materialized']:.2f}, fixable_A1_opinion_only_miss={v['mean_oracle_fixable_A1_opinion_only_miss']:.2f}")
    lines += ["", "## Frozen-probe feasibility"]
    for tr, v in group["probe"].items():
        m = v["mean"]
        lines.append(f"- {tr}: probe2_auc={m['probe2_auc']:.4f}, weak_best_gap={m['probe2_auc_minus_weak_best']:.4f}, vs_pair_score={m['probe2_auc_minus_pair_score']:.4f}")
    lines += ["", "Decision hint: move to RDI-v1 only if oracle headroom is meaningful and probe2 clearly beats weak logits."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {oracle_csv}")
    print(f"Wrote: {probe_csv}")
    print(f"Wrote: {group_json}")
    print(f"Wrote: {md}")


if __name__ == "__main__":
    main()

