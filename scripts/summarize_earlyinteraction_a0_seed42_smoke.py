"""Summarize seed42 smoke comparison:
baseline AGML-BR+A0 vs strongest Aff-ACR only vs early-interaction variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize early-interaction smoke")
    parser.add_argument(
        "--early_output_root",
        required=True,
        help="Output root of run_earlyinteraction_a0_seed42_smoke.py",
    )
    parser.add_argument(
        "--baseline_summary",
        default="outputs/stage2_e2e_agmlbr_a0_multiseed_20260317_095403/summary/agmlbr_a0_multiseed_summary.csv",
    )
    parser.add_argument(
        "--affacr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/summary/affacr_a0_4seed_per_seed.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return float(v)


def outranker(diag: dict, key: str):
    d = diag.get("first_outranker_type_ratio", {}).get(key)
    if isinstance(d, dict):
        return to_float(d.get("ratio"))
    return None


def collect_stage1(stage1_ckpt: str) -> dict:
    stage1_dir = Path(stage1_ckpt).resolve().parent
    breakdown = load_json(stage1_dir / "stage1_a_breakdown_test.json")
    diag = load_json(stage1_dir / "best_stage1_a3_diagnostics.json")
    metrics = load_json(stage1_dir / "stage1_metrics.json").get("best_dev_metrics", {})
    a1_side_path = stage1_dir / "a1_span_side_split_test.json"
    a1_side = load_json(a1_side_path) if a1_side_path.exists() else {}

    return {
        "A1": int(breakdown["A_breakdown_counts"]["A1"]),
        "A1_opinion_only_miss": a1_side.get("A1_split_counts", {}).get("opinion_only_miss"),
        "A3": int(breakdown["A_breakdown_counts"]["A3"]),
        "A3_topn_drop": int(breakdown["A3_subtypes"]["topn_drop"]),
        "A3_cat_aff_not_materialized": int(breakdown["A3_subtypes"]["cat_aff_not_materialized"]),
        "A_total": int(breakdown["total_A_miss"]),
        "sample_has_positive_after_retention_ratio": to_float(
            diag.get("sample_has_positive_after_retention_ratio")
        ),
        "gold_pair_mean_rank": to_float(diag.get("gold_pair_rank", {}).get("mean")),
        "gold_pair_median_rank": to_float(diag.get("gold_pair_rank", {}).get("median")),
        "score_topn_minus_gold_mean": to_float(
            diag.get("score_topn_minus_gold_pair", {}).get("mean")
        ),
        "score_topn_minus_gold_median": to_float(
            diag.get("score_topn_minus_gold_pair", {}).get("median")
        ),
        "first_outranker_null_ratio": outranker(diag, "NULL"),
        "first_outranker_near_miss_ratio": outranker(diag, "near_miss"),
        "first_outranker_other_ratio": outranker(diag, "other"),
        "gold_pair_recall_pair_space": to_float(metrics.get("gold_pair_recall_pair_space")),
        "gold_pair_recall_after_gate": to_float(metrics.get("gold_pair_recall_after_gate")),
        "avg_pairs_into_stage2": to_float(metrics.get("avg_pairs_into_stage2")),
        "avg_candidates_into_stage2": to_float(metrics.get("avg_candidates_into_stage2")),
        "stage1_dir": str(stage1_dir),
    }


def row_from_baseline(baseline_summary: Path, seed: int) -> dict:
    with baseline_summary.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) != seed:
                continue
            row = collect_stage1(r["stage1_ckpt"])
            row.update(
                {
                    "track": "baseline_agmlbr_a0",
                    "seed": seed,
                    "dev_quad_f1": to_float(r["agmlbr_A0_dev"]),
                    "test_quad_f1": to_float(r["agmlbr_A0_test"]),
                    "stage1_ckpt": r["stage1_ckpt"],
                    "run_dir": r["run_dir"],
                }
            )
            return row
    raise RuntimeError(f"Seed {seed} not found in baseline summary: {baseline_summary}")


def row_from_affacr(affacr_csv: Path, seed: int) -> dict:
    with affacr_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) != seed:
                continue
            row = collect_stage1(r["stage1_ckpt"])
            row.update(
                {
                    "track": "strongest_affacr_only",
                    "seed": seed,
                    "dev_quad_f1": to_float(r["dev_quad_f1"]),
                    "test_quad_f1": to_float(r["test_quad_f1"]),
                    "stage1_ckpt": r["stage1_ckpt"],
                    "run_dir": r["run_dir"],
                }
            )
            return row
    raise RuntimeError(f"Seed {seed} not found in affacr summary: {affacr_csv}")


def row_from_early(early_root: Path, seed: int) -> dict:
    manifests = list(early_root.glob("runs/*/run_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No run_manifest.json under: {early_root / 'runs'}")
    m = load_json(manifests[0])
    if int(m["seed"]) != seed:
        raise RuntimeError(f"Early run seed mismatch: expected {seed}, got {m['seed']}")
    eval_dev = load_json(Path(m["eval_dev_json"]))
    eval_test = load_json(Path(m["eval_test_json"]))
    row = collect_stage1(m["stage1_ckpt"])
    row.update(
        {
            "track": "early_interaction",
            "seed": seed,
            "dev_quad_f1": to_float(eval_dev.get("quad_f1")),
            "test_quad_f1": to_float(eval_test.get("quad_f1")),
            "stage1_ckpt": m["stage1_ckpt"],
            "run_dir": m["run_dir"],
        }
    )
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    early_root = Path(args.early_output_root).resolve()
    summary_dir = early_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    baseline_row = row_from_baseline(Path(args.baseline_summary).resolve(), seed)
    affacr_row = row_from_affacr(Path(args.affacr_per_seed).resolve(), seed)
    early_row = row_from_early(early_root, seed)

    rows = [baseline_row, affacr_row, early_row]
    fields = [
        "track",
        "seed",
        "dev_quad_f1",
        "test_quad_f1",
        "A1",
        "A1_opinion_only_miss",
        "A3",
        "A3_topn_drop",
        "A3_cat_aff_not_materialized",
        "A_total",
        "sample_has_positive_after_retention_ratio",
        "gold_pair_mean_rank",
        "gold_pair_median_rank",
        "score_topn_minus_gold_mean",
        "score_topn_minus_gold_median",
        "first_outranker_null_ratio",
        "first_outranker_near_miss_ratio",
        "first_outranker_other_ratio",
        "gold_pair_recall_pair_space",
        "gold_pair_recall_after_gate",
        "avg_pairs_into_stage2",
        "avg_candidates_into_stage2",
        "run_dir",
        "stage1_ckpt",
    ]
    compare_csv = summary_dir / "earlyinteraction_smoke_compare_seed42.csv"
    write_csv(compare_csv, rows, fields)

    # delta table: early vs baseline and early vs strongest affacr
    delta_rows = []
    for ref in [baseline_row, affacr_row]:
        delta = {"from_track": ref["track"], "to_track": early_row["track"]}
        for k in fields:
            if k in {"track", "seed", "run_dir", "stage1_ckpt"}:
                continue
            rv = ref.get(k)
            ev = early_row.get(k)
            if rv is None or ev is None:
                delta[f"delta_{k}"] = None
            else:
                delta[f"delta_{k}"] = float(ev) - float(rv)
        delta_rows.append(delta)

    delta_fields = ["from_track", "to_track"] + [
        f"delta_{k}" for k in fields if k not in {"track", "seed", "run_dir", "stage1_ckpt"}
    ]
    delta_csv = summary_dir / "earlyinteraction_smoke_key_deltas_seed42.csv"
    write_csv(delta_csv, delta_rows, delta_fields)

    takeaways = [
        "# Early Interaction Smoke (Seed42)",
        "",
        f"- baseline test: {baseline_row['test_quad_f1']:.6f}",
        f"- strongest affacr test: {affacr_row['test_quad_f1']:.6f}",
        f"- early interaction test: {early_row['test_quad_f1']:.6f}",
        f"- delta vs baseline: {early_row['test_quad_f1'] - baseline_row['test_quad_f1']:+.6f}",
        f"- delta vs strongest affacr: {early_row['test_quad_f1'] - affacr_row['test_quad_f1']:+.6f}",
    ]
    md = summary_dir / "earlyinteraction_smoke_takeaways_seed42.md"
    md.write_text("\n".join(takeaways) + "\n", encoding="utf-8")

    print(f"Wrote: {compare_csv}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {md}")


if __name__ == "__main__":
    main()
