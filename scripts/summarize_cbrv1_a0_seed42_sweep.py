"""Summarize CBR-v1 seed42 sweep against baseline and strongest Aff-ACR only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CBR-v1 seed42 sweep")
    parser.add_argument("--cbr_output_root", required=True)
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
        "boundary_active_positive_ratio": to_float(
            metrics.get("boundary_active_positive_ratio")
        ),
        "avg_cutoff_gap": to_float(metrics.get("avg_cutoff_gap")),
        "num_samples_with_active_boundary_loss": to_float(
            metrics.get("num_samples_with_active_boundary_loss")
        ),
        "cbr_loss_mean": to_float(metrics.get("cbr_loss_mean")),
        "stage1_dir": str(stage1_dir),
    }


def row_from_baseline(path: Path, seed: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
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
                    "run_dir": r["run_dir"],
                    "stage1_ckpt": r["stage1_ckpt"],
                }
            )
            return row
    raise RuntimeError(f"Seed {seed} not found in {path}")


def row_from_affacr(path: Path, seed: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
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
                    "run_dir": r["run_dir"],
                    "stage1_ckpt": r["stage1_ckpt"],
                }
            )
            return row
    raise RuntimeError(f"Seed {seed} not found in {path}")


def row_from_manifest(manifest_path: Path) -> dict:
    m = load_json(manifest_path)
    row = collect_stage1(m["stage1_ckpt"])
    eval_dev = load_json(Path(m["eval_dev_json"]))
    eval_test = load_json(Path(m["eval_test_json"]))
    row.update(
        {
            "track": f"cbr_v1_lam{m['cbr_v1_lambda']}_m{m['cbr_v1_margin']}",
            "seed": int(m["seed"]),
            "dev_quad_f1": to_float(eval_dev.get("quad_f1")),
            "test_quad_f1": to_float(eval_test.get("quad_f1")),
            "run_dir": m["run_dir"],
            "stage1_ckpt": m["stage1_ckpt"],
            "cbr_v1_lambda": float(m["cbr_v1_lambda"]),
            "cbr_v1_margin": float(m["cbr_v1_margin"]),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    cbr_root = Path(args.cbr_output_root).resolve()
    summary_dir = cbr_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    base_row = row_from_baseline(Path(args.baseline_summary).resolve(), seed)
    aff_row = row_from_affacr(Path(args.affacr_per_seed).resolve(), seed)
    cbr_rows = [row_from_manifest(p) for p in sorted(cbr_root.glob("runs/*/run_manifest.json"))]
    cbr_rows = [r for r in cbr_rows if int(r["seed"]) == seed]
    if not cbr_rows:
        raise RuntimeError(f"No CBR run manifests found for seed={seed}")

    rows = [base_row, aff_row] + sorted(
        cbr_rows, key=lambda r: (r.get("cbr_v1_lambda", 0.0), r.get("cbr_v1_margin", 0.0))
    )

    fields = [
        "track",
        "seed",
        "cbr_v1_lambda",
        "cbr_v1_margin",
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
        "boundary_active_positive_ratio",
        "avg_cutoff_gap",
        "num_samples_with_active_boundary_loss",
        "cbr_loss_mean",
        "run_dir",
        "stage1_ckpt",
    ]
    per_seed_csv = summary_dir / "cbrv1_seed42_sweep_compare.csv"
    write_csv(per_seed_csv, rows, fields)

    # Delta table against strongest affacr
    delta_rows = []
    for r in cbr_rows:
        d = {
            "track": r["track"],
            "seed": seed,
            "cbr_v1_lambda": r["cbr_v1_lambda"],
            "cbr_v1_margin": r["cbr_v1_margin"],
        }
        for k in [
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
            "boundary_active_positive_ratio",
            "avg_cutoff_gap",
            "num_samples_with_active_boundary_loss",
        ]:
            rv = r.get(k)
            av = aff_row.get(k)
            d[f"delta_vs_affacr_{k}"] = (
                float(rv) - float(av) if rv is not None and av is not None else None
            )

        cond_1 = (d.get("delta_vs_affacr_A3_topn_drop") is not None and d["delta_vs_affacr_A3_topn_drop"] < 0)
        cond_2 = (
            d.get("delta_vs_affacr_gold_pair_mean_rank") is not None and
            d.get("delta_vs_affacr_gold_pair_median_rank") is not None and
            d["delta_vs_affacr_gold_pair_mean_rank"] <= 0 and
            d["delta_vs_affacr_gold_pair_median_rank"] <= 0
        )
        cond_3 = (
            d.get("delta_vs_affacr_sample_has_positive_after_retention_ratio") is not None and
            d["delta_vs_affacr_sample_has_positive_after_retention_ratio"] >= 0
        )
        ok_count = int(cond_1) + int(cond_2) + int(cond_3)
        non_harm = (
            d.get("delta_vs_affacr_dev_quad_f1") is not None and
            d.get("delta_vs_affacr_test_quad_f1") is not None and
            d["delta_vs_affacr_dev_quad_f1"] >= -0.002 and
            d["delta_vs_affacr_test_quad_f1"] >= -0.002
        )
        d["cbrv1_continue_criteria_ok"] = bool(ok_count >= 2 and non_harm)
        d["criteria_hit_count"] = ok_count
        delta_rows.append(d)

    delta_fields = list(delta_rows[0].keys()) if delta_rows else []
    delta_csv = summary_dir / "cbrv1_seed42_sweep_deltas_vs_affacr.csv"
    write_csv(delta_csv, delta_rows, delta_fields)

    md_lines = [
        "# CBR-v1 Seed42 Sweep Takeaways",
        "",
        f"- strongest affacr test: {aff_row['test_quad_f1']:.6f}",
        "",
    ]
    for d in delta_rows:
        md_lines.append(
            f"- {d['track']}: "
            f"test_delta={d['delta_vs_affacr_test_quad_f1']:+.6f}, "
            f"A3_topn_drop_delta={d['delta_vs_affacr_A3_topn_drop']:+.1f}, "
            f"criteria_hit={d['criteria_hit_count']}/3, "
            f"continue={d['cbrv1_continue_criteria_ok']}"
        )
    md_path = summary_dir / "cbrv1_seed42_sweep_takeaways.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {per_seed_csv}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
