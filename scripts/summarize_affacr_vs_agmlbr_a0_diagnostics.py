"""Summarize baseline AGML-BR+A0 vs AGML-BR+A0+Aff-ACR diagnostics (4-seed)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diagnostic comparison tables for baseline vs Aff-ACR only"
    )
    parser.add_argument(
        "--baseline_summary",
        required=True,
        help="Path to agmlbr_a0_multiseed_summary.csv",
    )
    parser.add_argument(
        "--affacr_per_seed",
        required=True,
        help="Path to affacr_a0_4seed_per_seed.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output summary directory",
    )
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


def outranker_ratio(diag: dict, key: str) -> float | None:
    d = diag.get("first_outranker_type_ratio", {}).get(key)
    if isinstance(d, dict):
        return to_float(d.get("ratio"))
    return None


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_stage1_metrics(stage1_ckpt: str) -> dict:
    stage1_dir = Path(stage1_ckpt).resolve().parent
    breakdown = load_json(stage1_dir / "stage1_a_breakdown_test.json")
    diag = load_json(stage1_dir / "best_stage1_a3_diagnostics.json")
    stage1_metrics = load_json(stage1_dir / "stage1_metrics.json")
    best_dev = stage1_metrics.get("best_dev_metrics", {})

    a1_side_path = stage1_dir / "a1_span_side_split_test.json"
    a1_side = load_json(a1_side_path) if a1_side_path.exists() else {}
    a1_opinion_only = safe_get(a1_side, "A1_split_counts", "opinion_only_miss")

    return {
        "A1": int(safe_get(breakdown, "A_breakdown_counts", "A1", default=0)),
        "A3": int(safe_get(breakdown, "A_breakdown_counts", "A3", default=0)),
        "A_total": int(breakdown.get("total_A_miss", 0)),
        "A3_topn_drop": int(safe_get(breakdown, "A3_subtypes", "topn_drop", default=0)),
        "A3_cat_aff_not_materialized": int(
            safe_get(breakdown, "A3_subtypes", "cat_aff_not_materialized", default=0)
        ),
        "A1_opinion_only_miss": (
            int(a1_opinion_only) if a1_opinion_only is not None else None
        ),
        "sample_has_positive_after_retention_ratio": to_float(
            diag.get("sample_has_positive_after_retention_ratio")
        ),
        "gold_pair_mean_rank": to_float(safe_get(diag, "gold_pair_rank", "mean")),
        "gold_pair_median_rank": to_float(safe_get(diag, "gold_pair_rank", "median")),
        "score_topn_minus_gold_mean": to_float(
            safe_get(diag, "score_topn_minus_gold_pair", "mean")
        ),
        "score_topn_minus_gold_median": to_float(
            safe_get(diag, "score_topn_minus_gold_pair", "median")
        ),
        "first_outranker_null_ratio": outranker_ratio(diag, "NULL"),
        "first_outranker_near_miss_ratio": outranker_ratio(diag, "near_miss"),
        "first_outranker_other_ratio": outranker_ratio(diag, "other"),
        "gold_pair_recall_pair_space": to_float(best_dev.get("gold_pair_recall_pair_space")),
        "gold_pair_recall_after_gate": to_float(best_dev.get("gold_pair_recall_after_gate")),
        "avg_pairs_into_stage2": to_float(best_dev.get("avg_pairs_into_stage2")),
        "avg_candidates_into_stage2": to_float(best_dev.get("avg_candidates_into_stage2")),
        "stage1_dir": str(stage1_dir),
    }


def load_baseline(summary_csv: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            m = collect_stage1_metrics(r["stage1_ckpt"])
            m.update(
                {
                    "dev_quad_f1": to_float(r["agmlbr_A0_dev"]),
                    "test_quad_f1": to_float(r["agmlbr_A0_test"]),
                    "stage1_ckpt": r["stage1_ckpt"],
                    "run_dir": r["run_dir"],
                }
            )
            out[seed] = m
    return out


def load_affacr(per_seed_csv: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with per_seed_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            m = collect_stage1_metrics(r["stage1_ckpt"])
            m.update(
                {
                    "dev_quad_f1": to_float(r["dev_quad_f1"]),
                    "test_quad_f1": to_float(r["test_quad_f1"]),
                    "stage1_ckpt": r["stage1_ckpt"],
                    "run_dir": r["run_dir"],
                }
            )
            out[seed] = m
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def stats(vals: list[float | None]) -> tuple[float | None, float | None]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None, None
    return float(mean(xs)), float(pstdev(xs))


def main() -> None:
    args = parse_args()
    baseline_map = load_baseline(Path(args.baseline_summary).resolve())
    affacr_map = load_affacr(Path(args.affacr_per_seed).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = sorted(set(baseline_map.keys()) & set(affacr_map.keys()))
    if not seeds:
        raise RuntimeError("No overlapping seeds between baseline and affacr inputs.")

    metric_fields = [
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
    ]

    per_seed_rows = []
    for seed in seeds:
        b = baseline_map[seed]
        a = affacr_map[seed]
        row = {
            "seed": seed,
            "baseline_run_dir": b["run_dir"],
            "baseline_stage1_ckpt": b["stage1_ckpt"],
            "affacr_run_dir": a["run_dir"],
            "affacr_stage1_ckpt": a["stage1_ckpt"],
        }
        for m in metric_fields:
            bv = b.get(m)
            av = a.get(m)
            row[f"baseline_{m}"] = bv
            row[f"affacr_{m}"] = av
            row[f"delta_{m}"] = (
                (av - bv) if (av is not None and bv is not None) else None
            )
        per_seed_rows.append(row)

    per_seed_fields = ["seed"]
    for m in metric_fields:
        per_seed_fields.extend([f"baseline_{m}", f"affacr_{m}", f"delta_{m}"])
    per_seed_fields.extend(
        ["baseline_run_dir", "baseline_stage1_ckpt", "affacr_run_dir", "affacr_stage1_ckpt"]
    )

    per_seed_csv = output_dir / "affacr_vs_agmlbr_a0_diagnostic_per_seed.csv"
    write_csv(per_seed_csv, per_seed_rows, per_seed_fields)

    # mean/std JSON + key delta rows
    mean_std = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "metrics": {},
        "missing_metrics": {
            "baseline": {},
            "affacr": {},
            "delta": {},
        },
    }
    delta_rows = []

    for m in metric_fields:
        b_vals = [baseline_map[s].get(m) for s in seeds]
        a_vals = [affacr_map[s].get(m) for s in seeds]
        d_vals = [
            (affacr_map[s].get(m) - baseline_map[s].get(m))
            if (affacr_map[s].get(m) is not None and baseline_map[s].get(m) is not None)
            else None
            for s in seeds
        ]

        b_mean, b_std = stats(b_vals)
        a_mean, a_std = stats(a_vals)
        d_mean, d_std = stats(d_vals)

        mean_std["metrics"][m] = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "affacr_mean": a_mean,
            "affacr_std": a_std,
            "delta_mean": d_mean,
            "delta_std": d_std,
            "baseline_count": sum(v is not None for v in b_vals),
            "affacr_count": sum(v is not None for v in a_vals),
            "delta_count": sum(v is not None for v in d_vals),
        }
        mean_std["missing_metrics"]["baseline"][m] = sum(v is None for v in b_vals)
        mean_std["missing_metrics"]["affacr"][m] = sum(v is None for v in a_vals)
        mean_std["missing_metrics"]["delta"][m] = sum(v is None for v in d_vals)

        delta_rows.append(
            {
                "metric": m,
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "affacr_mean": a_mean,
                "affacr_std": a_std,
                "delta_mean": d_mean,
                "delta_std": d_std,
                "delta_nonneg_count": sum((v is not None and v >= 0) for v in d_vals),
                "delta_total_count": sum(v is not None for v in d_vals),
            }
        )

    mean_std_json = output_dir / "affacr_vs_agmlbr_a0_diagnostic_mean_std.json"
    mean_std_json.write_text(json.dumps(mean_std, indent=2, ensure_ascii=False), encoding="utf-8")

    key_delta_csv = output_dir / "affacr_vs_agmlbr_a0_key_deltas.csv"
    write_csv(
        key_delta_csv,
        delta_rows,
        [
            "metric",
            "baseline_mean",
            "baseline_std",
            "affacr_mean",
            "affacr_std",
            "delta_mean",
            "delta_std",
            "delta_nonneg_count",
            "delta_total_count",
        ],
    )

    # Optional concise takeaways.
    takeaways = []
    takeaways.append("# Aff-ACR vs AGML-BR+A0 (4-seed) Takeaways")
    takeaways.append("")
    for m in [
        "test_quad_f1",
        "dev_quad_f1",
        "A_total",
        "A1",
        "A1_opinion_only_miss",
        "A3",
        "A3_cat_aff_not_materialized",
        "score_topn_minus_gold_mean",
    ]:
        item = mean_std["metrics"].get(m, {})
        if item.get("delta_mean") is None:
            continue
        takeaways.append(
            f"- `{m}`: baseline {item['baseline_mean']:.6f} -> affacr {item['affacr_mean']:.6f} "
            f"(delta {item['delta_mean']:+.6f}, n={item['delta_count']})"
        )

    takeaways.append("")
    takeaways.append("## Verified Findings")
    takeaways.append("- Metrics with full 4/4 seed coverage are treated as verified in this report.")
    for m, miss in mean_std["missing_metrics"]["delta"].items():
        if miss == 0:
            takeaways.append(f"- `{m}`")

    prelim = [m for m, miss in mean_std["missing_metrics"]["delta"].items() if miss > 0]
    if prelim:
        takeaways.append("")
        takeaways.append("## Preliminary Findings")
        takeaways.append("- The following metrics have missing seed-level values:")
        for m in prelim:
            takeaways.append(f"- `{m}`")

    md_path = output_dir / "affacr_vs_agmlbr_a0_takeaways.md"
    md_path.write_text("\n".join(takeaways) + "\n", encoding="utf-8")

    print(f"Wrote: {per_seed_csv}")
    print(f"Wrote: {mean_std_json}")
    print(f"Wrote: {key_delta_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
