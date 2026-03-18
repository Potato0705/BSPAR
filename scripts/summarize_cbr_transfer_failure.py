"""Boundary->E2E transfer diagnosis for Aff-ACR only vs CBR-v1 (4-seed).

Outputs:
- summary/cbr_vs_affacr_transfer_diagnosis_per_seed.csv
- summary/cbr_vs_affacr_transfer_diagnosis_mean_std.json
- summary/cbr_vs_affacr_transfer_takeaways.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CBR transfer failure diagnosis")
    parser.add_argument(
        "--affacr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/summary/affacr_a0_4seed_per_seed.csv",
    )
    parser.add_argument(
        "--cbr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_20260318_082343/summary/cbrv1_a0_4seed_per_seed.csv",
    )
    parser.add_argument(
        "--baseline_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_multiseed_20260317_095403/summary/agmlbr_a0_multiseed_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <cbr_per_seed parent>",
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


def to_int(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return int(v)


def outranker_ratio(diag: dict, key: str):
    node = diag.get("first_outranker_type_ratio", {}).get(key)
    if isinstance(node, dict):
        return to_float(node.get("ratio"))
    return None


def stage1_extra(stage1_ckpt: str) -> dict:
    stage1_dir = Path(stage1_ckpt).resolve().parent
    diag = load_json(stage1_dir / "best_stage1_a3_diagnostics.json")
    m = load_json(stage1_dir / "stage1_metrics.json").get("best_dev_metrics", {})
    a1_split_path = stage1_dir / "a1_span_side_split_test.json"
    a1_split = load_json(a1_split_path) if a1_split_path.exists() else {}
    return {
        "A1_opinion_only_miss": to_int(a1_split.get("A1_split_counts", {}).get("opinion_only_miss")),
        "gold_pair_mean_rank": to_float(diag.get("gold_pair_rank", {}).get("mean")),
        "gold_pair_median_rank": to_float(diag.get("gold_pair_rank", {}).get("median")),
        "score_topn_minus_gold_mean": to_float(
            diag.get("score_topn_minus_gold_pair", {}).get("mean")
        ),
        "score_topn_minus_gold_median": to_float(
            diag.get("score_topn_minus_gold_pair", {}).get("median")
        ),
        "sample_has_positive_after_retention_ratio": to_float(
            diag.get("sample_has_positive_after_retention_ratio")
        ),
        "first_outranker_null_ratio": outranker_ratio(diag, "NULL"),
        "first_outranker_near_miss_ratio": outranker_ratio(diag, "near_miss"),
        "first_outranker_other_ratio": outranker_ratio(diag, "other"),
        "boundary_active_positive_ratio": to_float(m.get("boundary_active_positive_ratio")),
        "avg_cutoff_gap": to_float(m.get("avg_cutoff_gap")),
        "num_samples_with_active_boundary_loss": to_float(
            m.get("num_samples_with_active_boundary_loss")
        ),
    }


def load_affacr_map(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            extra = stage1_extra(r["stage1_ckpt"])
            out[seed] = {
                "seed": seed,
                "track": "affacr_only",
                "stage1_ckpt": r["stage1_ckpt"],
                "run_dir": r["run_dir"],
                "dev_quad_f1": to_float(r["dev_quad_f1"]),
                "test_quad_f1": to_float(r["test_quad_f1"]),
                "A1": to_int(r["A1"]),
                "A3": to_int(r["A3"]),
                "A3_topn_drop": to_int(r["A3_topn_drop"]),
                "A3_cat_aff_not_materialized": to_int(r["A3_cat_aff_not_materialized"]),
                "A_total": to_int(r["A_total"]),
                "sample_has_positive_after_retention_ratio": to_float(
                    r["sample_has_positive_after_retention_ratio"]
                ),
                "gold_pair_recall_pair_space": to_float(r.get("gold_pair_recall_pair_space")),
                "gold_pair_recall_after_gate": to_float(r.get("gold_pair_recall_after_gate")),
                "avg_pairs_into_stage2": to_float(r.get("avg_pairs_into_stage2")),
                "avg_candidates_into_stage2": to_float(r.get("avg_candidates_into_stage2")),
                **extra,
            }
    return out


def load_cbr_map(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            out[seed] = {
                "seed": seed,
                "track": "cbr_v1",
                "cbr_v1_lambda": to_float(r["cbr_v1_lambda"]),
                "cbr_v1_margin": to_float(r["cbr_v1_margin"]),
                "stage1_ckpt": r["stage1_ckpt"],
                "run_dir": r["run_dir"],
                "dev_quad_f1": to_float(r["dev_quad_f1"]),
                "test_quad_f1": to_float(r["test_quad_f1"]),
                "A1": to_int(r["A1"]),
                "A1_opinion_only_miss": to_int(r["A1_opinion_only_miss"]),
                "A3": to_int(r["A3"]),
                "A3_topn_drop": to_int(r["A3_topn_drop"]),
                "A3_cat_aff_not_materialized": to_int(r["A3_cat_aff_not_materialized"]),
                "A_total": to_int(r["A_total"]),
                "sample_has_positive_after_retention_ratio": to_float(
                    r["sample_has_positive_after_retention_ratio"]
                ),
                "gold_pair_mean_rank": to_float(r["gold_pair_mean_rank"]),
                "gold_pair_median_rank": to_float(r["gold_pair_median_rank"]),
                "score_topn_minus_gold_mean": to_float(r["score_topn_minus_gold_mean"]),
                "score_topn_minus_gold_median": to_float(r["score_topn_minus_gold_median"]),
                "first_outranker_null_ratio": to_float(r["first_outranker_null_ratio"]),
                "first_outranker_near_miss_ratio": to_float(r["first_outranker_near_miss_ratio"]),
                "first_outranker_other_ratio": to_float(r["first_outranker_other_ratio"]),
                "gold_pair_recall_pair_space": to_float(r.get("gold_pair_recall_pair_space")),
                "gold_pair_recall_after_gate": to_float(r.get("gold_pair_recall_after_gate")),
                "avg_pairs_into_stage2": to_float(r.get("avg_pairs_into_stage2")),
                "avg_candidates_into_stage2": to_float(r.get("avg_candidates_into_stage2")),
                "boundary_active_positive_ratio": to_float(r.get("boundary_active_positive_ratio")),
                "avg_cutoff_gap": to_float(r.get("avg_cutoff_gap")),
                "num_samples_with_active_boundary_loss": to_float(
                    r.get("num_samples_with_active_boundary_loss")
                ),
            }
    return out


def load_baseline_map(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            extra = stage1_extra(r["stage1_ckpt"])
            out[seed] = {
                "seed": seed,
                "track": "baseline_agmlbr_a0",
                "stage1_ckpt": r["stage1_ckpt"],
                "run_dir": r["run_dir"],
                "dev_quad_f1": to_float(r["agmlbr_A0_dev"]),
                "test_quad_f1": to_float(r["agmlbr_A0_test"]),
                **extra,
            }
    return out


def delta(cur: dict, ref: dict, key: str):
    cv = cur.get(key)
    rv = ref.get(key)
    if cv is None or rv is None:
        return None
    return float(cv) - float(rv)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(rows: list[dict], fields: list[str], prefix: str) -> dict:
    out = {}
    for k in fields:
        vals = [to_float(r.get(k)) for r in rows if r.get(k) is not None]
        out[f"{prefix}_{k}_mean"] = float(mean(vals)) if vals else None
        out[f"{prefix}_{k}_std"] = float(pstdev(vals)) if vals else None
    return out


def main() -> None:
    args = parse_args()
    aff_map = load_affacr_map(Path(args.affacr_per_seed).resolve())
    cbr_map = load_cbr_map(Path(args.cbr_per_seed).resolve())
    base_map = load_baseline_map(Path(args.baseline_per_seed).resolve())

    seeds = sorted(set(aff_map.keys()) & set(cbr_map.keys()))
    if not seeds:
        raise RuntimeError("No shared seeds between affacr and cbr outputs")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = Path(args.cbr_per_seed).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    transfer_fail_seeds = []
    for s in seeds:
        aff = aff_map[s]
        cbr = cbr_map[s]
        row = {
            "seed": s,
            "baseline_dev_quad_f1": base_map.get(s, {}).get("dev_quad_f1"),
            "baseline_test_quad_f1": base_map.get(s, {}).get("test_quad_f1"),
            "affacr_dev_quad_f1": aff["dev_quad_f1"],
            "affacr_test_quad_f1": aff["test_quad_f1"],
            "cbr_dev_quad_f1": cbr["dev_quad_f1"],
            "cbr_test_quad_f1": cbr["test_quad_f1"],
            "delta_dev_quad_f1": delta(cbr, aff, "dev_quad_f1"),
            "delta_test_quad_f1": delta(cbr, aff, "test_quad_f1"),
        }

        core_keys = [
            "A1",
            "A1_opinion_only_miss",
            "A3",
            "A3_topn_drop",
            "A3_cat_aff_not_materialized",
            "A_total",
            "gold_pair_mean_rank",
            "gold_pair_median_rank",
            "score_topn_minus_gold_mean",
            "score_topn_minus_gold_median",
            "sample_has_positive_after_retention_ratio",
            "first_outranker_null_ratio",
            "first_outranker_near_miss_ratio",
            "first_outranker_other_ratio",
            "boundary_active_positive_ratio",
            "avg_cutoff_gap",
            "num_samples_with_active_boundary_loss",
        ]
        for k in core_keys:
            row[f"affacr_{k}"] = aff.get(k)
            row[f"cbr_{k}"] = cbr.get(k)
            row[f"delta_{k}"] = delta(cbr, aff, k)

        # Boundary-positive indicators
        cond_topn = row["delta_A3_topn_drop"] is not None and row["delta_A3_topn_drop"] < 0
        cond_rank = (
            row["delta_gold_pair_mean_rank"] is not None
            and row["delta_gold_pair_median_rank"] is not None
            and row["delta_gold_pair_mean_rank"] <= 0
            and row["delta_gold_pair_median_rank"] <= 0
        )
        cond_cov = (
            row["delta_sample_has_positive_after_retention_ratio"] is not None
            and row["delta_sample_has_positive_after_retention_ratio"] >= 0
        )
        row["boundary_positive_hit_count"] = int(cond_topn) + int(cond_rank) + int(cond_cov)
        row["boundary_positive"] = row["boundary_positive_hit_count"] >= 2
        row["test_nonpositive"] = (
            row["delta_test_quad_f1"] is not None and row["delta_test_quad_f1"] <= 0
        )
        row["boundary_to_e2e_transfer_break"] = bool(
            row["boundary_positive"] and row["test_nonpositive"]
        )
        if row["boundary_to_e2e_transfer_break"]:
            transfer_fail_seeds.append(s)

        per_seed.append(row)

    per_seed_fields = [
        "seed",
        "baseline_dev_quad_f1",
        "baseline_test_quad_f1",
        "affacr_dev_quad_f1",
        "affacr_test_quad_f1",
        "cbr_dev_quad_f1",
        "cbr_test_quad_f1",
        "affacr_A1",
        "cbr_A1",
        "delta_A1",
        "affacr_A1_opinion_only_miss",
        "cbr_A1_opinion_only_miss",
        "delta_A1_opinion_only_miss",
        "affacr_A3",
        "cbr_A3",
        "delta_A3",
        "affacr_A3_topn_drop",
        "cbr_A3_topn_drop",
        "delta_A3_topn_drop",
        "affacr_A3_cat_aff_not_materialized",
        "cbr_A3_cat_aff_not_materialized",
        "delta_A3_cat_aff_not_materialized",
        "affacr_A_total",
        "cbr_A_total",
        "delta_A_total",
        "affacr_gold_pair_mean_rank",
        "cbr_gold_pair_mean_rank",
        "delta_gold_pair_mean_rank",
        "affacr_gold_pair_median_rank",
        "cbr_gold_pair_median_rank",
        "delta_gold_pair_median_rank",
        "affacr_score_topn_minus_gold_mean",
        "cbr_score_topn_minus_gold_mean",
        "delta_score_topn_minus_gold_mean",
        "affacr_score_topn_minus_gold_median",
        "cbr_score_topn_minus_gold_median",
        "delta_score_topn_minus_gold_median",
        "affacr_sample_has_positive_after_retention_ratio",
        "cbr_sample_has_positive_after_retention_ratio",
        "delta_sample_has_positive_after_retention_ratio",
        "affacr_first_outranker_null_ratio",
        "cbr_first_outranker_null_ratio",
        "delta_first_outranker_null_ratio",
        "affacr_first_outranker_near_miss_ratio",
        "cbr_first_outranker_near_miss_ratio",
        "delta_first_outranker_near_miss_ratio",
        "affacr_first_outranker_other_ratio",
        "cbr_first_outranker_other_ratio",
        "delta_first_outranker_other_ratio",
        "affacr_boundary_active_positive_ratio",
        "cbr_boundary_active_positive_ratio",
        "delta_boundary_active_positive_ratio",
        "affacr_avg_cutoff_gap",
        "cbr_avg_cutoff_gap",
        "delta_avg_cutoff_gap",
        "affacr_num_samples_with_active_boundary_loss",
        "cbr_num_samples_with_active_boundary_loss",
        "delta_num_samples_with_active_boundary_loss",
        "delta_dev_quad_f1",
        "delta_test_quad_f1",
        "boundary_positive_hit_count",
        "boundary_positive",
        "test_nonpositive",
        "boundary_to_e2e_transfer_break",
    ]
    per_seed_path = out_dir / "cbr_vs_affacr_transfer_diagnosis_per_seed.csv"
    write_csv(per_seed_path, per_seed, per_seed_fields)

    numeric_core = [
        "dev_quad_f1",
        "test_quad_f1",
        "A1",
        "A1_opinion_only_miss",
        "A3",
        "A3_topn_drop",
        "A3_cat_aff_not_materialized",
        "A_total",
        "gold_pair_mean_rank",
        "gold_pair_median_rank",
        "score_topn_minus_gold_mean",
        "score_topn_minus_gold_median",
        "sample_has_positive_after_retention_ratio",
        "first_outranker_null_ratio",
        "first_outranker_near_miss_ratio",
        "first_outranker_other_ratio",
        "boundary_active_positive_ratio",
        "avg_cutoff_gap",
        "num_samples_with_active_boundary_loss",
    ]
    aff_rows = list(aff_map[s] for s in seeds)
    cbr_rows = list(cbr_map[s] for s in seeds)
    delta_rows = []
    for r in per_seed:
        d = {"seed": r["seed"]}
        for k in numeric_core:
            d[k] = r.get(f"delta_{k}")
        delta_rows.append(d)

    summary = {
        "availability": {
            "A1_A3_A_total": True,
            "A1_opinion_only_miss": all(aff_map[s].get("A1_opinion_only_miss") is not None for s in seeds),
            "A3_topn_drop_cat_aff": True,
            "gold_pair_rank_mean_median": True,
            "score_topn_minus_gold": True,
            "sample_has_positive_after_retention_ratio": True,
            "first_outranker_type_ratio": True,
            "boundary_logs_cbr": True,
            "boundary_logs_affacr": False,
            "pair_score_diagnosis_outputs": False,
        },
        "group_affacr": mean_std(aff_rows, numeric_core, "affacr"),
        "group_cbr": mean_std(cbr_rows, numeric_core, "cbr"),
        "delta_cbr_minus_affacr": mean_std(delta_rows, numeric_core, "delta"),
        "transfer_break": {
            "num_seeds": len(seeds),
            "boundary_positive_seeds": int(sum(1 for r in per_seed if r["boundary_positive"])),
            "test_positive_seeds": int(sum(1 for r in per_seed if (r["delta_test_quad_f1"] or 0.0) > 0)),
            "boundary_to_e2e_transfer_break_seeds": transfer_fail_seeds,
        },
    }
    summary_path = out_dir / "cbr_vs_affacr_transfer_diagnosis_mean_std.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Takeaways markdown
    d = summary["delta_cbr_minus_affacr"]
    md = [
        "# CBR-v1 vs Aff-ACR Transfer Diagnosis",
        "",
        f"- seeds: {seeds}",
        f"- delta test mean: {d.get('delta_test_quad_f1_mean')}",
        f"- delta dev mean: {d.get('delta_dev_quad_f1_mean')}",
        "",
        "## Boundary-side vs Materialization-side",
        f"- A3_topn_drop delta mean: {d.get('delta_A3_topn_drop_mean')} (negative means improved)",
        f"- rank mean/median delta: {d.get('delta_gold_pair_mean_rank_mean')} / {d.get('delta_gold_pair_median_rank_mean')}",
        f"- sample_pos delta mean: {d.get('delta_sample_has_positive_after_retention_ratio_mean')}",
        f"- A3_cat_aff_not_materialized delta mean: {d.get('delta_A3_cat_aff_not_materialized_mean')}",
        f"- A1_opinion_only_miss delta mean: {d.get('delta_A1_opinion_only_miss_mean')}",
        "",
        "## Outranker shift",
        f"- NULL ratio delta mean: {d.get('delta_first_outranker_null_ratio_mean')}",
        f"- near_miss ratio delta mean: {d.get('delta_first_outranker_near_miss_ratio_mean')}",
        f"- other ratio delta mean: {d.get('delta_first_outranker_other_ratio_mean')}",
        "",
        "## Transfer break seeds",
        f"- boundary-positive seeds: {summary['transfer_break']['boundary_positive_seeds']}/{len(seeds)}",
        f"- test-positive seeds: {summary['transfer_break']['test_positive_seeds']}/{len(seeds)}",
        f"- boundary->E2E break seeds (boundary positive but test non-positive): {transfer_fail_seeds}",
        "",
        "## Conclusion",
        "- CBR-v1 currently appears boundary-positive but transfer-unstable across seeds.",
        "- It should be treated as a preliminary boundary-positive direction rather than a verified refinement.",
    ]
    md_path = out_dir / "cbr_vs_affacr_transfer_takeaways.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {per_seed_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
