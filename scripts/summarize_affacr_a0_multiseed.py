"""Summarize AGML-BR + A0 + Aff-ACR controlled multiseed outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Aff-ACR A0 multiseed runs")
    parser.add_argument("--output_root", required=True, help="Runner output root path")
    parser.add_argument(
        "--baseline_summary",
        default=None,
        help="Optional baseline CSV (e.g., agmlbr_a0_multiseed_summary.csv) for dev/test deltas",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(v):
    if v is None:
        return None
    return float(v)


def collect_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in sorted(output_root.glob("runs/*/run_manifest.json")):
        m = load_json(manifest_path)
        seed = int(m["seed"])
        run_dir = Path(m["run_dir"])
        stage1_dir = Path(m["stage1_ckpt"]).parent

        eval_dev = load_json(Path(m["eval_dev_json"]))
        eval_test = load_json(Path(m["eval_test_json"]))
        a_breakdown = load_json(Path(m["stage1_a_breakdown_test"]))

        stage1_metrics_path = stage1_dir / "stage1_metrics.json"
        stage1_metrics = load_json(stage1_metrics_path) if stage1_metrics_path.exists() else {}
        best_dev_metrics = stage1_metrics.get("best_dev_metrics", {})

        a1_side_path = stage1_dir / "a1_span_side_split_test.json"
        a1_side = load_json(a1_side_path) if a1_side_path.exists() else {}
        opinion_only = (
            a1_side.get("A1_split_counts", {}).get("opinion_only_miss")
            if a1_side else None
        )

        rows.append(
            {
                "seed": seed,
                "dev_quad_f1": safe_float(eval_dev.get("quad_f1")),
                "test_quad_f1": safe_float(eval_test.get("quad_f1")),
                "A1": int(a_breakdown["A_breakdown_counts"]["A1"]),
                "A1_opinion_only_miss": opinion_only,
                "A3": int(a_breakdown["A_breakdown_counts"]["A3"]),
                "A3_topn_drop": int(a_breakdown["A3_subtypes"]["topn_drop"]),
                "A3_cat_aff_not_materialized": int(
                    a_breakdown["A3_subtypes"]["cat_aff_not_materialized"]
                ),
                "A_total": int(a_breakdown["total_A_miss"]),
                "sample_has_positive_after_retention_ratio": safe_float(
                    best_dev_metrics.get("sample_has_positive_after_retention_ratio")
                ),
                "gold_pair_recall_pair_space": safe_float(
                    best_dev_metrics.get("gold_pair_recall_pair_space")
                ),
                "gold_pair_recall_after_gate": safe_float(
                    best_dev_metrics.get("gold_pair_recall_after_gate")
                ),
                "avg_pairs_into_stage2": safe_float(
                    best_dev_metrics.get("avg_pairs_into_stage2")
                ),
                "avg_candidates_into_stage2": safe_float(
                    best_dev_metrics.get("avg_candidates_into_stage2")
                ),
                "run_dir": str(run_dir),
                "stage1_ckpt": m["stage1_ckpt"],
                "stage2_ckpt": m["stage2_ckpt"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_mean_std(rows: list[dict], numeric_fields: list[str]) -> dict:
    out = {}
    for k in numeric_fields:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[f"{k}_mean"] = None
            out[f"{k}_std"] = None
            continue
        out[f"{k}_mean"] = float(mean(vals))
        out[f"{k}_std"] = float(pstdev(vals))
    return out


def load_baseline_map(path: Path) -> dict[int, dict]:
    baseline: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            baseline[seed] = {
                "dev_quad_f1": float(r["agmlbr_A0_dev"]),
                "test_quad_f1": float(r["agmlbr_A0_test"]),
            }
    return baseline


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "summary"

    rows = collect_rows(output_root)
    if not rows:
        raise RuntimeError(f"No run manifests found under: {output_root / 'runs'}")

    rows = sorted(rows, key=lambda x: x["seed"])

    per_seed_fields = [
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
        "gold_pair_recall_pair_space",
        "gold_pair_recall_after_gate",
        "avg_pairs_into_stage2",
        "avg_candidates_into_stage2",
        "run_dir",
        "stage1_ckpt",
        "stage2_ckpt",
    ]
    per_seed_csv = summary_dir / "affacr_a0_4seed_per_seed.csv"
    write_csv(per_seed_csv, rows, per_seed_fields)

    numeric_fields = [
        "dev_quad_f1",
        "test_quad_f1",
        "A1",
        "A3",
        "A3_topn_drop",
        "A3_cat_aff_not_materialized",
        "A_total",
        "sample_has_positive_after_retention_ratio",
        "gold_pair_recall_pair_space",
        "gold_pair_recall_after_gate",
        "avg_pairs_into_stage2",
        "avg_candidates_into_stage2",
    ]
    # A1 opinion-only may be missing if no side-split artifact exists.
    if all(r.get("A1_opinion_only_miss") is not None for r in rows):
        numeric_fields.append("A1_opinion_only_miss")

    mean_std = summarize_mean_std(rows, numeric_fields)
    mean_std["num_runs"] = len(rows)
    mean_std["has_a1_opinion_only_miss"] = all(
        r.get("A1_opinion_only_miss") is not None for r in rows
    )
    mean_std_json = summary_dir / "affacr_a0_4seed_mean_std.json"
    mean_std_json.write_text(json.dumps(mean_std, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.baseline_summary:
        baseline_map = load_baseline_map(Path(args.baseline_summary).resolve())
        delta_rows = []
        for r in rows:
            b = baseline_map.get(int(r["seed"]))
            if b is None:
                continue
            delta_rows.append(
                {
                    "seed": r["seed"],
                    "baseline_dev_quad_f1": b["dev_quad_f1"],
                    "baseline_test_quad_f1": b["test_quad_f1"],
                    "affacr_dev_quad_f1": r["dev_quad_f1"],
                    "affacr_test_quad_f1": r["test_quad_f1"],
                    "dev_delta": r["dev_quad_f1"] - b["dev_quad_f1"],
                    "test_delta": r["test_quad_f1"] - b["test_quad_f1"],
                }
            )
        delta_csv = summary_dir / "affacr_a0_delta_vs_agmlbr_a0.csv"
        write_csv(
            delta_csv,
            delta_rows,
            [
                "seed",
                "baseline_dev_quad_f1",
                "baseline_test_quad_f1",
                "affacr_dev_quad_f1",
                "affacr_test_quad_f1",
                "dev_delta",
                "test_delta",
            ],
        )

    print(f"Wrote per-seed: {per_seed_csv}")
    print(f"Wrote mean/std: {mean_std_json}")
    if args.baseline_summary:
        print(f"Wrote baseline delta: {summary_dir / 'affacr_a0_delta_vs_agmlbr_a0.csv'}")


if __name__ == "__main__":
    main()
