"""Summarize AGML-BR + A0 + Aff-ACR only + CBR-v1 controlled multiseed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CBR-v1 A0 multiseed")
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--baseline_summary",
        default="outputs/stage2_e2e_agmlbr_a0_multiseed_20260317_095403/summary/agmlbr_a0_multiseed_summary.csv",
    )
    parser.add_argument(
        "--affacr_per_seed",
        default="outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/summary/affacr_a0_4seed_per_seed.csv",
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


def outranker_ratio(diag: dict, key: str):
    d = diag.get("first_outranker_type_ratio", {}).get(key)
    if isinstance(d, dict):
        return to_float(d.get("ratio"))
    return None


def collect_stage1_from_ckpt(stage1_ckpt: str) -> dict:
    stage1_dir = Path(stage1_ckpt).resolve().parent
    breakdown = load_json(stage1_dir / "stage1_a_breakdown_test.json")
    diag = load_json(stage1_dir / "best_stage1_a3_diagnostics.json")
    metrics = load_json(stage1_dir / "stage1_metrics.json").get("best_dev_metrics", {})
    a1_split_path = stage1_dir / "a1_span_side_split_test.json"
    a1_split = load_json(a1_split_path) if a1_split_path.exists() else {}
    return {
        "A1": int(breakdown["A_breakdown_counts"]["A1"]),
        "A1_opinion_only_miss": a1_split.get("A1_split_counts", {}).get("opinion_only_miss"),
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
        "first_outranker_null_ratio": outranker_ratio(diag, "NULL"),
        "first_outranker_near_miss_ratio": outranker_ratio(diag, "near_miss"),
        "first_outranker_other_ratio": outranker_ratio(diag, "other"),
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
        "stage1_dir": str(stage1_dir),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_mean_std(rows: list[dict], numeric_fields: list[str]) -> dict:
    out = {}
    for k in numeric_fields:
        vals = [to_float(r.get(k)) for r in rows if r.get(k) is not None]
        if not vals:
            out[f"{k}_mean"] = None
            out[f"{k}_std"] = None
            continue
        out[f"{k}_mean"] = float(mean(vals))
        out[f"{k}_std"] = float(pstdev(vals))
    return out


def load_baseline_map(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            stage1 = collect_stage1_from_ckpt(r["stage1_ckpt"])
            stage1.update(
                {
                    "seed": seed,
                    "dev_quad_f1": to_float(r["agmlbr_A0_dev"]),
                    "test_quad_f1": to_float(r["agmlbr_A0_test"]),
                    "run_dir": r["run_dir"],
                    "stage1_ckpt": r["stage1_ckpt"],
                    "track": "baseline_agmlbr_a0",
                }
            )
            out[seed] = stage1
    return out


def load_affacr_map(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            stage1 = collect_stage1_from_ckpt(r["stage1_ckpt"])
            stage1.update(
                {
                    "seed": seed,
                    "dev_quad_f1": to_float(r["dev_quad_f1"]),
                    "test_quad_f1": to_float(r["test_quad_f1"]),
                    "run_dir": r["run_dir"],
                    "stage1_ckpt": r["stage1_ckpt"],
                    "track": "strongest_affacr_only",
                }
            )
            out[seed] = stage1
    return out


def collect_cbr_rows(output_root: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(output_root.glob("runs/*/run_manifest.json")):
        m = load_json(manifest_path)
        stage1 = collect_stage1_from_ckpt(m["stage1_ckpt"])
        eval_dev = load_json(Path(m["eval_dev_json"]))
        eval_test = load_json(Path(m["eval_test_json"]))
        row = {
            "track": "cbr_v1_affacr_a0",
            "seed": int(m["seed"]),
            "cbr_v1_lambda": float(m["cbr_v1_lambda"]),
            "cbr_v1_margin": float(m["cbr_v1_margin"]),
            "dev_quad_f1": to_float(eval_dev.get("quad_f1")),
            "test_quad_f1": to_float(eval_test.get("quad_f1")),
            "run_dir": m["run_dir"],
            "stage1_ckpt": m["stage1_ckpt"],
            "stage2_ckpt": m["stage2_ckpt"],
        }
        row.update(stage1)
        rows.append(row)
    return sorted(rows, key=lambda r: r["seed"])


def delta_row(cur: dict, ref: dict, ref_name: str) -> dict:
    out = {
        "seed": cur["seed"],
        "reference": ref_name,
        "cbr_v1_lambda": cur.get("cbr_v1_lambda"),
        "cbr_v1_margin": cur.get("cbr_v1_margin"),
    }
    fields = [
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
    ]
    for k in fields:
        cv = cur.get(k)
        rv = ref.get(k)
        out[f"delta_{k}"] = (float(cv) - float(rv)) if cv is not None and rv is not None else None
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    cbr_rows = collect_cbr_rows(output_root)
    if not cbr_rows:
        raise RuntimeError(f"No CBR run manifests found under: {output_root / 'runs'}")

    baseline_map = load_baseline_map(Path(args.baseline_summary).resolve())
    affacr_map = load_affacr_map(Path(args.affacr_per_seed).resolve())

    per_seed_fields = [
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
        "run_dir",
        "stage1_ckpt",
        "stage2_ckpt",
    ]
    per_seed_csv = summary_dir / "cbrv1_a0_4seed_per_seed.csv"
    write_csv(per_seed_csv, cbr_rows, per_seed_fields)

    numeric_fields = [
        "dev_quad_f1",
        "test_quad_f1",
        "A1",
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
    ]
    if all(r.get("A1_opinion_only_miss") is not None for r in cbr_rows):
        numeric_fields.append("A1_opinion_only_miss")
    mean_std = summarize_mean_std(cbr_rows, numeric_fields)
    mean_std["num_runs"] = len(cbr_rows)
    mean_std["config"] = {
        "cbr_v1_lambda": cbr_rows[0]["cbr_v1_lambda"],
        "cbr_v1_margin": cbr_rows[0]["cbr_v1_margin"],
    }
    mean_std_json = summary_dir / "cbrv1_a0_4seed_mean_std.json"
    mean_std_json.write_text(json.dumps(mean_std, indent=2, ensure_ascii=False), encoding="utf-8")

    delta_vs_base = []
    delta_vs_affacr = []
    criteria_rows = []
    for r in cbr_rows:
        seed = int(r["seed"])
        b = baseline_map.get(seed)
        a = affacr_map.get(seed)
        if b:
            delta_vs_base.append(delta_row(r, b, "baseline_agmlbr_a0"))
        if a:
            d = delta_row(r, a, "strongest_affacr_only")
            cond_1 = (d.get("delta_A3_topn_drop") is not None and d["delta_A3_topn_drop"] < 0)
            cond_2 = (
                d.get("delta_gold_pair_mean_rank") is not None
                and d.get("delta_gold_pair_median_rank") is not None
                and d["delta_gold_pair_mean_rank"] <= 0
                and d["delta_gold_pair_median_rank"] <= 0
            )
            cond_3 = (
                d.get("delta_sample_has_positive_after_retention_ratio") is not None
                and d["delta_sample_has_positive_after_retention_ratio"] >= 0
            )
            hit = int(cond_1) + int(cond_2) + int(cond_3)
            non_harm = (
                d.get("delta_dev_quad_f1") is not None
                and d.get("delta_test_quad_f1") is not None
                and d["delta_dev_quad_f1"] >= -0.002
                and d["delta_test_quad_f1"] >= -0.002
            )
            d["criteria_hit_count"] = hit
            d["cbrv1_continue_criteria_ok"] = bool(hit >= 2 and non_harm)
            delta_vs_affacr.append(d)
            criteria_rows.append(
                {
                    "seed": seed,
                    "cond_A3_topn_drop_improve": bool(cond_1),
                    "cond_rank_mean_median_improve": bool(cond_2),
                    "cond_sample_pos_nonneg": bool(cond_3),
                    "non_harm_dev_test": bool(non_harm),
                    "criteria_hit_count": hit,
                    "continue_ok": bool(hit >= 2 and non_harm),
                }
            )

    if delta_vs_base:
        write_csv(
            summary_dir / "cbrv1_a0_delta_vs_agmlbr_a0.csv",
            delta_vs_base,
            list(delta_vs_base[0].keys()),
        )
    if delta_vs_affacr:
        write_csv(
            summary_dir / "cbrv1_a0_delta_vs_affacr_only.csv",
            delta_vs_affacr,
            list(delta_vs_affacr[0].keys()),
        )

    continue_summary = {
        "num_seeds": len(criteria_rows),
        "num_continue_ok": sum(1 for r in criteria_rows if r["continue_ok"]),
        "continue_ok_ratio": (
            (sum(1 for r in criteria_rows if r["continue_ok"]) / len(criteria_rows))
            if criteria_rows else 0.0
        ),
        "criteria_per_seed": criteria_rows,
    }
    (summary_dir / "cbrv1_a0_continue_check.json").write_text(
        json.dumps(continue_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# CBR-v1 A0 4-seed Takeaways",
        "",
        f"- output_root: {output_root}",
        f"- runs: {len(cbr_rows)}",
        f"- fixed lambda/margin: {cbr_rows[0]['cbr_v1_lambda']} / {cbr_rows[0]['cbr_v1_margin']}",
        "",
    ]
    for r in criteria_rows:
        md.append(
            f"- seed{r['seed']}: hit={r['criteria_hit_count']}/3, "
            f"non_harm={r['non_harm_dev_test']}, continue={r['continue_ok']}"
        )
    (summary_dir / "cbrv1_a0_takeaways.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {per_seed_csv}")
    print(f"Wrote: {mean_std_json}")
    if delta_vs_base:
        print(f"Wrote: {summary_dir / 'cbrv1_a0_delta_vs_agmlbr_a0.csv'}")
    if delta_vs_affacr:
        print(f"Wrote: {summary_dir / 'cbrv1_a0_delta_vs_affacr_only.csv'}")
    print(f"Wrote: {summary_dir / 'cbrv1_a0_continue_check.json'}")
    print(f"Wrote: {summary_dir / 'cbrv1_a0_takeaways.md'}")


if __name__ == "__main__":
    main()

