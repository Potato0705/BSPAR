"""Ablation: Stage-1 pair-level ranking loss.

Sweep lambda_pair_rank × pair_rank_margin on multiple seeds.
Reports: Dev Quad-F1, A3 count, gold pair rank stats, outranker analysis,
         sample_has_positive_after_retention_ratio.

Usage:
    python scripts/run_ablation_pair_rank.py --config configs/asqp_rest15.yaml
"""

import sys
import os
import argparse
import json
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data, build_category_map, get_categories_for_dataset
)
from bspar.training.stage1_trainer import Stage1Trainer
from bspar.utils.seed import set_seed


def load_config(config_path: str) -> tuple:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config, cfg_dict


def run_single(config, train_examples, dev_examples, cat_to_id, id_to_cat,
               seed, output_dir):
    """Run a single Stage-1 training and return final metrics."""
    set_seed(seed)
    trainer = Stage1Trainer(config, train_examples, dev_examples,
                            cat_to_id, id_to_cat)
    final_metrics = trainer.train(output_dir)
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Pair Ranking Loss Ablation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,123,456,789",
                        help="Comma-separated seeds")
    parser.add_argument("--lambda_pair_rank", type=str, default="0.0,0.1,0.3,0.5",
                        help="Comma-separated lambda values")
    parser.add_argument("--pair_rank_margin", type=str, default="0.1,0.2",
                        help="Comma-separated margin values")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_pair_rank")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    lambdas = [float(v) for v in args.lambda_pair_rank.split(",")]
    margins = [float(v) for v in args.pair_rank_margin.split(",")]

    base_config, cfg_dict = load_config(args.config)
    dataset_name = cfg_dict.get("dataset_name", "asqp_rest15")
    data_dir = cfg_dict.get("data_dir", "data/asqp_rest15")
    data_format = cfg_dict.get("data_format", "auto")

    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, id_to_cat = build_category_map(categories)
    base_config.num_categories = len(categories)

    train_file = os.path.join(data_dir, cfg_dict.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))
    train_examples = load_data(train_file, data_format, categories)
    dev_examples = load_data(dev_file, data_format, categories)

    print(f"=== Pair Ranking Loss Ablation ===")
    print(f"Dataset: {dataset_name} | Train: {len(train_examples)} | Dev: {len(dev_examples)}")
    print(f"Seeds: {seeds}")
    print(f"lambda_pair_rank: {lambdas}")
    print(f"pair_rank_margin: {margins}")
    print()

    # Build experiment grid
    experiments = []
    for lam in lambdas:
        for mar in margins:
            # Skip margin sweep for baseline (lambda=0)
            if lam == 0.0 and mar != margins[0]:
                continue
            experiments.append({"lambda_pair_rank": lam, "pair_rank_margin": mar})

    all_results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for exp in experiments:
        lam = exp["lambda_pair_rank"]
        mar = exp["pair_rank_margin"]
        exp_name = f"lam{lam}_mar{mar}"
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            config = BSPARConfig()
            for key, value in cfg_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.lambda_pair_rank = lam
            config.pair_rank_margin = mar
            config.num_categories = len(categories)
            config.__post_init__()

            run_dir = os.path.join(args.output_dir, exp_name, f"seed{seed}")
            print(f"\n--- Seed {seed} ---")
            metrics = run_single(config, train_examples, dev_examples,
                                 cat_to_id, id_to_cat, seed, run_dir)
            seed_results.append({"seed": seed, "metrics": metrics})

        # Aggregate across seeds
        quad_f1s = [r["metrics"].get("quad_f1", 0.0) for r in seed_results]
        span_f1s = [r["metrics"].get("span_f1", 0.0) for r in seed_results]

        a3_counts = []
        gold_rank_means = []
        gold_rank_medians = []
        score_gap_means = []
        score_gap_medians = []
        null_outranker_ratios = []
        near_miss_outranker_ratios = []
        pos_retention_ratios = []

        for r in seed_results:
            a3 = r["metrics"].get("a3", {})
            a3_counts.append(a3.get("a3_count", 0))
            if "gold_pair_mean_rank" in a3:
                gold_rank_means.append(a3["gold_pair_mean_rank"])
            if "gold_pair_median_rank" in a3:
                gold_rank_medians.append(a3["gold_pair_median_rank"])
            if "score_gap_mean" in a3:
                score_gap_means.append(a3["score_gap_mean"])
            if "score_gap_median" in a3:
                score_gap_medians.append(a3["score_gap_median"])
            null_outranker_ratios.append(a3.get("null_outranker_ratio", 0))
            near_miss_outranker_ratios.append(a3.get("near_miss_outranker_ratio", 0))
            pos_retention_ratios.append(
                a3.get("sample_has_positive_after_retention_ratio", 0))

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def _std(lst):
            if len(lst) < 2:
                return 0.0
            m = _mean(lst)
            return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

        summary = {
            "experiment": exp_name,
            "lambda_pair_rank": lam,
            "pair_rank_margin": mar,
            "num_seeds": len(seeds),
            "quad_f1_mean": _mean(quad_f1s),
            "quad_f1_std": _std(quad_f1s),
            "span_f1_mean": _mean(span_f1s),
            "span_f1_std": _std(span_f1s),
            "a3_count_mean": _mean(a3_counts),
            "gold_pair_mean_rank_mean": _mean(gold_rank_means),
            "gold_pair_median_rank_mean": _mean(gold_rank_medians),
            "score_gap_mean_mean": _mean(score_gap_means),
            "score_gap_median_mean": _mean(score_gap_medians),
            "null_outranker_ratio_mean": _mean(null_outranker_ratios),
            "near_miss_outranker_ratio_mean": _mean(near_miss_outranker_ratios),
            "sample_has_positive_after_retention_ratio_mean": _mean(pos_retention_ratios),
            "per_seed": seed_results,
        }
        all_results.append(summary)

        print(f"\n--- Summary: {exp_name} ---")
        print(f"  A. Dev Quad-F1: {summary['quad_f1_mean']:.4f} +/- {summary['quad_f1_std']:.4f}")
        print(f"  B. A3 count: {summary['a3_count_mean']:.1f}")
        print(f"  C. Gold pair rank: mean={summary['gold_pair_mean_rank_mean']:.1f} "
              f"median={summary['gold_pair_median_rank_mean']:.1f}")
        print(f"  D. Score gap: mean={summary['score_gap_mean_mean']:.4f} "
              f"median={summary['score_gap_median_mean']:.4f}")
        print(f"  E. NULL outranker ratio: {summary['null_outranker_ratio_mean']:.4f}")
        print(f"  F. Near-miss outranker ratio: {summary['near_miss_outranker_ratio_mean']:.4f}")
        print(f"  G. pos_retention_ratio: {summary['sample_has_positive_after_retention_ratio_mean']:.4f}")

    # Save full results
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    # Convert non-serializable items for JSON
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(results_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print final comparison table
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Experiment':<20} {'Quad-F1':>10} {'A3':>6} {'Rank_mean':>10} "
          f"{'Gap_mean':>10} {'NULL%':>8} {'NearMiss%':>10} {'PosRet':>8}")
    print("-" * 80)
    for s in all_results:
        print(f"{s['experiment']:<20} "
              f"{s['quad_f1_mean']:>10.4f} "
              f"{s['a3_count_mean']:>6.1f} "
              f"{s['gold_pair_mean_rank_mean']:>10.1f} "
              f"{s['score_gap_mean_mean']:>10.4f} "
              f"{s['null_outranker_ratio_mean']:>8.4f} "
              f"{s['near_miss_outranker_ratio_mean']:>10.4f} "
              f"{s['sample_has_positive_after_retention_ratio_mean']:>8.4f}")


if __name__ == "__main__":
    main()
