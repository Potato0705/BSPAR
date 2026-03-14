"""Run Stage-1 pair-rank ablations over lambda/margin/seeds."""

import argparse
import copy
import csv
import json
import os
import sys
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data,
    build_category_map,
    get_categories_for_dataset,
)
from bspar.training.stage1_trainer import Stage1Trainer
from bspar.utils.seed import set_seed
from scripts.stage1_run_manager import (
    init_run_manifest,
    finalize_manifest,
    build_run_index_row,
    update_run_index,
    write_metrics_file,
    build_command,
    tee_to_log,
)


def parse_num_list(raw, cast_fn):
    return [cast_fn(item.strip()) for item in raw.split(",") if item.strip()]


def build_config(cfg_dict):
    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config


def tag_float(value):
    return str(value).replace(".", "p")


def load_a3_metrics(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Stage-1 pair-rank ablation sweep")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/asqp_rest15_phase5_pairrank.yaml",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage1_pairrank",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.1,0.3,0.5",
        help="Comma-separated lambda_pair_rank values",
    )
    parser.add_argument(
        "--margins",
        type=str,
        default="0.1,0.2",
        help="Comma-separated pair_rank_margin values",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,3407",
        help="Comma-separated seeds",
    )
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    dataset_name = base_cfg.get("dataset_name", "asqp_rest15")
    data_format = base_cfg.get("data_format", "auto")
    data_dir = base_cfg.get("data_dir", "data/asqp_rest15")
    train_file = os.path.join(data_dir, base_cfg.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, base_cfg.get("dev_file", "dev.txt"))

    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, id_to_cat = build_category_map(categories)
    train_examples = load_data(train_file, data_format, categories)
    dev_examples = load_data(dev_file, data_format, categories)

    lambda_values = parse_num_list(args.lambdas, float)
    margin_values = parse_num_list(args.margins, float)
    seeds = parse_num_list(args.seeds, int)

    run_index_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(args.output_root)), "run_index.csv")
    )
    ablation_summary_root = os.path.join(os.path.abspath(args.output_root), "ablation")
    os.makedirs(ablation_summary_root, exist_ok=True)
    rows = []
    stop_requested = False

    for lambda_pair_rank in lambda_values:
        for pair_rank_margin in margin_values:
            setting_tag = (
                f"lam{tag_float(lambda_pair_rank)}_margin{tag_float(pair_rank_margin)}"
            )
            for seed in seeds:
                manifest, _ = init_run_manifest(
                    output_root=args.output_root,
                    purpose="ablation",
                    dataset=dataset_name,
                    seed=seed,
                    lambda_pair_rank=lambda_pair_rank,
                    pair_rank_margin=pair_rank_margin,
                    retention=base_cfg.get("stage1_pair_retention_strategy", "topn_only"),
                    top_n=base_cfg.get("stage1_pair_top_n", 20),
                    config_path=args.base_config,
                    command=build_command(),
                    run_tag=setting_tag,
                )
                update_run_index(run_index_path, build_run_index_row(manifest))
                run_dir = manifest["run_dir"]
                final_status = "running"
                error_message = None
                trainer = None

                cfg_dict = copy.deepcopy(base_cfg)
                cfg_dict["lambda_pair_rank"] = float(lambda_pair_rank)
                cfg_dict["pair_rank_margin"] = float(pair_rank_margin)
                cfg_dict["output_dir"] = run_dir

                with open(
                    os.path.join(run_dir, "resolved_config.yaml"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

                config = build_config(cfg_dict)
                config.num_categories = len(categories)

                with tee_to_log(manifest["log_path"]):
                    print(
                        f"[PairRank] run_id={manifest['run_id']} | seed={seed} "
                        f"lambda={lambda_pair_rank} margin={pair_rank_margin} -> {run_dir}"
                    )
                    try:
                        set_seed(seed)
                        trainer = Stage1Trainer(
                            config,
                            train_examples,
                            dev_examples,
                            cat_to_id,
                            id_to_cat,
                        )
                        trainer.train(run_dir)
                        write_metrics_file(
                            manifest["metrics_path"],
                            getattr(trainer, "best_dev_metrics", {}),
                            getattr(trainer, "final_dev_metrics", {}),
                        )
                        final_status = "finished"
                    except KeyboardInterrupt:
                        final_status = "interrupted"
                        error_message = "Interrupted by user"
                        print(error_message)
                    except Exception:
                        final_status = "failed"
                        error_message = traceback.format_exc()
                        print(error_message)
                    finally:
                        if trainer is not None:
                            write_metrics_file(
                                manifest["metrics_path"],
                                getattr(trainer, "best_dev_metrics", {}),
                                getattr(trainer, "final_dev_metrics", {}),
                            )

                manifest, _ = finalize_manifest(
                    manifest,
                    output_root=args.output_root,
                    status=final_status,
                    error_message=error_message,
                )
                best_metrics = dict(getattr(trainer, "best_dev_metrics", {}))
                update_run_index(
                    run_index_path,
                    build_run_index_row(manifest, best_metrics),
                )
                a3_metrics = load_a3_metrics(manifest["a3_best_path"])

                row = {
                    "run_id": manifest.get("run_id", ""),
                    "status": manifest.get("status", ""),
                    "seed": seed,
                    "lambda_pair_rank": float(lambda_pair_rank),
                    "pair_rank_margin": float(pair_rank_margin),
                    "output_dir": manifest.get("run_dir", run_dir),
                    "best_quad_f1": best_metrics.get("quad_f1", 0.0),
                    "best_span_f1": best_metrics.get("span_f1", 0.0),
                    "best_gold_pair_recall_after_gate": best_metrics.get(
                        "gold_pair_recall_after_gate", 0.0
                    ),
                    "best_sample_has_positive_after_retention_ratio": best_metrics.get(
                        "sample_has_positive_after_retention_ratio", 0.0
                    ),
                    "a3_total_gold_pairs": a3_metrics.get("counts", {}).get(
                        "total_a3_gold_pairs", 0
                    ),
                    "a3_gold_pair_rank_mean": a3_metrics.get(
                        "gold_pair_rank", {}
                    ).get("mean", 0.0),
                    "a3_gold_pair_rank_median": a3_metrics.get(
                        "gold_pair_rank", {}
                    ).get("median", 0.0),
                    "a3_gap_mean": a3_metrics.get(
                        "score_topn_minus_gold_pair", {}
                    ).get("mean", 0.0),
                    "a3_gap_median": a3_metrics.get(
                        "score_topn_minus_gold_pair", {}
                    ).get("median", 0.0),
                    "a3_first_outranker_null_ratio": a3_metrics.get(
                        "first_outranker_type_ratio", {}
                    ).get("NULL", {}).get("ratio", 0.0),
                    "a3_first_outranker_nearmiss_ratio": a3_metrics.get(
                        "first_outranker_type_ratio", {}
                    ).get("near_miss", {}).get("ratio", 0.0),
                    "a3_first_outranker_other_ratio": a3_metrics.get(
                        "first_outranker_type_ratio", {}
                    ).get("other", {}).get("ratio", 0.0),
                }
                rows.append(row)
                if final_status == "interrupted":
                    stop_requested = True
                    break
            if stop_requested:
                break
        if stop_requested:
            break

    summary = {
        "base_config": args.base_config,
        "output_root": args.output_root,
        "lambda_values": lambda_values,
        "margin_values": margin_values,
        "seeds": seeds,
        "interrupted": stop_requested,
        "rows": rows,
    }

    summary_json = os.path.join(ablation_summary_root, "ablation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(ablation_summary_root, "ablation_summary.csv")
    fieldnames = [
        "run_id",
        "status",
        "seed",
        "lambda_pair_rank",
        "pair_rank_margin",
        "best_quad_f1",
        "best_span_f1",
        "best_gold_pair_recall_after_gate",
        "best_sample_has_positive_after_retention_ratio",
        "a3_total_gold_pairs",
        "a3_gold_pair_rank_mean",
        "a3_gold_pair_rank_median",
        "a3_gap_mean",
        "a3_gap_median",
        "a3_first_outranker_null_ratio",
        "a3_first_outranker_nearmiss_ratio",
        "a3_first_outranker_other_ratio",
        "output_dir",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved -> {summary_json}")
    print(f"Saved -> {summary_csv}")


if __name__ == "__main__":
    main()
