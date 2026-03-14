"""Entry point for Stage-1 training."""

import sys
import os
import argparse
import traceback
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data, build_category_map, get_categories_for_dataset
)
from bspar.training.stage1_trainer import Stage1Trainer
from bspar.utils.seed import set_seed
from scripts.stage1_run_manager import (
    infer_purpose,
    init_run_manifest,
    finalize_manifest,
    build_run_index_row,
    update_run_index,
    write_metrics_file,
    build_command,
    tee_to_log,
)


def load_config(config_path: str) -> tuple[BSPARConfig, dict]:
    """Load config from YAML and create BSPARConfig."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config, cfg_dict


def main():
    parser = argparse.ArgumentParser(description="BSPAR Stage-1 Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional run dir override")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage1_pairrank",
        help="Governed output root",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        choices=["smoke", "baseline", "ablation"],
        default=None,
        help="Run purpose bucket",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag")
    args = parser.parse_args()

    # Load config
    config, cfg_dict = load_config(args.config)

    # Overrides
    data_dir = args.data_dir or cfg_dict.get("data_dir", "data/asqp_rest15")
    dataset_name = cfg_dict.get("dataset_name", "asqp_rest15")
    data_format = cfg_dict.get("data_format", "auto")
    lambda_pair_rank = cfg_dict.get(
        "lambda_pair_rank", getattr(config, "lambda_pair_rank", 0.0)
    )
    pair_rank_margin = cfg_dict.get(
        "pair_rank_margin", getattr(config, "pair_rank_margin", 0.1)
    )
    purpose = args.purpose or infer_purpose(lambda_pair_rank)

    manifest, _ = init_run_manifest(
        output_root=args.output_root,
        purpose=purpose,
        dataset=dataset_name,
        seed=args.seed,
        lambda_pair_rank=lambda_pair_rank,
        pair_rank_margin=pair_rank_margin,
        retention=cfg_dict.get("stage1_pair_retention_strategy", "topn_only"),
        top_n=cfg_dict.get("stage1_pair_top_n", 20),
        config_path=args.config,
        command=build_command(),
        run_tag=args.run_tag,
        run_dir_override=args.output_dir,
    )
    run_dir = manifest["run_dir"]
    run_index_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(args.output_root)), "run_index.csv")
    )
    update_run_index(run_index_path, build_run_index_row(manifest))

    final_status = "running"
    error_message = None
    trainer = None

    with tee_to_log(manifest["log_path"]):
        print(f"Run ID: {manifest['run_id']}")
        print(f"Purpose: {purpose}")
        print(f"Config: {args.config}")
        print(f"Dataset: {dataset_name} | Data: {data_dir}")
        print(f"Output: {run_dir}")
        print(f"Model: {config.model_name}")
        print(f"Seed: {args.seed}")
        try:
            # Seed
            set_seed(args.seed)

            # Category mapping
            categories = get_categories_for_dataset(dataset_name)
            cat_to_id, id_to_cat = build_category_map(categories)
            config.num_categories = len(categories)

            # Load data
            train_file = os.path.join(data_dir, cfg_dict.get("train_file", "train.txt"))
            dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))
            train_examples = load_data(train_file, data_format, categories)
            dev_examples = load_data(dev_file, data_format, categories)

            print(f"Train: {len(train_examples)} examples")
            print(f"Dev: {len(dev_examples)} examples")
            print(f"Categories: {len(categories)}")

            # Train
            trainer = Stage1Trainer(
                config, train_examples, dev_examples, cat_to_id, id_to_cat
            )
            best_f1 = trainer.train(run_dir)
            write_metrics_file(
                manifest["metrics_path"],
                getattr(trainer, "best_dev_metrics", {}),
                getattr(trainer, "final_dev_metrics", {}),
            )
            print(f"\nDone! Best dev Quad-F1: {best_f1:.4f}")
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
    update_run_index(
        run_index_path,
        build_run_index_row(manifest, getattr(trainer, "best_dev_metrics", {})),
    )

    if final_status != "finished":
        raise SystemExit(130 if final_status == "interrupted" else 1)


if __name__ == "__main__":
    main()
