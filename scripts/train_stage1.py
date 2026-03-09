"""Entry point for Stage-1 training."""

import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_asqp_file, ASQP_CATEGORIES, build_category_map
)
from bspar.training.stage1_trainer import Stage1Trainer
from bspar.utils.seed import set_seed


def load_config(config_path: str) -> BSPARConfig:
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
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    args = parser.parse_args()

    # Load config
    config, cfg_dict = load_config(args.config)

    # Overrides
    data_dir = args.data_dir or cfg_dict.get("data_dir", "data/asqp_rest15")
    output_dir = args.output_dir or cfg_dict.get("output_dir", "outputs/stage1")

    # Seed
    set_seed(args.seed)
    print(f"BSPAR Stage-1 Training | Seed: {args.seed}")
    print(f"Config: {args.config}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {config.model_name}")

    # Category mapping
    cat_to_id, id_to_cat = build_category_map(ASQP_CATEGORIES)
    config.num_categories = len(ASQP_CATEGORIES)

    # Load data
    data_format = cfg_dict.get("data_format", "asqp_txt")
    train_file = os.path.join(data_dir, cfg_dict.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))

    if data_format == "asqp_txt":
        train_examples = load_asqp_file(train_file, ASQP_CATEGORIES)
        dev_examples = load_asqp_file(dev_file, ASQP_CATEGORIES)
    elif data_format == "jsonl":
        from bspar.data.preprocessor import load_jsonl_file
        train_examples = load_jsonl_file(train_file)
        dev_examples = load_jsonl_file(dev_file)
    else:
        raise ValueError(f"Unknown data format: {data_format}")

    print(f"Train: {len(train_examples)} examples")
    print(f"Dev: {len(dev_examples)} examples")

    # Train
    trainer = Stage1Trainer(
        config, train_examples, dev_examples, cat_to_id, id_to_cat
    )
    best_f1 = trainer.train(output_dir)
    print(f"\nDone! Best dev Quad-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
