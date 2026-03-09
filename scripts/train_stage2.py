"""Entry point for Stage-2 reranker training."""

import sys
import os
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.training.stage2_trainer import Stage2Trainer
from bspar.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="BSPAR Stage-2 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--candidates_dir", type=str, required=True,
                        help="Directory with rerank_train.pt and rerank_dev.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()

    set_seed(args.seed)

    # Load real candidates
    train_rerank = torch.load(
        os.path.join(args.candidates_dir, "rerank_train.pt"),
        weights_only=False,
    )
    dev_rerank = torch.load(
        os.path.join(args.candidates_dir, "rerank_dev.pt"),
        weights_only=False,
    )

    print(f"Train rerank examples: {len(train_rerank)}")
    print(f"Dev rerank examples: {len(dev_rerank)}")

    trainer = Stage2Trainer(config, train_rerank, dev_rerank)
    trainer.train(args.output_dir)


if __name__ == "__main__":
    main()
