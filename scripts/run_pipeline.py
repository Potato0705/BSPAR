"""Full BSPAR pipeline: Stage-1 → Candidate Generation → Stage-2.

Single-command entry point for the complete training pipeline.
"""

import sys
import os
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_asqp_file, ASQP_CATEGORIES, build_category_map
)
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.training.stage1_trainer import Stage1Trainer
from bspar.training.candidate_generator import CandidateGenerator
from bspar.training.stage2_trainer import Stage2Trainer
from bspar.utils.seed import set_seed
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="BSPAR Full Pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage-1 and use existing checkpoint")
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()

    set_seed(args.seed)
    output_dir = cfg_dict.get("output_dir", "outputs")
    data_dir = cfg_dict.get("data_dir", "data/asqp_rest15")

    cat_to_id, id_to_cat = build_category_map(ASQP_CATEGORIES)
    config.num_categories = len(ASQP_CATEGORIES)

    # Load data
    train_file = os.path.join(data_dir, cfg_dict.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))
    train_examples = load_asqp_file(train_file, ASQP_CATEGORIES)
    dev_examples = load_asqp_file(dev_file, ASQP_CATEGORIES)

    print(f"=== BSPAR Full Pipeline ===")
    print(f"Train: {len(train_examples)} | Dev: {len(dev_examples)}")
    print(f"Categories: {len(ASQP_CATEGORIES)}")

    # =========================================================================
    # Phase A: Stage-1 Training
    # =========================================================================
    stage1_dir = os.path.join(output_dir, "stage1")
    stage1_ckpt = args.stage1_ckpt or os.path.join(stage1_dir, "best_stage1.pt")

    if not args.skip_stage1:
        print("\n=== Phase A: Stage-1 Training ===")
        trainer = Stage1Trainer(config, train_examples, dev_examples,
                                cat_to_id, id_to_cat)
        trainer.train(stage1_dir)
    else:
        print(f"\n=== Skipping Stage-1, using: {stage1_ckpt} ===")

    # =========================================================================
    # Phase A→B: Generate Real Candidates
    # =========================================================================
    print("\n=== Phase A→B: Generating Real Candidates ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from bspar.models.bspar_stage1 import BSPARStage1
    model = BSPARStage1(config)
    ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    generator = CandidateGenerator(model, config, id_to_cat)

    cand_dir = os.path.join(output_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    for split, examples in [("train", train_examples), ("dev", dev_examples)]:
        dataset = BSPARStage1Dataset(examples, config.model_name)
        loader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_stage1)
        rerank_examples = generator.generate(loader, examples)
        torch.save(rerank_examples, os.path.join(cand_dir, f"rerank_{split}.pt"))
        print(f"  {split}: {len(rerank_examples)} examples")

    # =========================================================================
    # Phase B: Stage-2 Training
    # =========================================================================
    print("\n=== Phase B: Stage-2 Training ===")
    train_rerank = torch.load(os.path.join(cand_dir, "rerank_train.pt"),
                              weights_only=False)
    dev_rerank = torch.load(os.path.join(cand_dir, "rerank_dev.pt"),
                            weights_only=False)

    if len(train_rerank) > 0 and len(dev_rerank) > 0:
        stage2_dir = os.path.join(output_dir, "stage2")
        stage2_trainer = Stage2Trainer(config, train_rerank, dev_rerank)
        stage2_trainer.train(stage2_dir)
    else:
        print("  WARNING: Not enough rerank examples. Skipping Stage-2.")
        print(f"  Train rerank: {len(train_rerank)}, Dev rerank: {len(dev_rerank)}")

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
