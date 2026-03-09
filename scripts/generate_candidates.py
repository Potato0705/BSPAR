"""Generate real candidates from trained Stage-1 model for Stage-2 training."""

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
from bspar.models.bspar_stage1 import BSPARStage1
from bspar.training.candidate_generator import CandidateGenerator
from bspar.utils.seed import set_seed
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Generate Stage-2 Candidates")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_stage1.pt")
    parser.add_argument("--output", type=str, default="outputs/candidates")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = BSPARConfig()
    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()

    set_seed(42)

    cat_to_id, id_to_cat = build_category_map(ASQP_CATEGORIES)
    config.num_categories = len(ASQP_CATEGORIES)

    data_dir = cfg_dict.get("data_dir", "data/asqp_rest15")
    train_file = os.path.join(data_dir, cfg_dict.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))

    train_examples = load_asqp_file(train_file, ASQP_CATEGORIES)
    dev_examples = load_asqp_file(dev_file, ASQP_CATEGORIES)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSPARStage1(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    generator = CandidateGenerator(model, config, id_to_cat)

    # Generate for train and dev
    os.makedirs(args.output, exist_ok=True)

    for split, examples in [("train", train_examples), ("dev", dev_examples)]:
        dataset = BSPARStage1Dataset(examples, config.model_name)
        loader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_stage1)

        rerank_examples = generator.generate(loader, examples)
        out_path = os.path.join(args.output, f"rerank_{split}.pt")
        torch.save(rerank_examples, out_path)
        print(f"{split}: {len(rerank_examples)} examples with candidates → {out_path}")


if __name__ == "__main__":
    main()
