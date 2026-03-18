"""Extract A1 span-side split (aspect/opinion/both miss) for one Stage-1 checkpoint.

Outputs a compact JSON:
{
  "split": "test",
  "A1_total": ...,
  "A1_split_counts": {
    "opinion_only_miss": ...,
    "both_miss": ...,
    "aspect_only_miss": ...
  },
  "A1_split_ratio_over_A1": {...}
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.data.preprocessor import build_category_map, get_categories_for_dataset, load_data
from bspar.models.bspar_stage1 import BSPARStage1
from bspar.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract A1 span-side split on test/dev split")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--stage1_ckpt", required=True, help="Stage-1 checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def load_config(config_path: str) -> tuple[BSPARConfig, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = BSPARConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, raw


def build_loader(cfg: BSPARConfig, raw_cfg: dict, split: str):
    dataset_name = raw_cfg.get("dataset_name", "asqp_rest15")
    data_format = raw_cfg.get("data_format", "auto")
    data_dir = raw_cfg.get("data_dir", "data/asqp_rest15")
    split_file = raw_cfg.get(f"{split}_file", f"{split}.txt")
    split_path = os.path.join(data_dir, split_file)

    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, _ = build_category_map(categories)
    cfg.num_categories = len(categories)

    examples = load_data(split_path, data_format, categories)
    dataset = BSPARStage1Dataset(examples, cfg.model_name)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )
    return loader


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg, raw_cfg = load_config(args.config)
    loader = build_loader(cfg, raw_cfg, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSPARStage1(cfg).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    pair_thr = getattr(cfg, "stage1_pair_score_threshold", 0.01)
    pair_strategy = getattr(cfg, "stage1_pair_retention_strategy", "topn_only")
    pair_top_n = getattr(cfg, "stage1_pair_top_n", 20)
    max_span_len = int(getattr(cfg, "max_span_length", 8))

    counts = {
        "opinion_only_miss": 0,
        "both_miss": 0,
        "aspect_only_miss": 0,
    }
    a1_total = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                word_to_subword=batch["word_to_subword"],
                mode="inference",
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )

            asp_indices_batch = outputs.get("asp_indices", [])
            opn_indices_batch = outputs.get("opn_indices", [])

            for b_idx, gold_quads in enumerate(batch["gold_quads"]):
                if b_idx >= len(asp_indices_batch) or b_idx >= len(opn_indices_batch):
                    continue

                w2s = batch["word_to_subword"][b_idx]
                asp_set = set(tuple(x) for x in asp_indices_batch[b_idx])
                opn_set = set(tuple(x) for x in opn_indices_batch[b_idx])

                for g in gold_quads:
                    # Use the same assignment priority as analyze_stage1_a_breakdown.py.
                    a_sub = (
                        model._word_span_to_subword(g.aspect.start, g.aspect.end, w2s)
                        if not g.aspect.is_null else (-1, -1)
                    )
                    o_sub = (
                        model._word_span_to_subword(g.opinion.start, g.opinion.end, w2s)
                        if not g.opinion.is_null else (-1, -1)
                    )

                    a_over = (
                        a_sub is not None
                        and a_sub != (-1, -1)
                        and (a_sub[1] - a_sub[0] + 1) > max_span_len
                    )
                    o_over = (
                        o_sub is not None
                        and o_sub != (-1, -1)
                        and (o_sub[1] - o_sub[0] + 1) > max_span_len
                    )
                    if a_over or o_over:
                        continue  # A5
                    if g.aspect.is_null or g.opinion.is_null:
                        continue  # A4

                    a_in_topk = (a_sub is not None) and (tuple(a_sub) in asp_set)
                    o_in_topk = (o_sub is not None) and (tuple(o_sub) in opn_set)
                    if a_in_topk and o_in_topk:
                        continue  # Not A1

                    a1_total += 1
                    if (not a_in_topk) and (not o_in_topk):
                        counts["both_miss"] += 1
                    elif (not a_in_topk) and o_in_topk:
                        counts["aspect_only_miss"] += 1
                    elif a_in_topk and (not o_in_topk):
                        counts["opinion_only_miss"] += 1

    ratios = {
        k: (v / a1_total if a1_total > 0 else 0.0)
        for k, v in counts.items()
    }
    out = {
        "split": args.split,
        "A1_total": a1_total,
        "A1_split_counts": counts,
        "A1_split_ratio_over_A1": ratios,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
