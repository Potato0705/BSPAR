"""Diagnose Stage-1 pair score separation and calibration."""

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
from bspar.data.preprocessor import (
    build_category_map,
    get_categories_for_dataset,
    load_data,
)
from bspar.models.bspar_stage1 import BSPARStage1


def quantile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * q)
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def roc_auc_binary(scores, labels):
    """Mann-Whitney U based ROC-AUC with tie handling."""
    pairs = list(zip(scores, labels))
    n = len(pairs)
    if n == 0:
        return 0.0
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pairs.sort(key=lambda x: x[0])  # ascending by score
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = 0.0
    for r, (_, y) in zip(ranks, pairs):
        if y == 1:
            rank_sum_pos += r

    u = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def pr_auc_binary(scores, labels):
    """Average precision (step-wise PR-AUC) for binary labels."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0

    tp = 0
    ap_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            ap_sum += tp / rank
    return float(ap_sum / n_pos)


def summarize_group(scores):
    return {
        "count": len(scores),
        "q50": quantile(scores, 0.5),
        "q90": quantile(scores, 0.9),
        "q95": quantile(scores, 0.95),
        "mean": float(sum(scores) / len(scores)) if scores else 0.0,
    }


def build_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = BSPARConfig()
    for key, value in raw.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config, raw


def build_splits(config, raw_cfg):
    dataset_name = raw_cfg.get("dataset_name", "asqp_rest15")
    data_format = raw_cfg.get("data_format", "auto")
    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, _ = build_category_map(categories)
    config.num_categories = len(categories)

    data_dir = raw_cfg.get("data_dir", "data/asqp_rest15")
    train_file = os.path.join(data_dir, raw_cfg.get("train_file", "train.txt"))
    dev_file = os.path.join(data_dir, raw_cfg.get("dev_file", "dev.txt"))

    train_examples = load_data(train_file, data_format, categories)
    dev_examples = load_data(dev_file, data_format, categories)

    train_ds = BSPARStage1Dataset(
        train_examples,
        config.model_name,
        max_length=128,
        max_span_length=config.max_span_length,
    )
    dev_ds = BSPARStage1Dataset(
        dev_examples,
        config.model_name,
        max_length=128,
        max_span_length=config.max_span_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )

    return train_loader, dev_loader


def collect_pair_stats(model, loader):
    """Collect score distributions on one split."""
    device = next(model.parameters()).device
    all_scores = []
    all_labels = []

    scores_pos = []
    scores_easy = []
    scores_nearmiss = []
    scores_cat_confused = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_quads_batch = batch["gold_quads"]
            w2s_batch = batch["word_to_subword"]

            h = model.encoder(input_ids, attention_mask)
            proposal = model.span_proposal(h, attention_mask)
            pruned = model._prune_spans(proposal, mode="inference")
            pair_inputs, pair_map = model._construct_pairs(pruned)
            pair_out = model.pair_module(**pair_inputs)
            pair_probs = torch.sigmoid(pair_out["pair_scores"])

            for b in range(input_ids.size(0)):
                w2s = w2s_batch[b]
                gold_quads = gold_quads_batch[b]
                asp_indices = pruned["asp_indices"][b]
                opn_indices = pruned["opn_indices"][b]

                gold_pairs = set()
                gold_asp_spans = set()
                gold_opn_spans = set()

                for q in gold_quads:
                    if not q.aspect.is_null:
                        a_sub = model._word_span_to_subword(
                            q.aspect.start, q.aspect.end, w2s
                        )
                        if a_sub is not None:
                            gold_asp_spans.add(a_sub)
                    else:
                        a_sub = (-1, -1)

                    if not q.opinion.is_null:
                        o_sub = model._word_span_to_subword(
                            q.opinion.start, q.opinion.end, w2s
                        )
                        if o_sub is not None:
                            gold_opn_spans.add(o_sub)
                    else:
                        o_sub = (-1, -1)

                    if a_sub is not None and o_sub is not None:
                        gold_pairs.add((a_sub, o_sub))

                for p, (ai, oi) in enumerate(pair_map):
                    a_span = asp_indices[ai]
                    o_span = opn_indices[oi]
                    pair_key = (a_span, o_span)
                    score = pair_probs[b, p].item()

                    label = 1 if pair_key in gold_pairs else 0
                    all_scores.append(score)
                    all_labels.append(label)

                    if label == 1:
                        scores_pos.append(score)
                        continue

                    a_gold = a_span in gold_asp_spans and a_span != (-1, -1)
                    o_gold = o_span in gold_opn_spans and o_span != (-1, -1)
                    if a_gold and o_gold:
                        scores_cat_confused.append(score)
                    elif a_gold or o_gold:
                        scores_nearmiss.append(score)
                    else:
                        scores_easy.append(score)

    hard_scores = scores_nearmiss + scores_cat_confused
    return {
        "n_pairs": len(all_scores),
        "n_pos": sum(all_labels),
        "n_neg": len(all_labels) - sum(all_labels),
        "roc_auc": roc_auc_binary(all_scores, all_labels),
        "pr_auc": pr_auc_binary(all_scores, all_labels),
        "groups": {
            "positive": summarize_group(scores_pos),
            "easy_negative": summarize_group(scores_easy),
            "span_near_miss_negative": summarize_group(scores_nearmiss),
            "category_confused_negative": summarize_group(scores_cat_confused),
            "hard_negative_all": summarize_group(hard_scores),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Stage-1 pair scores")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config, raw_cfg = build_config(args.config)
    train_loader, dev_loader = build_splits(config, raw_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSPARStage1(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("Collecting train split statistics...")
    train_stats = collect_pair_stats(model, train_loader)
    print("Collecting dev split statistics...")
    dev_stats = collect_pair_stats(model, dev_loader)

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "retention_baseline": {
            "strategy": getattr(config, "stage1_pair_retention_strategy", "topn_only"),
            "top_n": getattr(config, "stage1_pair_top_n", 20),
        },
        "train": train_stats,
        "dev": dev_stats,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
