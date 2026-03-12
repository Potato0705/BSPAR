"""Decompose Stage-1 A-type misses into A1-A5 on dev set.

A1. gold span not in top-k
A2. gold spans in top-k, but gold pair not in pair-space
A3. gold pair in pair-space, but not in retained top-n candidates
A4. implicit / NULL related miss
A5. max_span_length constraint miss
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data,
    build_category_map,
    get_categories_for_dataset,
    SENTIMENT_TO_ID,
)
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.models.bspar_stage1 import BSPARStage1
from bspar.utils.seed import set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = BSPARConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, cfg_dict


def gold_word_span(g_span):
    if g_span.is_null:
        return (-1, -1)
    return (g_span.start, g_span.end)


def match_gold_exact_in_stage1_candidates(candidates, gold, cat_to_id, task_type="asqp"):
    gold_a = gold_word_span(gold.aspect)
    gold_o = gold_word_span(gold.opinion)
    gold_cat_id = cat_to_id.get(gold.category, -1)
    gold_aff = SENTIMENT_TO_ID.get(gold.sentiment, -1) if task_type == "asqp" else None

    for c in candidates:
        if tuple(c["asp_span"]) != gold_a:
            continue
        if tuple(c["opn_span"]) != gold_o:
            continue
        if int(c["category_id"]) != gold_cat_id:
            continue
        if task_type == "asqp" and int(c["affective"]) != gold_aff:
            continue
        return True
    return False


def pick_representatives(records, n=20):
    buckets = defaultdict(list)
    for r in records:
        buckets[r["a_type"]].append(r)

    # Ensure class diversity first, then fill by major classes.
    out = []
    per_class_target = max(1, n // 5)
    for cls in ["A1", "A2", "A3", "A4", "A5"]:
        out.extend(buckets[cls][:per_class_target])
    if len(out) >= n:
        return out[:n]

    # Fill remaining slots by descending class size.
    class_order = sorted(
        ["A1", "A2", "A3", "A4", "A5"],
        key=lambda c: len(buckets[c]),
        reverse=True,
    )
    used_ids = set(id(x) for x in out)
    for cls in class_order:
        for rec in buckets[cls]:
            if id(rec) in used_ids:
                continue
            out.append(rec)
            used_ids.add(id(rec))
            if len(out) >= n:
                return out[:n]
    return out[:n]


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage-1 A-type breakdown")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    config, cfg_dict = load_config(args.config)

    dataset_name = cfg_dict.get("dataset_name", "asqp_rest15")
    data_format = cfg_dict.get("data_format", "auto")
    data_dir = cfg_dict.get("data_dir", "data/asqp_rest15")
    dev_file = os.path.join(data_dir, cfg_dict.get("dev_file", "dev.txt"))

    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, id_to_cat = build_category_map(categories)
    config.num_categories = len(categories)

    dev_examples = load_data(dev_file, data_format, categories)
    dev_dataset = BSPARStage1Dataset(dev_examples, config.model_name)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = BSPARStage1(config)
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt["model_state_dict"], strict=False)
    stage1.to(device)
    stage1.eval()

    pair_thr = getattr(config, "stage1_pair_score_threshold", 0.01)
    pair_strategy = getattr(config, "stage1_pair_retention_strategy", "topn_only")
    pair_top_n = getattr(config, "stage1_pair_top_n", 20)

    max_span_len = int(getattr(config, "max_span_length", 8))

    # A-type decomposition stats
    a_counts = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0}
    a3_sub = {"topn_drop": 0, "cat_aff_not_materialized": 0}

    total_gold = 0
    total_a_miss = 0

    records = []

    ex_idx = 0
    with torch.no_grad():
        for batch in dev_loader:
            outputs = stage1(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                word_to_subword=batch["word_to_subword"],
                mode="inference",
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )

            pair_map = outputs.get("pair_map", [])
            selected_pair_ids_batch = outputs.get("selected_pair_ids", [])
            asp_indices_batch = outputs.get("asp_indices", [])
            opn_indices_batch = outputs.get("opn_indices", [])

            bsz = len(outputs["candidates"])
            for b in range(bsz):
                ex = dev_examples[ex_idx]
                gold_quads = batch["gold_quads"][b]
                w2s = batch["word_to_subword"][b]
                candidates = outputs["candidates"][b]

                total_gold += len(gold_quads)

                asp_indices = asp_indices_batch[b]
                opn_indices = opn_indices_batch[b]
                asp_set = set(tuple(x) for x in asp_indices)
                opn_set = set(tuple(x) for x in opn_indices)

                pair_space = set()
                for pid, (ai, oi) in enumerate(pair_map):
                    pair_space.add((tuple(asp_indices[ai]), tuple(opn_indices[oi])))

                selected_ids = selected_pair_ids_batch[b] if b < len(selected_pair_ids_batch) else []
                retained_pairs = set()
                for pid in selected_ids:
                    if pid >= len(pair_map):
                        continue
                    ai, oi = pair_map[pid]
                    retained_pairs.add((tuple(asp_indices[ai]), tuple(opn_indices[oi])))

                for g in gold_quads:
                    if match_gold_exact_in_stage1_candidates(candidates, g, cat_to_id, config.task_type):
                        continue

                    total_a_miss += 1

                    is_implicit = g.aspect.is_null or g.opinion.is_null

                    a_sub = stage1._word_span_to_subword(g.aspect.start, g.aspect.end, w2s) if not g.aspect.is_null else (-1, -1)
                    o_sub = stage1._word_span_to_subword(g.opinion.start, g.opinion.end, w2s) if not g.opinion.is_null else (-1, -1)

                    # max-span check on subword lengths
                    a_over = False
                    o_over = False
                    if a_sub is not None and a_sub != (-1, -1):
                        a_over = (a_sub[1] - a_sub[0] + 1) > max_span_len
                    if o_sub is not None and o_sub != (-1, -1):
                        o_over = (o_sub[1] - o_sub[0] + 1) > max_span_len
                    hit_max_span = a_over or o_over

                    a_in_topk = True if g.aspect.is_null else (a_sub is not None and tuple(a_sub) in asp_set)
                    o_in_topk = True if g.opinion.is_null else (o_sub is not None and tuple(o_sub) in opn_set)

                    pair_key = None
                    in_pair_space = False
                    in_retained = False
                    if a_sub is not None and o_sub is not None:
                        pair_key = (tuple(a_sub), tuple(o_sub))
                        in_pair_space = pair_key in pair_space
                        in_retained = pair_key in retained_pairs

                    # Priority for mutually-exclusive assignment
                    if hit_max_span:
                        cls = "A5"
                        reason = "Gold span exceeds max_span_length"
                    elif is_implicit:
                        cls = "A4"
                        reason = "Implicit/NULL related miss"
                    elif not (a_in_topk and o_in_topk):
                        cls = "A1"
                        reason = "Gold aspect/opinion span missing from top-k"
                    elif not in_pair_space:
                        cls = "A2"
                        reason = "Gold spans present but pair missing in pair-space"
                    elif not in_retained:
                        cls = "A3"
                        a3_sub["topn_drop"] += 1
                        reason = "Gold pair in pair-space but dropped by top-n retention"
                    else:
                        cls = "A3"
                        a3_sub["cat_aff_not_materialized"] += 1
                        reason = "Gold pair retained, but correct cat/aff candidate not materialized"

                    a_counts[cls] += 1

                    records.append({
                        "a_type": cls,
                        "example_id": ex.id,
                        "example_index": ex_idx,
                        "text": ex.text,
                        "gold": {
                            "asp_span": gold_word_span(g.aspect),
                            "opn_span": gold_word_span(g.opinion),
                            "category": g.category,
                            "affective": g.sentiment,
                        },
                        "gold_subword": {
                            "asp": a_sub,
                            "opn": o_sub,
                        },
                        "is_implicit": is_implicit,
                        "hit_max_span": hit_max_span,
                        "a_in_topk": a_in_topk,
                        "o_in_topk": o_in_topk,
                        "in_pair_space": in_pair_space,
                        "in_retained_pair": in_retained,
                        "reason": reason,
                    })

                ex_idx += 1

    # Safety: ensure all A misses assigned.
    assigned = sum(a_counts.values())
    if assigned != total_a_miss:
        raise RuntimeError(f"Assigned {assigned} != total A misses {total_a_miss}")

    reps = pick_representatives(records, n=args.max_examples)

    out = {
        "config": args.config,
        "stage1_ckpt": args.stage1_ckpt,
        "max_span_length": max_span_len,
        "total_gold": total_gold,
        "total_A_miss": total_a_miss,
        "A_breakdown_counts": a_counts,
        "A3_subtypes": a3_sub,
        "A_breakdown_ratio_over_A": {
            k: (a_counts[k] / total_a_miss if total_a_miss > 0 else 0.0)
            for k in ["A1", "A2", "A3", "A4", "A5"]
        },
        "A_breakdown_ratio_over_gold": {
            k: (a_counts[k] / total_gold if total_gold > 0 else 0.0)
            for k in ["A1", "A2", "A3", "A4", "A5"]
        },
        "representative_examples": reps,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "total_gold": total_gold,
        "total_A_miss": total_a_miss,
        "A_breakdown_counts": a_counts,
        "A3_subtypes": a3_sub,
        "A_breakdown_ratio_over_A": out["A_breakdown_ratio_over_A"],
        "output": args.output,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
