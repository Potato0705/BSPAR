"""Evaluate Stage-2 reranker on full dev with fixed Stage-1 candidates."""

import os
import sys
import json
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data,
    build_category_map,
    get_categories_for_dataset,
    ID_TO_SENTIMENT,
)
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.models.bspar_stage1 import BSPARStage1
from bspar.models.bspar_stage2 import BSPARStage2
from bspar.evaluation.metrics import compute_quad_f1
from bspar.utils.seed import set_seed


def qval(values, q):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * q)
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def match_gold(candidate, gold_quads, id_to_cat, task_type="asqp"):
    for gq in gold_quads:
        asp_match = (
            (candidate["asp_span"] == (-1, -1) and gq.aspect.is_null)
            or (candidate["asp_span"] == (gq.aspect.start, gq.aspect.end))
        )
        opn_match = (
            (candidate["opn_span"] == (-1, -1) and gq.opinion.is_null)
            or (candidate["opn_span"] == (gq.opinion.start, gq.opinion.end))
        )
        cat_match = (id_to_cat.get(candidate["category_id"]) == gq.category)

        if task_type == "asqp":
            pred_sent = ID_TO_SENTIMENT.get(int(candidate["affective"]), "NEU")
            aff_match = pred_sent == gq.sentiment
        else:
            v_pred, a_pred = candidate["affective"]
            v_diff = abs(v_pred - (gq.valence or 0.0))
            a_diff = abs(a_pred - (gq.arousal or 0.0))
            aff_match = (v_diff < 0.5) and (a_diff < 0.5)

        if asp_match and opn_match and cat_match and aff_match:
            return 1
    return 0


def build_group_keys(candidates):
    """Build (asp_idx_or_NULL, opn_idx_or_NULL) keys without strings."""
    asp_map = {}
    opn_map = {}
    keys = []

    for c in candidates:
        a = tuple(c["asp_span"])
        o = tuple(c["opn_span"])

        if a == (-1, -1):
            a_idx = -1
        else:
            if a not in asp_map:
                asp_map[a] = len(asp_map)
            a_idx = asp_map[a]

        if o == (-1, -1):
            o_idx = -1
        else:
            if o not in opn_map:
                opn_map[o] = len(opn_map)
            o_idx = opn_map[o]

        keys.append((a_idx, o_idx))

    return keys


def update_group_stats(stats, candidates, scores, labels):
    keys = build_group_keys(candidates)
    group_to_ids = {}
    for i, key in enumerate(keys):
        group_to_ids.setdefault(key, []).append(i)

    for ids in group_to_ids.values():
        if len(ids) < 2:
            continue
        stats["group_count_all"] += 1

        g_scores = [scores[i] for i in ids]
        g_labels = [labels[i] for i in ids]

        top_i = max(range(len(ids)), key=lambda j: g_scores[j])
        top_is_pos = 1.0 if g_labels[top_i] == 1 else 0.0
        stats["group_acc_correct_all"] += top_is_pos

        has_pos = any(v == 1 for v in g_labels)
        has_neg = any(v == 0 for v in g_labels)
        if not (has_pos and has_neg):
            continue

        stats["group_count_conflict"] += 1
        stats["group_acc_correct_conflict"] += top_is_pos
        stats["group_hits1_conflict"] += top_is_pos

        sorted_ids = sorted(range(len(ids)), key=lambda j: g_scores[j], reverse=True)
        rr = 0.0
        for rank, local_idx in enumerate(sorted_ids, start=1):
            if g_labels[local_idx] == 1:
                rr = 1.0 / rank
                break
        stats["group_rr_sum_conflict"] += rr


def decode_topk(candidates, scores, max_pred=10):
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    selected = []
    seen = set()
    for i in order:
        c = candidates[i]
        key = (tuple(c["asp_span"]), tuple(c["opn_span"]), int(c["category_id"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "asp_span": tuple(c["asp_span"]),
            "opn_span": tuple(c["opn_span"]),
            "category_id": int(c["category_id"]),
            "affective": int(c["affective"]) if isinstance(c["affective"], int) else c["affective"],
            "pair_score": float(scores[i]),
            "cat_prob": float(c.get("cat_prob", 0.0)),
        })
        if len(selected) >= max_pred:
            break
    return selected


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = BSPARConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, cfg_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 on full dev")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pred", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
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
    ckpt1 = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt1["model_state_dict"])
    stage1.to(device)
    stage1.eval()

    stage2 = BSPARStage2(config)
    ckpt2 = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    stage2.load_state_dict(ckpt2["model_state_dict"])
    stage2.to(device)
    stage2.eval()

    pair_thr = getattr(config, "stage1_pair_score_threshold", 0.01)
    pair_strategy = getattr(config, "stage1_pair_retention_strategy", "topn_only")
    pair_top_n = getattr(config, "stage1_pair_top_n", 20)

    all_preds = []
    all_golds = []

    group_stats = {
        "group_count_all": 0,
        "group_count_conflict": 0,
        "group_acc_correct_all": 0.0,
        "group_acc_correct_conflict": 0.0,
        "group_hits1_conflict": 0.0,
        "group_rr_sum_conflict": 0.0,
    }

    pos_scores = []
    neg_scores = []

    ex_ptr = 0
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

            batch_size = len(outputs["candidates"])
            for b in range(batch_size):
                candidates = outputs["candidates"][b]
                gold_quads = batch["gold_quads"][b]
                all_golds.append(gold_quads)

                if not candidates:
                    all_preds.append([])
                    ex_ptr += 1
                    continue

                pair_reprs = torch.stack([c["pair_repr"] for c in candidates]).to(device)
                cat_ids = torch.tensor([c["category_id"] for c in candidates], dtype=torch.long, device=device)
                if config.task_type == "asqp":
                    aff_input = torch.tensor([c["affective"] for c in candidates], dtype=torch.long, device=device)
                else:
                    aff_input = torch.tensor([c["affective"] for c in candidates], dtype=torch.float, device=device)
                meta = torch.tensor([c["meta_features"] for c in candidates], dtype=torch.float, device=device)

                stage2_out = stage2(
                    pair_reprs=pair_reprs.unsqueeze(0),
                    cat_ids=cat_ids.unsqueeze(0),
                    aff_input=aff_input.unsqueeze(0),
                    meta_features=meta.unsqueeze(0),
                    mode="inference",
                )
                scores = stage2_out["quad_scores"][0].detach().cpu().tolist()

                labels = [match_gold(c, gold_quads, id_to_cat, config.task_type) for c in candidates]
                for s, y in zip(scores, labels):
                    if y == 1:
                        pos_scores.append(float(s))
                    else:
                        neg_scores.append(float(s))

                update_group_stats(group_stats, candidates, scores, labels)

                preds = decode_topk(candidates, scores, max_pred=args.max_pred)
                all_preds.append(preds)
                ex_ptr += 1

    quad_metrics = compute_quad_f1(all_preds, all_golds, id_to_cat, cat_to_id)

    n_group_all = max(group_stats["group_count_all"], 1)
    n_group_conflict = max(group_stats["group_count_conflict"], 1)

    out = {
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "dev_examples": len(all_golds),
        "quad_f1": float(quad_metrics["quad_f1"]),
        "span_f1": float(quad_metrics["span_f1"]),
        "group_accuracy": float(group_stats["group_acc_correct_all"] / n_group_all),
        "conflict_group_accuracy": float(group_stats["group_acc_correct_conflict"] / n_group_conflict),
        "group_mrr": float(group_stats["group_rr_sum_conflict"] / n_group_conflict),
        "group_hits1": float(group_stats["group_hits1_conflict"] / n_group_conflict),
        "group_count_all": int(group_stats["group_count_all"]),
        "group_count_conflict": int(group_stats["group_count_conflict"]),
        "score_quantiles": {
            "pos_q50": qval(pos_scores, 0.5),
            "pos_q90": qval(pos_scores, 0.9),
            "pos_q95": qval(pos_scores, 0.95),
            "neg_q50": qval(neg_scores, 0.5),
            "neg_q90": qval(neg_scores, 0.9),
            "neg_q95": qval(neg_scores, 0.95),
        },
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
