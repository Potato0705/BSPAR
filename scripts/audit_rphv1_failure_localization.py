import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bspar.config import BSPARConfig
from bspar.data.dataset import BSPARStage1Dataset, collate_stage1
from bspar.data.preprocessor import SENTIMENT_TO_ID, build_category_map, get_categories_for_dataset, load_data
from bspar.evaluation.metrics import compute_a3_diagnostics
from bspar.models.bspar_stage1 import BSPARStage1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--affacr_per_seed', default='outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_20260317_152500/summary/affacr_a0_4seed_per_seed.csv')
    p.add_argument('--rph_root', default='outputs/stage2_e2e_agmlbr_a0_rphv1_seed42_smoke_20260318_163444')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_reps', type=int, default=10)
    p.add_argument('--output_root', default=None)
    return p.parse_args()


def to_float(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == '':
        return None
    return float(v)


def load_cfg(cfg_path):
    raw = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    cfg = BSPARConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg, raw


def build_loader(cfg, raw_cfg, split, batch_size):
    dataset_name = raw_cfg.get('dataset_name', 'asqp_rest15')
    categories = get_categories_for_dataset(dataset_name)
    cat_to_id, _ = build_category_map(categories)
    cfg.num_categories = len(categories)
    data_dir = Path(raw_cfg.get('data_dir', 'data/asqp_rest15'))
    if split == 'dev':
        fn = raw_cfg.get('dev_file', 'dev.txt')
    elif split == 'test':
        fn = raw_cfg.get('test_file', 'test.txt')
    else:
        raise ValueError(split)
    examples = load_data(str(data_dir / fn), raw_cfg.get('data_format', 'auto'), categories)
    ds = BSPARStage1Dataset(examples, cfg.model_name, max_length=128, max_span_length=cfg.max_span_length, allow_offline_tokenizer_fallback=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stage1)
    return loader, examples, cat_to_id


def word_to_sub(model, span, w2s):
    if span == (-1, -1):
        return (-1, -1)
    sub = model._word_span_to_subword(span[0], span[1], w2s)
    return None if sub is None else tuple(sub)


def classify_other_type(pair_key, gold_asps, gold_opns):
    a, o = pair_key
    if a == (-1, -1) or o == (-1, -1):
        return 'NULL'
    if a in gold_asps or o in gold_opns:
        return 'near_miss'
    return 'other'


def replay_stage1(cfg_path, ckpt_path, split, batch_size):
    cfg, raw_cfg = load_cfg(cfg_path)
    loader, examples, cat_to_id = build_loader(cfg, raw_cfg, split, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BSPARStage1(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()

    pair_thr = float(getattr(cfg, 'stage1_pair_score_threshold', 0.001))
    pair_strategy = str(getattr(cfg, 'stage1_pair_retention_strategy', 'topn_only'))
    pair_top_n = int(getattr(cfg, 'stage1_pair_top_n', 20))
    top_c = int(getattr(cfg, 'top_c_categories', 3))

    records = {}
    ex_ptr = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                word_to_subword=batch['word_to_subword'],
                mode='inference',
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )
            pair_map = list(outputs['pair_map'])
            pair_scores = torch.sigmoid(outputs['pair_scores']).detach().cpu()
            cat_probs = torch.softmax(outputs['cat_logits'], dim=-1).detach().cpu()
            aff_preds = torch.argmax(outputs['aff_output'], dim=-1).detach().cpu()
            selected_batch = outputs.get('selected_pair_ids', [])
            asp_batch = outputs['asp_indices']
            opn_batch = outputs['opn_indices']
            cand_batch = outputs['candidates']

            for b in range(len(cand_batch)):
                ex = examples[ex_ptr]
                w2s = batch['word_to_subword'][b]
                gold_quads = batch['gold_quads'][b]

                asp_indices = [tuple(x) for x in asp_batch[b]]
                opn_indices = [tuple(x) for x in opn_batch[b]]
                asp_set = set(asp_indices)
                opn_set = set(opn_indices)

                pairid_to_key = {}
                pair_space = set()
                for pid, (ai, oi) in enumerate(pair_map):
                    if ai < len(asp_indices) and oi < len(opn_indices):
                        k = (asp_indices[ai], opn_indices[oi])
                        pairid_to_key[pid] = k
                        pair_space.add(k)

                sorted_ids = sorted(pairid_to_key.keys(), key=lambda pid: float(pair_scores[b, pid].item()), reverse=True)
                rank_map = {pairid_to_key[pid]: i + 1 for i, pid in enumerate(sorted_ids)}
                score_map = {pairid_to_key[pid]: float(pair_scores[b, pid].item()) for pid in pairid_to_key}

                selected_ids = list(selected_batch[b]) if b < len(selected_batch) else []
                selected_set = {pairid_to_key[pid] for pid in selected_ids if pid in pairid_to_key}

                topcat_map = {}
                topcat1prob_map = {}
                aff_map = {}
                for pid in selected_ids:
                    if pid not in pairid_to_key:
                        continue
                    k = pairid_to_key[pid]
                    probs = cat_probs[b, pid]
                    kk = min(top_c, probs.numel())
                    topcat_map[k] = [int(x) for x in torch.topk(probs, k=kk).indices.tolist()]
                    topcat1prob_map[k] = float(torch.max(probs).item())
                    aff_map[k] = int(aff_preds[b, pid].item())

                decode_score_map = {}
                rph_prob_map = {}
                for c in cand_batch[b]:
                    a_sub = word_to_sub(model, tuple(c['asp_span']), w2s)
                    o_sub = word_to_sub(model, tuple(c['opn_span']), w2s)
                    if a_sub is None or o_sub is None:
                        continue
                    pk = (a_sub, o_sub)
                    decode_score_map[pk] = float(c.get('pair_score', 0.0))
                    rph_prob_map[pk] = float(c.get('rph_probe_prob', 0.5))

                gold_items = []
                gold_pairs = set()
                gold_asps = set()
                gold_opns = set()
                materialized_quads = set()

                for q in gold_quads:
                    if q.aspect.is_null:
                        a_sub = (-1, -1)
                    else:
                        a_sub = model._word_span_to_subword(q.aspect.start, q.aspect.end, w2s)
                        if a_sub is None:
                            continue
                        a_sub = tuple(a_sub)
                        gold_asps.add(a_sub)
                    if q.opinion.is_null:
                        o_sub = (-1, -1)
                    else:
                        o_sub = model._word_span_to_subword(q.opinion.start, q.opinion.end, w2s)
                        if o_sub is None:
                            continue
                        o_sub = tuple(o_sub)
                        gold_opns.add(o_sub)
                    cat_id = int(cat_to_id[q.category]) if q.category in cat_to_id else -1
                    sent_id = int(SENTIMENT_TO_ID[q.sentiment]) if (cfg.task_type == 'asqp' and q.sentiment in SENTIMENT_TO_ID) else None
                    gold_items.append({'a_sub': a_sub, 'o_sub': o_sub, 'cat_id': cat_id, 'sent_id': sent_id, 'is_implicit': q.aspect.is_null or q.opinion.is_null})
                    gold_pairs.add((a_sub, o_sub))
                    pk = (a_sub, o_sub)
                    if pk in topcat_map and pk in aff_map:
                        cat_hit = (cat_id in topcat_map[pk]) if cat_id >= 0 else False
                        aff_hit = (sent_id is not None and aff_map[pk] == sent_id) if cfg.task_type == 'asqp' else True
                        if cat_hit and aff_hit:
                            materialized_quads.add((a_sub, o_sub, cat_id, sent_id))

                a3_record = {
                    'pair_scores': [float(x) for x in pair_scores[b].tolist()],
                    'pair_map': [tuple(x) for x in pair_map],
                    'asp_indices': asp_indices,
                    'opn_indices': opn_indices,
                    'selected_pair_ids': selected_ids,
                    'gold_pairs': [(tuple(x[0]), tuple(x[1])) for x in sorted(gold_pairs)],
                }

                records[ex.id] = {
                    'example_id': ex.id,
                    'gold_items': gold_items,
                    'gold_pairs': gold_pairs,
                    'gold_asps': gold_asps,
                    'gold_opns': gold_opns,
                    'asp_topk_set': asp_set,
                    'opn_topk_set': opn_set,
                    'pair_space_set': pair_space,
                    'selected_pair_set': selected_set,
                    'rank_map': rank_map,
                    'pair_score_map': score_map,
                    'pair_decode_score_map': decode_score_map,
                    'pair_rph_prob_map': rph_prob_map,
                    'pair_topcat_map': topcat_map,
                    'pair_topcat1prob_map': topcat1prob_map,
                    'pair_aff_pred_map': aff_map,
                    'materialized_quad_set': materialized_quads,
                    'a3_diag_record': a3_record,
                }
                ex_ptr += 1

    meta = {
        'pair_top_n': pair_top_n,
        'config_path': str(cfg_path),
        'checkpoint': str(ckpt_path),
        'split': split,
    }
    return records, meta


def compute_breakdown(records, max_span_length):
    a = {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0}
    a3 = {'topn_drop': 0, 'cat_aff_not_materialized': 0}
    a1s = {'opinion_only_miss': 0, 'both_miss': 0, 'aspect_only_miss': 0}
    total_gold = 0
    total_A = 0
    for rec in records.values():
        for g in rec['gold_items']:
            total_gold += 1
            qk = (g['a_sub'], g['o_sub'], g['cat_id'], g['sent_id'])
            if qk in rec['materialized_quad_set']:
                continue
            total_A += 1
            a_over = g['a_sub'] != (-1, -1) and (g['a_sub'][1] - g['a_sub'][0] + 1) > max_span_length
            o_over = g['o_sub'] != (-1, -1) and (g['o_sub'][1] - g['o_sub'][0] + 1) > max_span_length
            hit_max = a_over or o_over
            a_in = True if g['a_sub'] == (-1, -1) else (g['a_sub'] in rec['asp_topk_set'])
            o_in = True if g['o_sub'] == (-1, -1) else (g['o_sub'] in rec['opn_topk_set'])
            pk = (g['a_sub'], g['o_sub'])
            in_space = pk in rec['pair_space_set']
            in_ret = pk in rec['selected_pair_set']
            if hit_max:
                cls = 'A5'
            elif g['is_implicit']:
                cls = 'A4'
            elif not (a_in and o_in):
                cls = 'A1'
                if a_in and not o_in:
                    a1s['opinion_only_miss'] += 1
                elif (not a_in) and o_in:
                    a1s['aspect_only_miss'] += 1
                else:
                    a1s['both_miss'] += 1
            elif not in_space:
                cls = 'A2'
            elif not in_ret:
                cls = 'A3'
                a3['topn_drop'] += 1
            else:
                cls = 'A3'
                a3['cat_aff_not_materialized'] += 1
            a[cls] += 1
    return {'total_gold': total_gold, 'total_A_miss': total_A, 'A_breakdown_counts': a, 'A3_subtypes': a3, 'A1_split_counts': a1s}


def retained_overlap(left, right):
    ex_ids = sorted(set(left.keys()) & set(right.keys()))
    n = len(ex_ids)
    exact = 0
    jac = 0.0
    ov20 = 0.0
    lpos = 0
    rpos = 0
    for ex_id in ex_ids:
        ls = left[ex_id]['selected_pair_set']
        rs = right[ex_id]['selected_pair_set']
        inter = len(ls & rs)
        union = len(ls | rs)
        if ls == rs:
            exact += 1
        jac += (inter / union) if union > 0 else 1.0
        ov20 += inter / 20.0
        if left[ex_id]['gold_pairs'] & ls:
            lpos += 1
        if right[ex_id]['gold_pairs'] & rs:
            rpos += 1
    return {
        'num_examples': n,
        'retained_exact_match_ratio': (exact / n) if n else 0.0,
        'retained_jaccard_mean': (jac / n) if n else 0.0,
        'retained_overlap_at20_mean': (ov20 / n) if n else 0.0,
        'left_sample_has_positive_after_retention_ratio': (lpos / n) if n else 0.0,
        'right_sample_has_positive_after_retention_ratio': (rpos / n) if n else 0.0,
        'delta_sample_has_positive_after_retention_ratio': ((rpos - lpos) / n) if n else 0.0,
    }


def run_eval(cfg_path, stage1_ckpt, stage2_ckpt, seed, out_json):
    cmd = ['python', 'scripts/eval_stage2_dev.py', '--config', str(cfg_path), '--stage1_ckpt', str(stage1_ckpt), '--stage2_ckpt', str(stage2_ckpt), '--seed', str(seed), '--output', str(out_json)]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return json.loads(Path(out_json).read_text(encoding='utf-8'))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def build_mode_row(mode, breakdown, a3_diag, eval_dev, eval_test):
    ratios = a3_diag.get('first_outranker_type_ratio', {})
    return {
        'mode': mode,
        'dev_quad_f1': to_float(eval_dev.get('quad_f1')) if eval_dev else None,
        'test_quad_f1': to_float(eval_test.get('quad_f1')) if eval_test else None,
        'A1': int(breakdown['A_breakdown_counts']['A1']),
        'A1_opinion_only_miss': int(breakdown['A1_split_counts']['opinion_only_miss']),
        'A3': int(breakdown['A_breakdown_counts']['A3']),
        'A3_topn_drop': int(breakdown['A3_subtypes']['topn_drop']),
        'A3_cat_aff_not_materialized': int(breakdown['A3_subtypes']['cat_aff_not_materialized']),
        'A_total': int(breakdown['total_A_miss']),
        'sample_has_positive_after_retention_ratio': to_float(a3_diag.get('sample_has_positive_after_retention_ratio')),
        'first_outranker_null_ratio': to_float(ratios.get('NULL', {}).get('ratio')),
        'first_outranker_near_miss_ratio': to_float(ratios.get('near_miss', {}).get('ratio')),
        'first_outranker_other_ratio': to_float(ratios.get('other', {}).get('ratio')),
    }


def pick_representatives(base_records, rph_records, max_reps):
    rows = []
    for ex_id in sorted(set(base_records.keys()) & set(rph_records.keys())):
        b = base_records[ex_id]
        r = rph_records[ex_id]
        for g in r['gold_items']:
            pk = (g['a_sub'], g['o_sub'])
            qk = (g['a_sub'], g['o_sub'], g['cat_id'], g['sent_id'])
            if not (pk in b['selected_pair_set'] and pk in r['selected_pair_set']):
                continue
            if not (qk in b['materialized_quad_set'] and qk not in r['materialized_quad_set']):
                continue

            b_sorted = sorted(list(b['selected_pair_set']), key=lambda x: b['pair_decode_score_map'].get(x, b['pair_score_map'].get(x, 0.0)), reverse=True)
            r_sorted = sorted(list(r['selected_pair_set']), key=lambda x: r['pair_decode_score_map'].get(x, r['pair_score_map'].get(x, 0.0)), reverse=True)
            if pk not in b_sorted or pk not in r_sorted:
                continue
            b_rank = b_sorted.index(pk) + 1
            r_rank = r_sorted.index(pk) + 1

            other_pk = None
            for cand in r_sorted[:max(r_rank - 1, 0)]:
                if cand in r['gold_pairs']:
                    continue
                if classify_other_type(cand, r['gold_asps'], r['gold_opns']) == 'other':
                    other_pk = cand
                    break
            if other_pk is None:
                continue

            rows.append({
                'example_id': ex_id,
                'gold_pair_sub': str(pk),
                'gold_cat_id': int(g['cat_id']),
                'gold_sent_id': int(g['sent_id']) if g['sent_id'] is not None else None,
                'baseline_rank_in_retained': b_rank,
                'rph_rank_in_retained': r_rank,
                'baseline_decode_score': b['pair_decode_score_map'].get(pk, b['pair_score_map'].get(pk, 0.0)),
                'rph_decode_score': r['pair_decode_score_map'].get(pk, r['pair_score_map'].get(pk, 0.0)),
                'rph_probe_prob_gold': r['pair_rph_prob_map'].get(pk),
                'outranker_pair_sub': str(other_pk),
                'outranker_decode_score': r['pair_decode_score_map'].get(other_pk, r['pair_score_map'].get(other_pk, 0.0)),
                'outranker_pair_score': r['pair_score_map'].get(other_pk, 0.0),
                'outranker_rph_prob': r['pair_rph_prob_map'].get(other_pk),
                'outranker_topcat_ids': str(r['pair_topcat_map'].get(other_pk, [])),
                'outranker_topcat1_prob': r['pair_topcat1prob_map'].get(other_pk),
                'outranker_aff_pred': r['pair_aff_pred_map'].get(other_pk),
                'outranker_type': 'other',
            })
            if len(rows) >= max_reps:
                return rows
    return rows


def main():
    args = parse_args()
    seed = int(args.seed)
    rph_root = Path(args.rph_root).resolve()
    if args.output_root:
        out_root = Path(args.output_root).resolve()
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = (rph_root / 'summary' / f'rph_failure_localization_audit_{ts}').resolve()
    summary_dir = out_root / 'summary'
    notes_dir = out_root / 'notes'
    cfg_dir = out_root / 'configs'
    summary_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    aff_rows = list(csv.DictReader(Path(args.affacr_per_seed).open('r', encoding='utf-8')))
    aff_row = next(r for r in aff_rows if int(r['seed']) == seed)
    aff_stage1_ckpt = Path(aff_row['stage1_ckpt']).resolve()

    manifests = list((rph_root / 'runs').glob('*/run_manifest.json'))
    if not manifests:
        raise RuntimeError(f'No run_manifest under {rph_root / "runs"}')
    rph_manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    rph_stage1_ckpt = Path(rph_manifest['stage1_ckpt']).resolve()
    rph_stage2_ckpt = Path(rph_manifest['stage2_ckpt']).resolve()
    rph_train_cfg_on = Path(rph_manifest['train_config']).resolve()
    rph_testeval_cfg_on = Path(rph_manifest['testeval_config']).resolve()

    aff_manifest = json.loads((aff_stage1_ckpt.parent / 'manifest.json').read_text(encoding='utf-8'))
    aff_train_cfg = Path(aff_manifest['config_path']).resolve()
    aff_test_raw = yaml.safe_load(aff_train_cfg.read_text(encoding='utf-8'))
    aff_test_raw['dev_file'] = aff_test_raw.get('test_file', 'test.txt')
    aff_testeval_cfg = cfg_dir / 'affacr_testeval.yaml'
    aff_testeval_cfg.write_text(yaml.safe_dump(aff_test_raw, sort_keys=False, allow_unicode=True), encoding='utf-8')

    on_cfg = yaml.safe_load(rph_train_cfg_on.read_text(encoding='utf-8'))
    on_t_cfg = yaml.safe_load(rph_testeval_cfg_on.read_text(encoding='utf-8'))
    off_cfg = dict(on_cfg)
    off_cfg['use_rph_v1_decode_reweight'] = False
    off_t_cfg = dict(on_t_cfg)
    off_t_cfg['use_rph_v1_decode_reweight'] = False
    rph_train_cfg_off = cfg_dir / 'rph_decode_off_train.yaml'
    rph_testeval_cfg_off = cfg_dir / 'rph_decode_off_testeval.yaml'
    rph_train_cfg_off.write_text(yaml.safe_dump(off_cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    rph_testeval_cfg_off.write_text(yaml.safe_dump(off_t_cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')

    aff_dev, _ = replay_stage1(aff_train_cfg, aff_stage1_ckpt, 'dev', args.batch_size)
    aff_test, _ = replay_stage1(aff_testeval_cfg, aff_stage1_ckpt, 'test', args.batch_size)
    rph_on_dev, _ = replay_stage1(rph_train_cfg_on, rph_stage1_ckpt, 'dev', args.batch_size)
    rph_on_test, rph_on_meta = replay_stage1(rph_testeval_cfg_on, rph_stage1_ckpt, 'test', args.batch_size)
    rph_off_dev, _ = replay_stage1(rph_train_cfg_off, rph_stage1_ckpt, 'dev', args.batch_size)
    rph_off_test, rph_off_meta = replay_stage1(rph_testeval_cfg_off, rph_stage1_ckpt, 'test', args.batch_size)

    taskA = {
        'dev_affacr_vs_rph_on': retained_overlap(aff_dev, rph_on_dev),
        'test_affacr_vs_rph_on': retained_overlap(aff_test, rph_on_test),
        'dev_rph_off_vs_on': retained_overlap(rph_off_dev, rph_on_dev),
        'test_rph_off_vs_on': retained_overlap(rph_off_test, rph_on_test),
    }
    aff_break_test = compute_breakdown(aff_test, max_span_length=8)
    rph_on_break_test = compute_breakdown(rph_on_test, max_span_length=8)
    taskA['A3_topn_drop_affacr_test'] = int(aff_break_test['A3_subtypes']['topn_drop'])
    taskA['A3_topn_drop_rph_on_test'] = int(rph_on_break_test['A3_subtypes']['topn_drop'])
    taskA['delta_A3_topn_drop_test'] = int(rph_on_break_test['A3_subtypes']['topn_drop']) - int(aff_break_test['A3_subtypes']['topn_drop'])
    (summary_dir / 'taskA_retained_set_audit.json').write_text(json.dumps(taskA, indent=2, ensure_ascii=False), encoding='utf-8')

    eval_on_dev = json.loads(Path(rph_manifest['eval_dev_json']).read_text(encoding='utf-8'))
    eval_on_test = json.loads(Path(rph_manifest['eval_test_json']).read_text(encoding='utf-8'))
    eval_off_dev = run_eval(rph_train_cfg_off, rph_stage1_ckpt, rph_stage2_ckpt, seed, summary_dir / 'taskB_eval_rph_decode_off_dev.json')
    eval_off_test = run_eval(rph_testeval_cfg_off, rph_stage1_ckpt, rph_stage2_ckpt, seed, summary_dir / 'taskB_eval_rph_decode_off_test.json')

    a3_on = compute_a3_diagnostics([r['a3_diag_record'] for r in rph_on_test.values()], int(rph_on_meta['pair_top_n']))
    a3_off = compute_a3_diagnostics([r['a3_diag_record'] for r in rph_off_test.values()], int(rph_off_meta['pair_top_n']))
    brk_on = rph_on_break_test
    brk_off = compute_breakdown(rph_off_test, max_span_length=8)

    mode_off = build_mode_row('rph_decode_off', brk_off, a3_off, eval_off_dev, eval_off_test)
    mode_on = build_mode_row('rph_decode_on', brk_on, a3_on, eval_on_dev, eval_on_test)
    rows = [mode_off, mode_on]
    write_csv(summary_dir / 'taskB_rph_decode_on_off_compare.csv', rows)
    delta = {}
    for k in mode_on.keys():
        if k == 'mode':
            continue
        vo = mode_on.get(k)
        vf = mode_off.get(k)
        delta[f'delta_on_minus_off_{k}'] = (float(vo) - float(vf)) if (vo is not None and vf is not None) else None
    (summary_dir / 'taskB_rph_decode_on_off_delta.json').write_text(json.dumps(delta, indent=2, ensure_ascii=False), encoding='utf-8')

    reps = pick_representatives(aff_test, rph_on_test, args.max_reps)
    write_csv(summary_dir / 'taskC_harmful_other_representatives.csv', reps)
    (summary_dir / 'taskC_harmful_other_summary.json').write_text(json.dumps({'num_representatives': len(reps)}, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# RPH-v1 Failure Localization Audit',
        '',
        f"- A task output: {summary_dir / 'taskA_retained_set_audit.json'}",
        f"- B task output: {summary_dir / 'taskB_rph_decode_on_off_compare.csv'}",
        f"- C task output: {summary_dir / 'taskC_harmful_other_representatives.csv'}",
    ]
    (summary_dir / 'rph_failure_localization_takeaways.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    (notes_dir / 'audit_manifest.json').write_text(json.dumps({
        'seed': seed,
        'aff_stage1_ckpt': str(aff_stage1_ckpt),
        'rph_stage1_ckpt': str(rph_stage1_ckpt),
        'rph_stage2_ckpt': str(rph_stage2_ckpt),
        'rph_train_cfg_on': str(rph_train_cfg_on),
        'rph_testeval_cfg_on': str(rph_testeval_cfg_on),
        'rph_train_cfg_off': str(rph_train_cfg_off),
        'rph_testeval_cfg_off': str(rph_testeval_cfg_off),
        'output_root': str(out_root),
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Wrote audit outputs under: {out_root}')


if __name__ == '__main__':
    main()
