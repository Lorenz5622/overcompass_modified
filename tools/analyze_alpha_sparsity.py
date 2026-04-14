#!/usr/bin/env python3
"""Compare alpha-dependent routing sparsity and top-p expert-count usage.

Expected input schema:
results/.../piqa.json
  - details[sample_id].origin_prediction[option_id].probs.routing_eval

The script supports two levels of analysis:
1. Sample-level metrics already aggregated in ``routing_eval.global`` / ``by_layer``
2. Token-level top-p expert counts from
   ``by_layer[layer].top_p_expert_count_tokens`` when available
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


RecordKey = Tuple[str, str]  # (sample_id, option_id)
TokenRecordKey = Tuple[RecordKey, str, int]  # ((sample_id, option_id), layer, token_idx)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _layer_sort_key(name: str):
    if name.startswith('layer'):
        suffix = name[5:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def load_records(path: str) -> Dict[RecordKey, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get('details', {})
    records: Dict[RecordKey, dict] = {}

    for sample_id, item in details.items():
        if sample_id == 'type':
            continue

        ref = str(item.get('references')) if item.get('references') is not None else None
        pred = str(item.get('predictions')) if item.get('predictions') is not None else None
        sample_correct = bool(item.get('correct')) if item.get('correct') is not None else None

        origin = item.get('origin_prediction', {})
        if not isinstance(origin, dict):
            continue

        for option_id, option_payload in origin.items():
            if not str(option_id).isdigit():
                continue
            if not isinstance(option_payload, dict):
                continue

            probs = option_payload.get('probs', {})
            if not isinstance(probs, dict):
                continue
            routing_eval = probs.get('routing_eval')
            if not isinstance(routing_eval, dict):
                continue

            global_stats = routing_eval.get('global', {})
            by_layer = routing_eval.get('by_layer', {})
            if not isinstance(global_stats, dict):
                global_stats = {}
            if not isinstance(by_layer, dict):
                by_layer = {}

            key = (str(sample_id), str(option_id))
            records[key] = {
                'sample_id': str(sample_id),
                'option_id': str(option_id),
                'is_reference': (ref is not None and str(option_id) == ref),
                'is_prediction': (pred is not None and str(option_id) == pred),
                'sample_correct': sample_correct,
                'global': global_stats,
                'by_layer': by_layer,
            }

    return records


def sign_test_pvalue(deltas: Iterable[float]) -> Optional[float]:
    pos = 0
    neg = 0
    for d in deltas:
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return p


def paired_stats(vals_a: List[float], vals_b: List[float]) -> dict:
    deltas = [a - b for a, b in zip(vals_a, vals_b)]
    if not deltas:
        return {
            'n': 0,
            'mean_a': None,
            'mean_b': None,
            'mean_delta': None,
            'median_delta': None,
            'pos_ratio': None,
            'sign_test_pvalue': None,
        }
    pos = sum(1 for d in deltas if d > 0)
    return {
        'n': len(deltas),
        'mean_a': mean(vals_a),
        'mean_b': mean(vals_b),
        'mean_delta': mean(deltas),
        'median_delta': median(deltas),
        'pos_ratio': pos / len(deltas),
        'sign_test_pvalue': sign_test_pvalue(deltas),
    }


def select_pairs(
    rec_a: Dict[RecordKey, dict],
    rec_b: Dict[RecordKey, dict],
    metric: str,
    subset: str,
) -> Tuple[List[float], List[float], List[RecordKey]]:
    common_keys = sorted(set(rec_a.keys()) & set(rec_b.keys()))

    vals_a: List[float] = []
    vals_b: List[float] = []
    used_keys: List[RecordKey] = []

    for k in common_keys:
        ra = rec_a[k]
        rb = rec_b[k]

        if subset == 'reference_only' and not (ra.get('is_reference') and rb.get('is_reference')):
            continue
        if subset == 'prediction_only' and not (ra.get('is_prediction') and rb.get('is_prediction')):
            continue

        a = _safe_float(ra.get('global', {}).get(metric))
        b = _safe_float(rb.get('global', {}).get(metric))
        if a is None or b is None:
            continue

        vals_a.append(a)
        vals_b.append(b)
        used_keys.append(k)

    return vals_a, vals_b, used_keys


def layerwise_deltas(
    rec_a: Dict[RecordKey, dict],
    rec_b: Dict[RecordKey, dict],
    used_keys: List[RecordKey],
    metric: str,
) -> dict:
    bucket = defaultdict(list)

    for k in used_keys:
        ba = rec_a[k].get('by_layer', {})
        bb = rec_b[k].get('by_layer', {})
        if not isinstance(ba, dict) or not isinstance(bb, dict):
            continue

        for layer in set(ba.keys()) & set(bb.keys()):
            va = _safe_float(ba[layer].get(metric) if isinstance(ba[layer], dict) else None)
            vb = _safe_float(bb[layer].get(metric) if isinstance(bb[layer], dict) else None)
            if va is None or vb is None:
                continue
            bucket[layer].append(va - vb)

    out = {}
    for layer, deltas in bucket.items():
        pos = sum(1 for d in deltas if d > 0)
        out[layer] = {
            'n': len(deltas),
            'mean_delta': mean(deltas),
            'median_delta': median(deltas),
            'pos_ratio': pos / len(deltas),
            'sign_test_pvalue': sign_test_pvalue(deltas),
        }

    return dict(sorted(out.items(), key=lambda x: _layer_sort_key(x[0])))


def select_top_p_token_pairs(
    rec_a: Dict[RecordKey, dict],
    rec_b: Dict[RecordKey, dict],
    subset: str,
) -> Tuple[List[float], List[float], List[TokenRecordKey], List[float]]:
    common_keys = sorted(set(rec_a.keys()) & set(rec_b.keys()))
    vals_a: List[float] = []
    vals_b: List[float] = []
    used_tokens: List[TokenRecordKey] = []
    top_ps = set()

    for k in common_keys:
        ra = rec_a[k]
        rb = rec_b[k]

        if subset == 'reference_only' and not (ra.get('is_reference') and rb.get('is_reference')):
            continue
        if subset == 'prediction_only' and not (ra.get('is_prediction') and rb.get('is_prediction')):
            continue

        ba = ra.get('by_layer', {})
        bb = rb.get('by_layer', {})
        if not isinstance(ba, dict) or not isinstance(bb, dict):
            continue

        for layer in sorted(set(ba.keys()) & set(bb.keys()), key=_layer_sort_key):
            la = ba[layer] if isinstance(ba[layer], dict) else {}
            lb = bb[layer] if isinstance(bb[layer], dict) else {}
            ta = la.get('top_p_expert_count_tokens')
            tb = lb.get('top_p_expert_count_tokens')
            if not isinstance(ta, list) or not isinstance(tb, list):
                continue

            p_a = _safe_float(la.get('routing_top_p'))
            p_b = _safe_float(lb.get('routing_top_p'))
            if p_a is not None:
                top_ps.add(p_a)
            if p_b is not None:
                top_ps.add(p_b)

            limit = min(len(ta), len(tb))
            for token_idx in range(limit):
                va = _safe_float(ta[token_idx])
                vb = _safe_float(tb[token_idx])
                if va is None or vb is None:
                    continue
                vals_a.append(va)
                vals_b.append(vb)
                used_tokens.append((k, layer, token_idx))

    return vals_a, vals_b, used_tokens, sorted(top_ps)


def layerwise_token_deltas(
    used_tokens: List[TokenRecordKey],
    vals_a: List[float],
    vals_b: List[float],
) -> dict:
    bucket = defaultdict(list)
    for token_key, va, vb in zip(used_tokens, vals_a, vals_b):
        _, layer, _ = token_key
        bucket[layer].append(va - vb)

    out = {}
    for layer, deltas in bucket.items():
        pos = sum(1 for d in deltas if d > 0)
        out[layer] = {
            'n': len(deltas),
            'mean_delta': mean(deltas),
            'median_delta': median(deltas),
            'pos_ratio': pos / len(deltas),
            'sign_test_pvalue': sign_test_pvalue(deltas),
        }
    return dict(sorted(out.items(), key=lambda x: _layer_sort_key(x[0])))


def build_histogram(values: Iterable[float]) -> dict:
    counts = defaultdict(int)
    total = 0
    for v in values:
        counts[int(round(v))] += 1
        total += 1
    return {
        str(k): {
            'count': counts[k],
            'ratio': counts[k] / total if total else None,
        }
        for k in sorted(counts.keys())
    }



def fmt(x: Optional[float], nd=6) -> str:
    if x is None:
        return 'NA'
    return f'{x:.{nd}f}'


def print_summary(title: str, stats: dict, label_a: str, label_b: str):
    print(f'\n[{title}]')
    print(f"n={stats['n']}")
    print(f"mean({label_a})={fmt(stats['mean_a'])}")
    print(f"mean({label_b})={fmt(stats['mean_b'])}")
    print(f"mean_delta({label_a}-{label_b})={fmt(stats['mean_delta'])}")
    print(f"median_delta({label_a}-{label_b})={fmt(stats['median_delta'])}")
    print(f"pos_ratio(delta>0)={fmt(stats['pos_ratio'])}")
    print(f"sign_test_pvalue={fmt(stats['sign_test_pvalue'])}")


def print_layerwise(title: str, layer_stats: dict, delta_label: str):
    print(f'\n[{title}]')
    print(f'layer\tn\tmean_delta({delta_label})\tmedian_delta\tpos_ratio\tpvalue')
    for layer, st in layer_stats.items():
        print(
            f"{layer}\t{st['n']}\t{fmt(st['mean_delta'])}\t"
            f"{fmt(st['median_delta'])}\t{fmt(st['pos_ratio'])}\t"
            f"{fmt(st['sign_test_pvalue'])}")


def print_histogram(title: str, hist: dict):
    print(f'\n[{title}]')
    print('expert_count\tcount\tratio')
    for expert_count, payload in hist.items():
        print(f"{expert_count}\t{payload['count']}\t{fmt(payload['ratio'])}")


def compare_metric(
    rec15: Dict[RecordKey, dict],
    rec17: Dict[RecordKey, dict],
    metric: str,
    subset: str,
) -> Tuple[dict, dict, List[RecordKey]]:
    vals15, vals17, used_keys = select_pairs(rec15, rec17, metric=metric, subset=subset)
    global_stats = paired_stats(vals15, vals17)
    layer_metric = metric
    if metric == 'top1_expert_unique_mean':
        layer_metric = 'top1_expert_unique'
    layer_stats = layerwise_deltas(rec15, rec17, used_keys, metric=layer_metric)
    return global_stats, layer_stats, used_keys


def compare_top_p_token_counts(
    rec15: Dict[RecordKey, dict],
    rec17: Dict[RecordKey, dict],
    subset: str,
) -> Tuple[dict, dict, dict, dict, List[float]]:
    vals15, vals17, used_tokens, top_ps = select_top_p_token_pairs(
        rec15, rec17, subset=subset)
    global_stats = paired_stats(vals15, vals17)
    layer_stats = layerwise_token_deltas(used_tokens, vals15, vals17)
    hist15 = build_histogram(vals15)
    hist17 = build_histogram(vals17)
    return global_stats, layer_stats, hist15, hist17, top_ps


def main():
    parser = argparse.ArgumentParser(
        description='Analyze whether alpha changes MoE sparsity and expert-count usage.')
    parser.add_argument('--alpha15', required=True,
                        help='Path to result json for alpha=1.5')
    parser.add_argument('--alpha17', required=True,
                        help='Path to result json for alpha=1.7')
    parser.add_argument('--metric', default='nz_expert_mean',
                        choices=['nz_expert_mean', 'entropy_mean', 'top1_prob_mean',
                                 'top1_top2_margin_mean', 'top4_mass_mean',
                                 'top1_expert_unique_mean', 'top_p_expert_count_mean'],
                        help='Primary global metric to compare (default: nz_expert_mean).')
    parser.add_argument('--subset', default='all',
                        choices=['all', 'reference_only', 'prediction_only'],
                        help='Which option pairs to compare.')
    parser.add_argument('--with_expert_count', action='store_true',
                        help='Also report expert count usage metric top1_expert_unique_mean.')
    parser.add_argument('--with_top_p_token_count', action='store_true',
                        help='Also report token-level top-p expert-count statistics if present.')
    parser.add_argument('--out', default=None,
                        help='Optional output json path for summary.')
    args = parser.parse_args()

    rec15 = load_records(args.alpha15)
    rec17 = load_records(args.alpha17)

    print('Alpha routing comparison')
    print(f'alpha15 file: {args.alpha15}')
    print(f'alpha17 file: {args.alpha17}')
    print(f'primary metric: {args.metric}')
    print(f'subset: {args.subset}')

    global_primary, layer_primary, _ = compare_metric(
        rec15, rec17, metric=args.metric, subset=args.subset)
    print_summary(f'Global ({args.metric})', global_primary, 'alpha1.5', 'alpha1.7')

    layer_label = '1.5-1.7'
    primary_layer_metric = args.metric if args.metric != 'top1_expert_unique_mean' else 'top1_expert_unique'
    print_layerwise(f'Layerwise delta on {primary_layer_metric}', layer_primary, layer_label)

    interpretations: List[str] = []
    if args.metric == 'nz_expert_mean':
        interpretations.append(
            'nz_expert_mean 的 delta(1.5-1.7) > 0 表示 alpha=1.7 更稀疏；< 0 表示 alpha=1.5 更稀疏。')
    if args.metric == 'top_p_expert_count_mean':
        interpretations.append(
            'top_p_expert_count_mean 的 delta(1.5-1.7) > 0 表示 alpha=1.5 在 top-p 决策下平均为每个 token 保留了更多专家。')

    expert_count_summary = None
    if args.with_expert_count or args.metric != 'top1_expert_unique_mean':
        g_cnt, l_cnt, _ = compare_metric(
            rec15, rec17, metric='top1_expert_unique_mean', subset=args.subset)
        print_summary('Global (top1_expert_unique_mean)', g_cnt, 'alpha1.5', 'alpha1.7')
        print_layerwise('Layerwise delta on top1_expert_unique', l_cnt, layer_label)
        interpretations.append(
            'top1_expert_unique_mean 的 delta(1.5-1.7) > 0 表示 alpha=1.5 在单样本中调用了更多不同 top1 专家。')
        expert_count_summary = {
            'global_top1_expert_unique_mean': g_cnt,
            'layerwise_top1_expert_unique': l_cnt,
        }

    top_p_summary = None
    if args.with_top_p_token_count or args.metric == 'top_p_expert_count_mean':
        if args.metric != 'top_p_expert_count_mean':
            g_top_p, l_top_p, _ = compare_metric(
                rec15, rec17, metric='top_p_expert_count_mean', subset=args.subset)
            print_summary('Global (top_p_expert_count_mean)', g_top_p, 'alpha1.5', 'alpha1.7')
            print_layerwise('Layerwise delta on top_p_expert_count_mean', l_top_p, layer_label)
        else:
            g_top_p, l_top_p = global_primary, layer_primary

        token_global, token_layer, hist15, hist17, top_ps = compare_top_p_token_counts(
            rec15, rec17, subset=args.subset)

        if top_ps:
            print(f'\nrouting top-p: {top_ps}')

        if token_global['n'] > 0:
            print_summary('Token-level top-p expert count', token_global, 'alpha1.5', 'alpha1.7')
            print_layerwise('Token-level layerwise delta on top-p expert count', token_layer, layer_label)
            print_histogram('Token-level histogram alpha1.5', hist15)
            print_histogram('Token-level histogram alpha1.7', hist17)
            interpretations.append(
                'token-level top-p expert count 的 delta(1.5-1.7) > 0 表示 alpha=1.5 对同一 token 平均保留了更多候选专家。')
        else:
            print('\n[Token-level top-p expert count]')
            print('No per-token top-p expert-count data found in routing_eval.')
            print('Re-run inference with the updated HuggingFaceDynamicMoE routing_eval to populate by_layer.*.top_p_expert_count_tokens.')

        top_p_summary = {
            'global_top_p_expert_count_mean': g_top_p,
            'layerwise_top_p_expert_count_mean': l_top_p,
            'token_level_top_p_expert_count': token_global,
            'token_level_layerwise_top_p_expert_count': token_layer,
            'token_level_hist_alpha15': hist15,
            'token_level_hist_alpha17': hist17,
            'routing_top_p_values': top_ps,
        }

    print('\nInterpretation:')
    for line in interpretations:
        print('-', line)

    if args.out:
        payload = {
            'alpha15': args.alpha15,
            'alpha17': args.alpha17,
            'subset': args.subset,
            'primary_metric': args.metric,
            'primary_global': global_primary,
            'primary_layerwise': layer_primary,
            'expert_count': expert_count_summary,
            'top_p_expert_count': top_p_summary,
            'interpretation': interpretations,
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'\nSaved summary to: {args.out}')


if __name__ == '__main__':
    main()
