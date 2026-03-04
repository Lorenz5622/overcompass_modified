#!/usr/bin/env python3
"""Run minimal MoE routing experiments (exp2 + exp3) from routing TSV.

No third-party dependencies are required.

Experiment 2:
  - Difficulty-bucketed (token_nll quantile-like bins) within-bin Spearman.
Experiment 3:
  - Layer-wise Spearman scan for top4/entropy/delta vs token_nll.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def to_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def average_ranks(values: Sequence[float]) -> List[float]:
    """Return average ranks (1-based) with tie handling."""
    n = len(values)
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        # ranks from i+1 ... j+1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float('nan')
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    denx = 0.0
    deny = 0.0
    for a, b in zip(x, y):
        dx = a - mx
        dy = b - my
        num += dx * dy
        denx += dx * dx
        deny += dy * dy
    den = math.sqrt(denx * deny)
    if den == 0.0:
        return float('nan')
    return num / den


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float('nan')
    rx = average_ranks(x)
    ry = average_ranks(y)
    return pearson_corr(rx, ry)


def assign_bins_by_rank(values: Sequence[float], num_bins: int) -> List[int]:
    """Assign near-equal-frequency bins via sorted rank index."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    bins = [0] * n
    for rank_pos, original_idx in enumerate(order):
        b = min(num_bins - 1, int(rank_pos * num_bins / max(1, n)))
        bins[original_idx] = b
    return bins


def detect_layer_columns(headers: Sequence[str], prefix: str) -> List[Tuple[int, str]]:
    pat = re.compile(rf'^{re.escape(prefix)}_layer(\d+)$')
    out: List[Tuple[int, str]] = []
    for h in headers:
        m = pat.match(h)
        if m:
            out.append((int(m.group(1)), h))
    out.sort(key=lambda x: x[0])
    return out


def write_csv(path: str, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_experiment2(
    rows: List[Dict[str, str]],
    nll_col: str,
    metric_cols: Sequence[str],
    num_bins: int,
) -> List[Dict[str, object]]:
    data = []
    for r in rows:
        nll = to_float(r.get(nll_col, ''))
        if nll is None or nll <= 0.0:
            continue
        data.append((r, nll))

    if not data:
        return []

    nll_vals = [x[1] for x in data]
    bin_ids = assign_bins_by_rank(nll_vals, num_bins)

    results: List[Dict[str, object]] = []
    for b in range(num_bins):
        idxs = [i for i, bid in enumerate(bin_ids) if bid == b]
        if not idxs:
            continue
        nll_sub = [nll_vals[i] for i in idxs]
        nll_min = min(nll_sub)
        nll_max = max(nll_sub)
        nll_mean = sum(nll_sub) / len(nll_sub)

        for m in metric_cols:
            x = []
            y = []
            for i in idxs:
                v = to_float(data[i][0].get(m, ''))
                if v is None:
                    continue
                x.append(v)
                y.append(data[i][1])
            rho = spearman_corr(x, y)
            results.append(
                {
                    'bin_id': b,
                    'metric': m,
                    'n': len(x),
                    'nll_min': f'{nll_min:.6f}',
                    'nll_max': f'{nll_max:.6f}',
                    'nll_mean': f'{nll_mean:.6f}',
                    'rho_spearman': '' if math.isnan(rho) else f'{rho:.6f}',
                    'pvalue': '',
                }
            )

    return results


def run_experiment3(rows: List[Dict[str, str]], headers: Sequence[str], nll_col: str) -> List[Dict[str, object]]:
    filtered = []
    nll_vals = []
    for r in rows:
        nll = to_float(r.get(nll_col, ''))
        if nll is None or nll <= 0.0:
            continue
        filtered.append(r)
        nll_vals.append(nll)

    if not filtered:
        return []

    results: List[Dict[str, object]] = []
    for fam in ('top4', 'entropy', 'delta'):
        for layer_idx, col in detect_layer_columns(headers, fam):
            x = []
            y = []
            for r, nll in zip(filtered, nll_vals):
                v = to_float(r.get(col, ''))
                if v is None:
                    continue
                x.append(v)
                y.append(nll)
            rho = spearman_corr(x, y)
            results.append(
                {
                    'metric_family': fam,
                    'layer_idx': layer_idx,
                    'column': col,
                    'n': len(x),
                    'rho_spearman': '' if math.isnan(rho) else f'{rho:.6f}',
                    'pvalue': '',
                }
            )

    results.sort(key=lambda r: (r['metric_family'], int(r['layer_idx'])))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Run exp2 + exp3 on MoE routing TSV.')
    parser.add_argument('--tsv', required=True, help='Path to routing TSV file')
    parser.add_argument('--output-dir', default='moe_analysis_out', help='Output directory')
    parser.add_argument('--nll-col', default='token_nll', help='NLL column name')
    parser.add_argument('--num-bins', type=int, default=10, help='Number of bins for experiment2')
    parser.add_argument(
        '--exp2-metrics',
        nargs='*',
        default=['top4_all', 'entropy_all', 'delta_all', 'top4_last2', 'entropy_last2', 'delta_last2'],
        help='Metric columns for experiment2',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        headers = reader.fieldnames or []

    if args.nll_col not in headers:
        raise ValueError(f'Missing required NLL column: {args.nll_col}')

    cleaned = 0
    for r in rows:
        nll = to_float(r.get(args.nll_col, ''))
        if nll is not None and nll > 0.0:
            cleaned += 1
    print(f'[info] rows_total={len(rows)} rows_cleaned_nll_gt_0={cleaned}')

    exp2_rows = run_experiment2(rows, args.nll_col, args.exp2_metrics, args.num_bins)
    exp2_path = os.path.join(args.output_dir, 'experiment2_binwise_spearman.csv')
    write_csv(
        exp2_path,
        exp2_rows,
        ['bin_id', 'metric', 'n', 'nll_min', 'nll_max', 'nll_mean', 'rho_spearman', 'pvalue'],
    )
    print(f'[done] experiment2 -> {exp2_path} (rows={len(exp2_rows)})')

    exp3_rows = run_experiment3(rows, headers, args.nll_col)
    exp3_path = os.path.join(args.output_dir, 'experiment3_layerwise_spearman.csv')
    write_csv(
        exp3_path,
        exp3_rows,
        ['metric_family', 'layer_idx', 'column', 'n', 'rho_spearman', 'pvalue'],
    )
    print(f'[done] experiment3 -> {exp3_path} (rows={len(exp3_rows)})')


if __name__ == '__main__':
    main()
