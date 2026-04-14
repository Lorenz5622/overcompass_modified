#!/usr/bin/env python3
"""Inspect entmax effect by comparing input logits and output probabilities.

Examples:
  python tools/inspect_entmax_effect.py \
    --alphas 1.5 1.7 --matrix '[[2.0,1.0,0.5,-1.0],[3.0,2.5,0.1,-2.0]]'

  python tools/inspect_entmax_effect.py \
    --input-file /tmp/logits.npy --alphas 1.5 1.7 --save-dir /tmp/entmax_report
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def fallback_entmax_bisect(
    inputs: torch.Tensor,
    alpha: float = 1.5,
    dim: int = -1,
    n_iter: int = 32,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f'alpha must be in (1, 2], got {alpha}')

    alpha_m1 = alpha - 1.0
    inv_alpha_m1 = 1.0 / alpha_m1

    x = inputs - inputs.max(dim=dim, keepdim=True).values
    tau_lo = x.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) * 0.5
        p_mid = torch.clamp(alpha_m1 * (x - tau_mid), min=0.0) ** inv_alpha_m1
        sum_p = p_mid.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sum_p > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sum_p <= 1.0, tau_mid, tau_hi)

    tau_star = (tau_lo + tau_hi) * 0.5
    probs = torch.clamp(alpha_m1 * (x - tau_star), min=0.0) ** inv_alpha_m1
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs


def resolve_entmax_fn(module_name: str, extra_sys_path: Optional[str]) -> tuple[Callable, str]:
    if extra_sys_path:
        p = os.path.abspath(extra_sys_path)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, 'entmax_bisect', None)
        if callable(fn):
            return fn, f'{module_name}.entmax_bisect'
    except Exception:
        pass
    return fallback_entmax_bisect, 'fallback_entmax_bisect'


def load_matrix(args) -> torch.Tensor:
    if args.matrix is not None:
        arr = ast.literal_eval(args.matrix)
        t = torch.tensor(arr, dtype=torch.float32)
        return t

    if args.input_file is not None:
        path = Path(args.input_file)
        if not path.exists():
            raise FileNotFoundError(f'input file not found: {path}')
        if path.suffix.lower() == '.npy':
            arr = np.load(path)
            return torch.tensor(arr, dtype=torch.float32)
        if path.suffix.lower() == '.npz':
            z = np.load(path)
            if not z.files:
                raise ValueError('npz has no arrays')
            arr = z[z.files[0]]
            return torch.tensor(arr, dtype=torch.float32)
        if path.suffix.lower() in {'.json', '.js'}:
            arr = json.loads(path.read_text(encoding='utf-8'))
            return torch.tensor(arr, dtype=torch.float32)
        if path.suffix.lower() in {'.pt', '.pth'}:
            obj = torch.load(path, map_location='cpu')
            if isinstance(obj, torch.Tensor):
                return obj.float()
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, torch.Tensor):
                        return v.float()
            raise ValueError('pt/pth must contain tensor or dict with tensor')
        raise ValueError(f'unsupported input file suffix: {path.suffix}')

    # default demo matrix
    return torch.tensor([
        [3.2, 2.7, 1.9, 0.3, -0.2, -1.0, -2.0, -3.5],
        [1.5, 1.4, 1.3, 1.2, 1.1, 0.9, 0.6, 0.2],
        [4.0, 2.2, 0.1, -0.5, -1.2, -2.5, -3.1, -4.0],
    ], dtype=torch.float32)


def rowwise_stats(probs: torch.Tensor, eps: float = 1e-8) -> List[Dict]:
    if probs.dim() != 2:
        raise ValueError('rowwise_stats expects 2D tensor [rows, cols]')
    out = []
    for i in range(probs.size(0)):
        row = probs[i]
        nz = int((row > eps).sum().item())
        safe = row.clamp_min(eps)
        ent = float((-(safe * safe.log()).sum()).item())
        top1v, top1i = torch.max(row, dim=0)
        out.append({
            'row': i,
            'nz_count': nz,
            'entropy': ent,
            'top1_idx': int(top1i.item()),
            'top1_prob': float(top1v.item()),
        })
    return out


def print_stats_table(title: str, stats: List[Dict]) -> None:
    print(f'\n[{title}]')
    print('row\tnz_count\tentropy\ttop1_idx\ttop1_prob')
    for s in stats:
        print(f"{s['row']}\t{s['nz_count']}\t{s['entropy']:.6f}\t{s['top1_idx']}\t{s['top1_prob']:.6f}")


def save_csv(path: Path, tensor2d: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor2d.detach().cpu().numpy()
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow([f'{float(x):.8f}' for x in row])


def main():
    parser = argparse.ArgumentParser(description='Inspect entmax input/output matrices.')
    parser.add_argument('--matrix', type=str, default='[[2.0,1.0,0.2,-1.0],[3.0,2.9,0.1,-2.0]]',
                        help='Literal matrix string, e.g. "[[2,1,0], [3,2,1]]"')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Load matrix from .npy/.npz/.json/.pt/.pth')
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.5, 1.7],
                        help='Entmax alphas to compare, each in (1, 2].')
    parser.add_argument('--dim', type=int, default=-1,
                        help='Dimension for normalization (default: -1).')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Threshold for nz_count and entropy clamp.')
    parser.add_argument('--entmax-module', type=str,
                        default='qwen_moe.modeling.modeling_moe_dm',
                        help='Module name that provides entmax_bisect.')
    parser.add_argument('--module-root', type=str, default='/home/cyx',
                        help='Extra sys.path inserted before importing module.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='If set, save input/output csv and summary json.')
    parser.add_argument('--print-matrices', action='store_true',
                        help='Print full matrices.')
    args = parser.parse_args()

    logits = load_matrix(args)
    if logits.dim() != 2:
        raise ValueError(f'expected 2D matrix, got shape={tuple(logits.shape)}')

    entmax_fn, src = resolve_entmax_fn(args.entmax_module, args.module_root)

    softmax = F.softmax(logits, dim=args.dim)
    softmax_stats = rowwise_stats(softmax, eps=args.eps)

    print('Entmax effect inspection')
    print(f'logits shape: {tuple(logits.shape)}')
    print(f'entmax source: {src}')
    print(f'alphas: {args.alphas}')

    if args.print_matrices:
        print('\n[Input logits]')
        print(logits)
        print('\n[Softmax output]')
        print(softmax)

    print_stats_table('Softmax stats', softmax_stats)

    summary = {
        'shape': list(logits.shape),
        'entmax_source': src,
        'alphas': args.alphas,
        'softmax': {
            'row_stats': softmax_stats,
            'mean_nz_count': float(mean([x['nz_count'] for x in softmax_stats])),
            'mean_entropy': float(mean([x['entropy'] for x in softmax_stats])),
        },
        'entmax': {},
    }

    outputs = {}
    for alpha in args.alphas:
        probs = entmax_fn(logits.float(), alpha=float(alpha), dim=args.dim)
        stats = rowwise_stats(probs, eps=args.eps)
        outputs[alpha] = probs

        if args.print_matrices:
            print(f'\n[Entmax output alpha={alpha}]')
            print(probs)

        print_stats_table(f'Entmax stats alpha={alpha}', stats)
        summary['entmax'][str(alpha)] = {
            'row_stats': stats,
            'mean_nz_count': float(mean([x['nz_count'] for x in stats])),
            'mean_entropy': float(mean([x['entropy'] for x in stats])),
        }

    if len(args.alphas) >= 2:
        a0 = args.alphas[0]
        a1 = args.alphas[1]
        s0 = summary['entmax'][str(a0)]
        s1 = summary['entmax'][str(a1)]
        print('\n[Delta summary]')
        print(f'mean_nz_count(alpha={a0}) - mean_nz_count(alpha={a1}) = '
              f"{s0['mean_nz_count'] - s1['mean_nz_count']:.6f}")
        print(f'mean_entropy(alpha={a0}) - mean_entropy(alpha={a1}) = '
              f"{s0['mean_entropy'] - s1['mean_entropy']:.6f}")

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_csv(out_dir / 'input_logits.csv', logits)
        save_csv(out_dir / 'output_softmax.csv', softmax)
        for alpha, probs in outputs.items():
            name = f'output_entmax_alpha_{str(alpha).replace(".", "_")}.csv'
            save_csv(out_dir / name, probs)
        (out_dir / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f'\nSaved report to: {out_dir}')


if __name__ == '__main__':
    main()
