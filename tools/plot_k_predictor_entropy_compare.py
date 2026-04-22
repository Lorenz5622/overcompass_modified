#!/usr/bin/env python3
"""Plot per-layer k-predictor entropy for two Predict MoE runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_COMPLETE = (
    '/home/cyx/opencompass/expert_usage_stats_predict_moe_complete_v1.json'
)
DEFAULT_BASELINE = (
    '/home/cyx/opencompass/expert_usage_stats_predict_moe_no_harden_v1.json'
)
DEFAULT_OUTPUT = (
    '/home/cyx/opencompass/k_predictor_entropy_compare_complete_vs_no_harden_v1.png'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot per-layer k-predictor entropy curves for two runs.'
    )
    parser.add_argument(
        '--complete',
        default=DEFAULT_COMPLETE,
        help='Path to the "complete" JSON file.',
    )
    parser.add_argument(
        '--baseline',
        default=DEFAULT_BASELINE,
        help='Path to the baseline JSON file.',
    )
    parser.add_argument(
        '--baseline-label',
        default='w/o harden',
        help='Legend label for the baseline curve.',
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Where to save the figure.',
    )
    parser.add_argument(
        '--field',
        choices=['k_predictor_entropy_mean_nats', 'k_predictor_entropy_mean_bits'],
        default='k_predictor_entropy_mean_nats',
        help='Entropy field to plot.',
    )
    parser.add_argument(
        '--title',
        default='',
        help='Figure title.',
    )
    return parser.parse_args()


def load_series(path: str, field: str) -> tuple[list[int], list[float]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'per_layer' not in data:
        raise ValueError(f'Missing "per_layer" in {path}')

    layers = []
    values = []
    for item in data['per_layer']:
        if 'layer_idx' not in item:
            raise ValueError(f'Missing "layer_idx" in {path}')
        if field not in item:
            raise ValueError(f'Missing "{field}" in {path}')
        layers.append(int(item['layer_idx']))
        values.append(float(item[field]))
    return layers, values


def main() -> None:
    args = parse_args()
    complete_layers, complete_values = load_series(args.complete, args.field)
    baseline_layers, baseline_values = load_series(args.baseline, args.field)

    if complete_layers != baseline_layers:
        raise ValueError('Layer indices do not match between the two input files.')

    ylabel = 'Entropy (nats)'
    if args.field.endswith('_bits'):
        ylabel = 'Entropy (bits)'

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5.5))
    plt.plot(
        complete_layers,
        complete_values,
        marker='o',
        linewidth=2,
        label='complete',
    )
    plt.plot(
        baseline_layers,
        baseline_values,
        marker='s',
        linewidth=2,
        label=args.baseline_label,
    )
    plt.xlabel('Layer')
    plt.ylabel(ylabel)
    plt.title(args.title)
    plt.xticks(complete_layers)
    plt.ylim(top=3)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()

    print(output_path)


if __name__ == '__main__':
    main()
