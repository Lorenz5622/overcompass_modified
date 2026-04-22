#!/usr/bin/env python3
"""Plot per-layer expert-count usage distributions for two ablation settings.

The input JSON files are expected to follow the schema:
{
  "model_type": "...",
  "per_layer": [
    {
      "layer_idx": 0,
      "distribution": {"1": 123, "2": 456, ...},
      "total_tokens": 789,
      "avg_k": 2.34
    },
    ...
  ]
}

This tool focuses on the token-level distribution over selected expert counts
(`k=1..8`) rather than expert identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMPLETE = '/home/cyx/opencompass/expert_usage_stats_predict_moe_complete.json'
DEFAULT_BASELINE = '/home/cyx/opencompass/expert_usage_stats_predict_moe_no_harden.json'
DEFAULT_OUTPUT = '/home/cyx/opencompass/expert_usage_compare_complete_vs_no_hardening.png'


def entropy_from_probs(probs: Sequence[float]) -> float:
    arr = np.asarray(probs, dtype=float)
    valid = arr > 0.0
    if not np.any(valid):
        return 0.0
    return float(-(arr[valid] * np.log2(arr[valid])).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare per-layer expert-count usage distributions between two runs.'
    )
    parser.add_argument(
        '--complete',
        default=DEFAULT_COMPLETE,
        help='Path to the "complete" JSON file.',
    )
    parser.add_argument(
        '--baseline',
        default=DEFAULT_BASELINE,
        help='Path to the baseline JSON file, usually "no_hardening".',
    )
    parser.add_argument(
        '--baseline-label',
        default='no_hardening',
        help='Legend/display label for the baseline run.',
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Where to save the rendered figure.',
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=8,
        help='Maximum k value to display on the heatmaps.',
    )
    parser.add_argument(
        '--heatmap-only',
        action='store_true',
        help='Render only the two heatmaps and skip the line charts.',
    )
    parser.add_argument(
        '--split-heatmaps',
        action='store_true',
        help='When used with --heatmap-only, save each heatmap as a separate figure.',
    )
    parser.add_argument(
        '--no-title',
        action='store_true',
        help='Do not render subplot titles or figure titles.',
    )
    return parser.parse_args()


def load_usage(path: str, max_k: int) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    layers = data['per_layer']
    counts: List[List[int]] = []
    probs: List[List[float]] = []
    dominant_share: List[float] = []
    avg_k: List[float] = []
    p2_minus_p3: List[float] = []
    entropy: List[float] = []
    mode_k: List[int] = []
    layer_ids: List[int] = []

    for layer in layers:
        total = layer['total_tokens']
        dist = {int(k): int(v) for k, v in layer['distribution'].items()}
        count_row = [dist.get(k, 0) for k in range(1, max_k + 1)]
        row = [dist.get(k, 0) / total for k in range(1, max_k + 1)]
        counts.append(count_row)
        probs.append(row)
        dominant_share.append(max(row) if row else 0.0)
        avg_k.append(float(layer['avg_k']))
        p2_minus_p3.append(row[1] - row[2] if max_k >= 3 else 0.0)
        entropy.append(entropy_from_probs(row))
        mode_k.append(int(np.argmax(row)) + 1 if row else 0)
        layer_ids.append(int(layer['layer_idx']))

    counts_array = np.array(counts, dtype=float)
    probs_array = np.array(probs, dtype=float)
    entropy_array = np.array(entropy, dtype=float)
    total_counts = counts_array.sum(axis=0)
    total_prob = total_counts / total_counts.sum() if total_counts.sum() > 0 else np.zeros(max_k, dtype=float)

    return {
        'model_type': data.get('model_type', 'unknown'),
        'layers': layer_ids,
        'counts': counts_array,
        'probs': probs_array,
        'dominant_share': np.array(dominant_share, dtype=float),
        'avg_k': np.array(avg_k, dtype=float),
        'p2_minus_p3': np.array(p2_minus_p3, dtype=float),
        'entropy': entropy_array,
        'mean_entropy': float(entropy_array.mean()) if len(entropy_array) > 0 else 0.0,
        'sum_entropy': float(entropy_array.sum()),
        'global_prob': total_prob,
        'global_entropy': entropy_from_probs(total_prob),
        'mode_k': np.array(mode_k, dtype=int),
    }


def summarize(name: str, usage: Dict[str, object]) -> str:
    dominant_share = usage['dominant_share']
    avg_k = usage['avg_k']
    p2_minus_p3 = usage['p2_minus_p3']
    mean_entropy = usage['mean_entropy']
    sum_entropy = usage['sum_entropy']
    global_entropy = usage['global_entropy']
    mode_k = usage['mode_k']
    return (
        f'{name}: '
        f'mean dominant-share={dominant_share.mean():.3f}, '
        f'mean avg_k={avg_k.mean():.3f}, '
        f'mean (p2-p3)={p2_minus_p3.mean():.3f}, '
        f'mean layer-entropy={mean_entropy:.3f} bits, '
        f'sum layer-entropy={sum_entropy:.3f} bits, '
        f'global entropy={global_entropy:.3f} bits, '
        f'layers with mode=2: {(mode_k == 2).sum()}'
    )


def draw_heatmap(ax, probs: np.ndarray, title: str, layers: Sequence[int], max_k: int, show_title: bool):
    image = ax.imshow(
        probs,
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap='YlOrRd',
        vmin=0.0,
        vmax=float(probs.max()),
    )
    if show_title:
        ax.set_title(title)
    ax.set_xlabel('Selected expert count k')
    ax.set_ylabel('Layer')
    ax.set_xticks(np.arange(max_k))
    ax.set_xticklabels(np.arange(1, max_k + 1))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    return image


def save_single_heatmap(
    probs: np.ndarray,
    layers: Sequence[int],
    max_k: int,
    output_path: Path,
    title: str,
    show_title: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    image = ax.imshow(
        probs,
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap='YlOrRd',
        vmin=0.0,
        vmax=float(probs.max()),
    )
    if show_title:
        ax.set_title(title)
    ax.set_xlabel('Selected expert count k')
    ax.set_ylabel('Layer')
    ax.set_xticks(np.arange(max_k))
    ax.set_xticklabels(np.arange(1, max_k + 1))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    cbar = fig.colorbar(image, ax=ax, shrink=0.95)
    cbar.set_label('Probability mass at each k')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def split_output_paths(output: Path) -> tuple[Path, Path]:
    stem = output.stem
    suffix = output.suffix or '.png'
    return (
        output.with_name(f'{stem}_complete{suffix}'),
        output.with_name(f'{stem}_baseline{suffix}'),
    )


def main() -> None:
    args = parse_args()

    complete = load_usage(args.complete, args.max_k)
    baseline = load_usage(args.baseline, args.max_k)

    if complete['probs'].shape != baseline['probs'].shape:
        raise ValueError(
            'The two inputs must have the same number of layers and the same max-k setting.'
        )

    layers = complete['layers']
    x = np.array(layers, dtype=int)
    show_title = not args.no_title
    output_path = Path(args.output)

    if args.heatmap_only and args.split_heatmaps:
        complete_output, baseline_output = split_output_paths(output_path)
        save_single_heatmap(
            complete['probs'],
            layers,
            args.max_k,
            complete_output,
            'complete: per-layer k distribution',
            show_title,
        )
        save_single_heatmap(
            baseline['probs'],
            layers,
            args.max_k,
            baseline_output,
            f'{args.baseline_label}: per-layer k distribution',
            show_title,
        )
        print(summarize('complete', complete))
        print(summarize(args.baseline_label, baseline))
        print(f'saved figure to: {complete_output}')
        print(f'saved figure to: {baseline_output}')
        return

    if args.heatmap_only:
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax_heat_complete = fig.add_subplot(gs[0, 0])
        ax_heat_baseline = fig.add_subplot(gs[0, 1])
        ax_sharp = None
        ax_bias = None
    else:
        fig = plt.figure(figsize=(15, 11), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0])
        ax_heat_complete = fig.add_subplot(gs[0, 0])
        ax_heat_baseline = fig.add_subplot(gs[0, 1])
        ax_sharp = fig.add_subplot(gs[1, 0])
        ax_bias = fig.add_subplot(gs[1, 1])

    max_prob = max(float(complete['probs'].max()), float(baseline['probs'].max()))
    image_complete = ax_heat_complete.imshow(
        complete['probs'],
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap='YlOrRd',
        vmin=0.0,
        vmax=max_prob,
    )
    if show_title:
        ax_heat_complete.set_title('complete: per-layer k distribution')
    ax_heat_complete.set_xlabel('Selected expert count k')
    ax_heat_complete.set_ylabel('Layer')
    ax_heat_complete.set_xticks(np.arange(args.max_k))
    ax_heat_complete.set_xticklabels(np.arange(1, args.max_k + 1))
    ax_heat_complete.set_yticks(np.arange(len(layers)))
    ax_heat_complete.set_yticklabels(layers)

    ax_heat_baseline.imshow(
        baseline['probs'],
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap='YlOrRd',
        vmin=0.0,
        vmax=max_prob,
    )
    if show_title:
        ax_heat_baseline.set_title(f'{args.baseline_label}: per-layer k distribution')
    ax_heat_baseline.set_xlabel('Selected expert count k')
    ax_heat_baseline.set_ylabel('Layer')
    ax_heat_baseline.set_xticks(np.arange(args.max_k))
    ax_heat_baseline.set_xticklabels(np.arange(1, args.max_k + 1))
    ax_heat_baseline.set_yticks(np.arange(len(layers)))
    ax_heat_baseline.set_yticklabels(layers)

    cbar = fig.colorbar(image_complete, ax=[ax_heat_complete, ax_heat_baseline], shrink=0.94)
    cbar.set_label('Probability mass at each k')

    if not args.heatmap_only:
        ax_sharp.plot(x, complete['dominant_share'], marker='o', linewidth=2.0, label='complete')
        ax_sharp.plot(
            x,
            baseline['dominant_share'],
            marker='o',
            linewidth=2.0,
            label=args.baseline_label,
        )
        ax_sharp.set_title('Dominant-share by layer')
        ax_sharp.set_xlabel('Layer')
        ax_sharp.set_ylabel('Max probability over k')
        ax_sharp.set_ylim(0.35, 1.05)
        ax_sharp.grid(alpha=0.25, linestyle='--')
        ax_sharp.legend(frameon=False)

        ax_bias.plot(x, complete['p2_minus_p3'], marker='o', linewidth=2.0, label='complete')
        ax_bias.plot(
            x,
            baseline['p2_minus_p3'],
            marker='o',
            linewidth=2.0,
            label=args.baseline_label,
        )
        ax_bias.axhline(0.0, color='black', linewidth=1.0, alpha=0.5)
        ax_bias.set_title('Bias toward k=2 vs k=3')
        ax_bias.set_xlabel('Layer')
        ax_bias.set_ylabel('P(k=2) - P(k=3)')
        ax_bias.set_ylim(-1.05, 1.05)
        ax_bias.grid(alpha=0.25, linestyle='--')
        ax_bias.legend(frameon=False)

        if show_title:
            fig.suptitle(
                'Dynamic MoE expert-count comparison: complete vs no_hardening\n'
                'Heatmaps show the full per-layer distribution; lines highlight concentration and k=2/k=3 preference.',
                fontsize=14,
            )
    else:
        if show_title:
            fig.suptitle(
                'Dynamic MoE expert-count comparison: complete vs no_hardening',
                fontsize=14,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')

    print(summarize('complete', complete))
    print(summarize(args.baseline_label, baseline))
    print(f'saved figure to: {output_path}')


if __name__ == '__main__':
    main()
