#!/usr/bin/env python3
"""Compute dataset-level average PPL from OpenCompass result JSON files.

Supported inputs:
1. results/.../*.json
2. predictions/.../*.json

Examples:
    python tools/calc_avg_ppl.py \
        --result-file outputs/default/20260327_141906/results/dynamic_moe_dm/piqa.json

    python tools/calc_avg_ppl.py \
        --result-file outputs/default/20260327_141906/results/dynamic_moe_dm/piqa.json \
        --save-json outputs/default/20260327_141906/results/dynamic_moe_dm/piqa_ppl_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Iterator, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute average PPL statistics for an OpenCompass result file.')
    parser.add_argument(
        '--result-file',
        required=True,
        help='Path to a results/*.json or predictions/*.json file.')
    parser.add_argument(
        '--save-json',
        default=None,
        help='Optional path to save the summary as JSON.')
    return parser.parse_args()


def load_json(file_path: Path) -> dict:
    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def iter_samples(data: dict) -> Iterator[Tuple[str, dict]]:
    container = data.get('details', data)
    if not isinstance(container, dict):
        return
    for sample_id, sample in container.items():
        if sample_id == 'type' or not isinstance(sample, dict):
            continue
        yield str(sample_id), sample


def normalize_gold(sample: dict, options_container: dict) -> str | None:
    gold = options_container.get('gold', sample.get('references', sample.get('gold')))
    if gold is None:
        return None
    return str(gold)


def extract_option_ppls(sample: dict) -> Tuple[str | None, Dict[str, float]]:
    options_container = sample.get('origin_prediction')
    if not isinstance(options_container, dict):
        options_container = sample

    gold = normalize_gold(sample, options_container)
    option_ppls: Dict[str, float] = {}

    for key, value in options_container.items():
        if not isinstance(value, dict):
            continue

        option_id = None
        if key.isdigit():
            option_id = key
        elif key.startswith('label: '):
            option_id = key.split('label: ', 1)[1]

        if option_id is None:
            continue

        ppl = value.get('PPL')
        if ppl is None:
            continue
        option_ppls[str(option_id)] = float(ppl)

    return gold, option_ppls


def summarize(values: Iterable[float]) -> dict:
    values = list(values)
    if not values:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
        }
    return {
        'count': len(values),
        'mean': mean(values),
        'median': median(values),
        'min': min(values),
        'max': max(values),
    }


def analyze_file(file_path: Path) -> dict:
    data = load_json(file_path)

    correct_option_ppls = []
    all_option_ppls = []
    per_sample_mean_ppls = []

    total_samples = 0
    valid_samples = 0
    samples_with_gold_ppl = 0
    missing_gold_samples = []

    for sample_id, sample in iter_samples(data):
        total_samples += 1
        gold, option_ppls = extract_option_ppls(sample)
        if not option_ppls:
            continue

        valid_samples += 1
        all_option_ppls.extend(option_ppls.values())
        per_sample_mean_ppls.append(mean(option_ppls.values()))

        if gold is not None and gold in option_ppls:
            correct_option_ppls.append(option_ppls[gold])
            samples_with_gold_ppl += 1
        else:
            missing_gold_samples.append(sample_id)

    return {
        'result_file': str(file_path),
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'samples_with_gold_ppl': samples_with_gold_ppl,
        'missing_gold_ppl_samples': missing_gold_samples,
        'correct_option_ppl': summarize(correct_option_ppls),
        'all_option_ppl': summarize(all_option_ppls),
        'per_sample_mean_ppl': summarize(per_sample_mean_ppls),
    }


def format_stat_line(title: str, stat: dict) -> str:
    if stat['count'] == 0:
        return f'{title}: count=0'
    return (
        f'{title}: count={stat["count"]}, '
        f'mean={stat["mean"]:.6f}, '
        f'median={stat["median"]:.6f}, '
        f'min={stat["min"]:.6f}, '
        f'max={stat["max"]:.6f}'
    )


def print_summary(summary: dict) -> None:
    print(f'Result file: {summary["result_file"]}')
    print(f'Total samples: {summary["total_samples"]}')
    print(f'Valid samples with PPL: {summary["valid_samples"]}')
    print(f'Samples with gold-option PPL: {summary["samples_with_gold_ppl"]}')
    print(f'Total sample-option PPL entries: {summary["all_option_ppl"]["count"]}')
    print(format_stat_line('Correct-option PPL', summary['correct_option_ppl']))
    print(format_stat_line('All-option PPL', summary['all_option_ppl']))
    print(format_stat_line('Per-sample mean PPL', summary['per_sample_mean_ppl']))

    missing = summary['missing_gold_ppl_samples']
    if missing:
        preview = ', '.join(missing[:10])
        suffix = '' if len(missing) <= 10 else ' ...'
        print(f'Samples missing gold-option PPL: {len(missing)} ({preview}{suffix})')


def maybe_save_json(summary: dict, output_path: str | None) -> None:
    if output_path is None:
        return

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Saved summary JSON to: {out_path}')


def main() -> None:
    args = parse_args()
    result_file = Path(args.result_file)
    summary = analyze_file(result_file)
    print_summary(summary)
    maybe_save_json(summary, args.save_json)


if __name__ == '__main__':
    main()
