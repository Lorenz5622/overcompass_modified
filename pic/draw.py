#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw per-layer average expert usage (avg_k) bar charts for 3 models.

Input JSON format (as in your files):
{
  "model_type": ...,
  "per_layer": [
     {"layer_id": 0, "avg_k": 6.21, ...},
     ...
  ],
  "global_avg_k": ...,
  "total_tokens": ...
}

Usage:
  python plot_layer_avg_k.py \
    --no_expert_restrain /path/to/expert_usage_stats_dynamic_moe_no_expert_restrain.json \
    --no_hardening /path/to/expert_usage_stats_dynamic_moe_no_hardening.json \
    --complete /path/to/expert_usage_stats_dynamic_moe_complete.json \
    --out_dir ./figs
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_layer_avg_k(json_path: str) -> List[Tuple[int, float]]:
    """Return sorted list of (layer_idx, avg_k)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_layer" not in data or not isinstance(data["per_layer"], list):
        raise ValueError(f"Unexpected JSON structure (missing per_layer list): {json_path}")

    rows = []
    for item in data["per_layer"]:
        if "layer_idx" not in item or "avg_k" not in item:
            raise ValueError(f"per_layer item missing layer_idx/avg_k in {json_path}: {item}")
        rows.append((int(item["layer_idx"]), float(item["avg_k"])))

    rows.sort(key=lambda x: x[0])
    return rows


def load_global_avg_k(json_path: str) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("global_avg_k", float("nan")))


def plot_bar(layer_avg: List[Tuple[int, float]], title: str, out_path: str) -> None:
    layer_ids = [x for x, _ in layer_avg]
    avg_ks = [y for _, y in layer_avg]
    x_step = 0.5   # 越小越紧凑（但太小会重叠）
    x = [i * x_step for i in range(len(layer_ids))]
    plt.figure(figsize=(9, 5))
    plt.bar(layer_ids, avg_ks, width=0.8)  # don't hardcode colors
    plt.xlabel("Layer")
    plt.ylabel("Average Experts per token")
    plt.title(title)
    # plt.xticks(layer_ids, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--no_expert_restrain", required=True, help="JSON path for missing experts-total-loss")
    # parser.add_argument("--no_hardening", required=True, help="JSON path for missing hardening-loss")
    # parser.add_argument("--complete", required=True, help="JSON path for complete/original model")
    # parser.add_argument("--out_dir", default="./figs", help="Directory to save PNG figures")
    args = parser.parse_args()

    os.makedirs("./pic", exist_ok=True)

    model_paths: Dict[str, str] = {
        "no_expert_restrain": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_no_expert_restrain.json",
        "no_hardening": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_no_hardening.json",
        "complete": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_complete.json",
    }
    table_name = ["Predict MoE (w/o expert total loss): Per-Layer abg-k", "Predict MoE (w/o hardening loss): Per-Layer abg-k", "Predict MoE (complete model): Per-Layer abg-k"]
    for idx, (model_name, path) in enumerate(model_paths.items()):
        layer_avg = load_layer_avg_k(path)
        # global_avg = load_global_avg_k(path)
        out_path = os.path.join("./pic", f"{model_name}_layer_avg_k.png")
        title = table_name[idx]
        plot_bar(layer_avg, title, out_path)
        print(f"[Saved] {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
