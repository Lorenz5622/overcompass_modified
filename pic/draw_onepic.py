#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 17   # 全局默认字号

def load_layer_avg_k(json_path: str) -> List[Tuple[int, float]]:
    """Return sorted list of (layer_idx, avg_k). Compatible with layer_idx or layer_id."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_layer" not in data or not isinstance(data["per_layer"], list):
        raise ValueError(f"Unexpected JSON structure (missing per_layer list): {json_path}")

    rows = []
    for item in data["per_layer"]:
        # 兼容 layer_idx / layer_id
        layer_key = "layer_idx" if "layer_idx" in item else "layer_id" if "layer_id" in item else None
        if layer_key is None or "avg_k" not in item:
            raise ValueError(f"per_layer item missing layer_idx/layer_id or avg_k in {json_path}: {item}")
        rows.append((int(item[layer_key]), float(item["avg_k"])))

    rows.sort(key=lambda x: x[0])
    return rows


def plot_grouped_bar(
    model_to_layeravg: Dict[str, List[Tuple[int, float]]],
    title: str,
    out_path: str,
    out_path_eps: str | None = None,
) -> None:
    """
    Draw grouped (clustered) bar chart: each layer is a group, each model is a bar in that group.
    X/Y labels keep same as your code.
    """
    # 取所有 layer 的并集并排序（一般三份都一样，但这样更稳）
    all_layers = sorted({layer for rows in model_to_layeravg.values() for layer, _ in rows})
    n_layers = len(all_layers)
    layer_to_pos = {layer: i for i, layer in enumerate(all_layers)}
    x = np.arange(n_layers)

    # 组内柱子更窄、间距更开：width 小一点，group_span 大一点
    n_models = len(model_to_layeravg)
    width = 0.2  # 柱子变窄（你可继续调小：0.15）
    group_span = 0.60  # 组内总展开宽度（越大越“拉开”）
    offsets = np.linspace(-group_span / 2, group_span / 2, n_models)

    # 颜色（你可以按需改）
    colors = ["#7189b1", "#e36281", "#a39f9e"]  # 蓝、橙、绿

    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)

    # 画每个模型的一组柱
    for i, (model_name, rows) in enumerate(model_to_layeravg.items()):
        # 对齐到 all_layers 上（缺的层用 nan）
        y = np.full(n_layers, np.nan, dtype=float)
        for layer, avgk in rows:
            y[layer_to_pos[layer]] = avgk

        ax.bar(
            x + offsets[i],
            y,
            width=width,
            color=colors[i % len(colors)],
            label=model_name,
        )

    # 坐标轴保持和你原来一致
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Experts per token")
    # ax.set_title(title)

    # x 轴用 layer id 显示
    ax.set_xticks(x)
    ax.set_xticklabels(all_layers)  # 如果你想稀疏显示，比如每隔2层显示一次也可以再改

    ax.legend(loc="upper right")
    plt.tight_layout()

    # 保存（PNG + 可选 EPS）
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if out_path_eps is not None:
        plt.savefig(out_path_eps, format="eps", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    # 你如果想从命令行传参，也可以重新打开这几个 argument
    # parser.add_argument("--out_dir", default="./figs", help="Directory to save figures")
    args = parser.parse_args()

    out_dir = "./pic"
    os.makedirs(out_dir, exist_ok=True)

    model_paths: Dict[str, str] = {
        "w/o expert total loss": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_no_expert_restrain.json",
        "w/o hardening loss": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_no_hardening.json",
        "complete model": "/home/cyx/opencompass/expert_usage_stats_dynamic_moe_complete.json",
    }

    # 读数据
    model_to_layeravg = {name: load_layer_avg_k(path) for name, path in model_paths.items()}

    # 输出
    out_png = os.path.join(out_dir, "layer_avg_k_all_models.png")
    out_eps = os.path.join(out_dir, "layer_avg_k_all_models.eps")
    title = "Predict MoE: Per-Layer avg-k (3 models)"

    plot_grouped_bar(model_to_layeravg, title, out_png, out_path_eps=out_eps)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_eps}")
    print("Done.")


if __name__ == "__main__":
    main()
