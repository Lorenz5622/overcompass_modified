#!/usr/bin/env python3
"""
专家调用统计功能使用示例

该脚本展示如何在评测过程中收集和保存专家调用统计信息。
仅在 Predict MoE 模型（使用 modeling_moe_infer）中生效。
"""

import os
import sys

# 示例1: 在评测脚本中调用统计功能
def example_with_opencompass():
    """
    在 OpenCompass 评测流程中使用统计功能
    """
    # 设置环境变量
    os.environ["OC_MODEL_PATH"] = "/data/cyx/models/Predict_MoE_k25/"
    os.environ["OC_MODEL_NAME"] = "predict_moe"
    
    # 正常运行评测
    # python run.py configs/eval_xxx.py --model configs/models/dynamic_moe/dynamic_moe.py
    
    # 评测结束后，模型实例会自动检测并保存统计信息（如果启用了）
    # 统计结果默认保存在: ./expert_usage_stats.json


# 示例2: 手动获取和保存统计信息
def example_manual_stats():
    """
    手动获取和保存统计信息
    """
    from opencompass.models import HuggingFaceDynamicMoE
    import torch
    
    # 初始化模型
    model = HuggingFaceDynamicMoE(
        path="/data/cyx/models/Predict_MoE_k25/",
        tokenizer_path="/data/cyx/models/Predict_MoE_k25/",
        max_out_len=128,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            enable_expert_stats=True,  # 启用统计
            expert_stats_path='./my_custom_stats.json',
        ),
        batch_size=16,
    )
    
    # 进行推理
    inputs = ["What is the capital of France?", "Explain quantum computing."]
    outputs = model.generate(inputs, max_out_len=50)
    
    # 获取统计信息（不保存到文件）
    stats = model.get_expert_usage_stats()
    if stats:
        print(f"Total tokens processed: {stats['total_tokens']}")
        print(f"Global average experts per token: {stats['global_avg_k']:.2f}")
        print(f"Number of layers: {len(stats['per_layer'])}")
        
        # 查看某一层的统计
        if len(stats['per_layer']) > 0:
            layer_0 = stats['per_layer'][0]
            print(f"\nLayer 0 statistics:")
            print(f"  Average k: {layer_0['avg_k']:.2f}")
            print(f"  Distribution: {layer_0['distribution']}")
    
    # 保存到文件
    model.save_expert_usage_stats('./expert_stats_custom.json')
    
    # 重置统计（如果需要重新计算）
    model.reset_expert_usage_stats()


# 示例3: 分析统计结果
def analyze_stats(stats_file='./expert_usage_stats.json'):
    """
    分析保存的统计文件
    """
    import json
    import matplotlib.pyplot as plt
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("=" * 60)
    print(f"Model Type: {stats['model_type']}")
    print(f"Global Average Experts per Token: {stats['global_avg_k']:.2f}")
    print(f"Total Tokens Processed: {stats['total_tokens']}")
    print("=" * 60)
    
    # 按层分析
    print("\nPer-layer statistics:")
    for layer_stat in stats['per_layer']:
        layer_idx = layer_stat['layer_idx']
        avg_k = layer_stat['avg_k']
        total_tokens = layer_stat['total_tokens']
        distribution = layer_stat['distribution']
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    Average k: {avg_k:.2f}")
        print(f"    Total tokens: {total_tokens}")
        print(f"    Distribution: {distribution}")
    
    # 可视化（可选）
    try:
        layer_indices = [s['layer_idx'] for s in stats['per_layer']]
        avg_ks = [s['avg_k'] for s in stats['per_layer']]
        
        plt.figure(figsize=(10, 5))
        plt.bar(layer_indices, avg_ks)
        plt.xlabel('Layer Index')
        plt.ylabel('Average Experts per Token')
        plt.title('Expert Usage Distribution Across Layers')
        plt.savefig('expert_usage_analysis.png')
        print("\n[INFO] Visualization saved to expert_usage_analysis.png")
    except ImportError:
        print("\n[WARN] matplotlib not installed, skipping visualization")


# 示例4: 在评测脚本结束时自动保存（通过 hook）
def add_stats_saving_hook():
    """
    在评测脚本中添加 hook，在评测结束时自动保存统计
    
    将以下代码添加到你的评测脚本（如 eval_single_dynamic_moe.py）的末尾：
    """
    code_snippet = '''
# 在评测脚本末尾添加：
from opencompass.registry import MODELS

# 获取所有已注册的模型实例
for model_name, model_instance in MODELS._module_dict.items():
    if hasattr(model_instance, 'save_expert_usage_stats'):
        model_instance.save_expert_usage_stats()
        print(f"[INFO] Saved expert stats for {model_name}")
'''
    print(code_snippet)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("统计功能已集成到 OpenCompass 评测框架中")
    print("=" * 60)
    print("\n关键点：")
    print("1. 仅在 Predict MoE 模型（modeling_moe_infer）中启用")
    print("2. Dynamic MoE（modeling_moe_ori）不受影响")
    print("3. 通过 model_kwargs 中的 enable_expert_stats=True 启用")
    print("4. 评测结束后可通过 save_expert_usage_stats() 保存结果")
    print("\n" + "=" * 60)
    
    # 如果提供了统计文件，进行分析
    if len(sys.argv) > 1:
        stats_file = sys.argv[1]
        if os.path.exists(stats_file):
            print(f"\nAnalyzing {stats_file}...\n")
            analyze_stats(stats_file)
        else:
            print(f"\n[ERROR] File not found: {stats_file}")
    else:
        print("\n使用方式:")
        print(f"  python {sys.argv[0]} <stats_file.json>  # 分析统计文件")
        print(f"  python {sys.argv[0]}                    # 查看使用示例")
