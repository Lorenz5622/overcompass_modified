#!/usr/bin/env python
"""
混合策略优化工具：基于PPL阈值混合Predict_MoE和baseline的预测结果
用法: python tools/apply_hybrid_strategy.py --v2_path <v2预测文件> --baseline_path <baseline预测文件> --threshold 0.05
"""

import json
import argparse
import numpy as np
from pathlib import Path


def apply_hybrid_strategy(v2_path, baseline_path, output_path, threshold=0.05):
    """应用混合策略优化预测结果
    
    Args:
        v2_path: Predict_MoE_v2的预测文件路径
        baseline_path: baseline模型的预测文件路径  
        output_path: 输出文件路径
        threshold: PPL差异阈值，小于此值时使用baseline的预测
    """
    # 读取数据
    with open(v2_path, 'r') as f:
        v2_data = json.load(f)
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    # 创建优化后的版本
    optimized_data = {}
    stats = {
        'total': len(v2_data),
        'changed': 0,
        'improved': 0,
        'degraded': 0,
        'unchanged': 0,
    }
    
    for key, item in v2_data.items():
        # 复制原始数据
        optimized_data[key] = item.copy()
        
        # 计算PPL差异
        ppl0 = item['label: 0']['PPL']
        ppl1 = item['label: 1']['PPL']
        ppl_diff = abs(ppl0 - ppl1)
        
        original_pred = item['prediction']
        gold = item['gold']
        
        # 如果PPL差异小于阈值，使用baseline的预测
        if ppl_diff < threshold:
            base_pred = baseline_data[key]['prediction']
            optimized_data[key]['prediction'] = base_pred
            
            if base_pred != original_pred:
                stats['changed'] += 1
                if base_pred == gold and original_pred != gold:
                    stats['improved'] += 1
                elif base_pred != gold and original_pred == gold:
                    stats['degraded'] += 1
        else:
            if original_pred == gold:
                stats['unchanged'] += 1
    
    # 计算准确率
    correct = sum(1 for k, v in optimized_data.items() if v['prediction'] == v['gold'])
    accuracy = correct / len(optimized_data) * 100
    
    # 保存结果
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(optimized_data, f, indent=4)
    
    return accuracy, stats


def main():
    parser = argparse.ArgumentParser(description='Apply hybrid strategy to optimize predictions')
    parser.add_argument('--v2_path', required=True, help='Path to Predict_MoE_v2 predictions')
    parser.add_argument('--baseline_path', required=True, help='Path to baseline predictions')
    parser.add_argument('--output_path', required=True, help='Path to save optimized predictions')
    parser.add_argument('--threshold', type=float, default=0.05, help='PPL difference threshold (default: 0.05)')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('混合策略优化工具')
    print('=' * 60)
    print(f'V2模型预测文件: {args.v2_path}')
    print(f'Baseline预测文件: {args.baseline_path}')
    print(f'PPL阈值: {args.threshold}')
    print(f'输出文件: {args.output_path}')
    print('=' * 60)
    
    accuracy, stats = apply_hybrid_strategy(
        args.v2_path,
        args.baseline_path,
        args.output_path,
        args.threshold
    )
    
    print('\n优化结果:')
    print(f'  总样本数: {stats["total"]}')
    print(f'  改变预测: {stats["changed"]}个')
    print(f'  - 修正错误: {stats["improved"]}个')
    print(f'  - 引入错误: {stats["degraded"]}个')
    print(f'  - 净提升: {stats["improved"] - stats["degraded"]}个')
    print(f'  优化后准确率: {accuracy:.2f}%')
    print(f'\n已保存到: {args.output_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
