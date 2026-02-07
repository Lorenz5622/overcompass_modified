#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPL分布诊断脚本
用于分析模型在PPL评测中的表现，对比正确答案和错误答案的PPL分布

Usage:
    python tools/diagnose_ppl_distribution.py \
        --result_file outputs/xxx/predictions/model_name/piqa.json \
        --output_dir outputs/ppl_diagnosis/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_predictions(file_path):
    """加载预测结果JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_ppl_distribution(predictions):
    """分析PPL分布"""
    stats = {
        'correct_ppls': [],
        'incorrect_ppls': [],
        'all_ppls': [],
        'ppl_gaps': [],  # 正确答案PPL - 最小错误答案PPL
        'correct_is_min': 0,  # 正确答案PPL最小的样本数
        'total_samples': 0,
        'per_sample_analysis': []
    }
    
    for idx, sample in predictions.items():
        if not idx.isdigit():
            continue
            
        stats['total_samples'] += 1
        
        # 获取gold答案
        if 'gold' not in sample:
            continue
        gold = sample['gold']
        pred = sample.get('prediction', None)
        
        # 收集所有label的PPL
        label_ppls = {}
        for key, value in sample.items():
            if key.startswith('label: '):
                label = key.replace('label: ', '')
                ppl = value.get('PPL', None)
                if ppl is not None:
                    label_ppls[label] = ppl
        
        if not label_ppls:
            continue
        
        # 分析此样本
        sample_analysis = {
            'idx': idx,
            'gold': gold,
            'prediction': pred,
            'label_ppls': label_ppls,
            'correct': pred == gold if pred is not None else False
        }
        
        # 获取正确答案和错误答案的PPL
        gold_str = str(gold)
        if gold_str in label_ppls:
            correct_ppl = label_ppls[gold_str]
            stats['correct_ppls'].append(correct_ppl)
            sample_analysis['gold_ppl'] = correct_ppl
            
            # 获取所有错误答案的PPL
            incorrect_ppls = [ppl for label, ppl in label_ppls.items() if label != gold_str]
            stats['incorrect_ppls'].extend(incorrect_ppls)
            
            # 计算PPL gap
            if incorrect_ppls:
                min_incorrect_ppl = min(incorrect_ppls)
                ppl_gap = correct_ppl - min_incorrect_ppl
                stats['ppl_gaps'].append(ppl_gap)
                sample_analysis['ppl_gap'] = ppl_gap
                sample_analysis['min_incorrect_ppl'] = min_incorrect_ppl
                
                # 检查正确答案是否PPL最小
                min_ppl = min(label_ppls.values())
                if correct_ppl == min_ppl:
                    stats['correct_is_min'] += 1
                    sample_analysis['gold_is_min_ppl'] = True
                else:
                    sample_analysis['gold_is_min_ppl'] = False
        
        stats['all_ppls'].extend(label_ppls.values())
        stats['per_sample_analysis'].append(sample_analysis)
    
    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("=" * 80)
    print("PPL分布诊断报告")
    print("=" * 80)
    
    print(f"\n总样本数: {stats['total_samples']}")
    print(f"正确答案PPL最小的样本数: {stats['correct_is_min']}")
    print(f"准确率: {stats['correct_is_min'] / stats['total_samples'] * 100:.2f}%")
    
    if stats['correct_ppls']:
        print(f"\n【正确答案PPL统计】")
        print(f"  平均值: {np.mean(stats['correct_ppls']):.4f}")
        print(f"  中位数: {np.median(stats['correct_ppls']):.4f}")
        print(f"  标准差: {np.std(stats['correct_ppls']):.4f}")
        print(f"  最小值: {np.min(stats['correct_ppls']):.4f}")
        print(f"  最大值: {np.max(stats['correct_ppls']):.4f}")
    
    if stats['incorrect_ppls']:
        print(f"\n【错误答案PPL统计】")
        print(f"  平均值: {np.mean(stats['incorrect_ppls']):.4f}")
        print(f"  中位数: {np.median(stats['incorrect_ppls']):.4f}")
        print(f"  标准差: {np.std(stats['incorrect_ppls']):.4f}")
        print(f"  最小值: {np.min(stats['incorrect_ppls']):.4f}")
        print(f"  最大值: {np.max(stats['incorrect_ppls']):.4f}")
    
    if stats['ppl_gaps']:
        print(f"\n【PPL差距统计 (正确答案PPL - 最小错误答案PPL)】")
        print(f"  平均值: {np.mean(stats['ppl_gaps']):.4f}")
        print(f"  中位数: {np.median(stats['ppl_gaps']):.4f}")
        print(f"  标准差: {np.std(stats['ppl_gaps']):.4f}")
        print(f"  正值样本数 (正确答案PPL更高): {sum(1 for x in stats['ppl_gaps'] if x > 0)}")
        print(f"  负值样本数 (正确答案PPL更低): {sum(1 for x in stats['ppl_gaps'] if x < 0)}")
    
    print("\n" + "=" * 80)


def plot_distributions(stats, output_dir):
    """绘制分布图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. PPL分布对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 正确答案vs错误答案PPL分布
    if stats['correct_ppls'] and stats['incorrect_ppls']:
        axes[0, 0].hist([stats['correct_ppls'], stats['incorrect_ppls']], 
                       bins=50, label=['Correct', 'Incorrect'], alpha=0.7)
        axes[0, 0].set_xlabel('PPL')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('PPL Distribution: Correct vs Incorrect')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # PPL差距分布
    if stats['ppl_gaps']:
        axes[0, 1].hist(stats['ppl_gaps'], bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Gap')
        axes[0, 1].set_xlabel('PPL Gap (Correct - Min Incorrect)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('PPL Gap Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot对比
    if stats['correct_ppls'] and stats['incorrect_ppls']:
        axes[1, 0].boxplot([stats['correct_ppls'], stats['incorrect_ppls']], 
                          labels=['Correct', 'Incorrect'])
        axes[1, 0].set_ylabel('PPL')
        axes[1, 0].set_title('PPL Box Plot Comparison')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 累积分布
    if stats['correct_ppls'] and stats['incorrect_ppls']:
        sorted_correct = np.sort(stats['correct_ppls'])
        sorted_incorrect = np.sort(stats['incorrect_ppls'])
        axes[1, 1].plot(sorted_correct, np.linspace(0, 1, len(sorted_correct)), 
                       label='Correct', linewidth=2)
        axes[1, 1].plot(sorted_incorrect, np.linspace(0, 1, len(sorted_incorrect)), 
                       label='Incorrect', linewidth=2)
        axes[1, 1].set_xlabel('PPL')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppl_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"分布图已保存到: {os.path.join(output_dir, 'ppl_distribution.png')}")
    plt.close()


def save_bad_cases(stats, output_dir, top_n=20):
    """保存错误案例"""
    bad_cases = [s for s in stats['per_sample_analysis'] if not s.get('gold_is_min_ppl', False)]
    
    # 按PPL gap排序（gap越大，说明正确答案PPL比错误答案高越多，越值得分析）
    bad_cases_with_gap = [s for s in bad_cases if 'ppl_gap' in s]
    bad_cases_sorted = sorted(bad_cases_with_gap, key=lambda x: x['ppl_gap'], reverse=True)
    
    output_file = os.path.join(output_dir, 'bad_cases.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"错误案例分析 (Top {top_n})\n")
        f.write("=" * 80 + "\n\n")
        
        for i, case in enumerate(bad_cases_sorted[:top_n], 1):
            f.write(f"案例 {i} (Sample {case['idx']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"金标准答案: {case['gold']}\n")
            f.write(f"模型预测: {case['prediction']}\n")
            f.write(f"预测是否正确: {case['correct']}\n")
            f.write(f"\n所有选项的PPL:\n")
            for label, ppl in sorted(case['label_ppls'].items(), key=lambda x: x[1]):
                marker = " ← [GOLD]" if str(label) == str(case['gold']) else ""
                marker += " ← [PRED]" if str(label) == str(case['prediction']) else ""
                f.write(f"  选项 {label}: {ppl:.4f}{marker}\n")
            
            if 'ppl_gap' in case:
                f.write(f"\nPPL差距 (正确-最小错误): {case['ppl_gap']:.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"错误案例已保存到: {output_file}")


def compare_two_models(file1, file2, output_dir):
    """对比两个模型的PPL分布"""
    print("\n对比两个模型...")
    
    data1 = load_predictions(file1)
    data2 = load_predictions(file2)
    
    stats1 = analyze_ppl_distribution(data1)
    stats2 = analyze_ppl_distribution(data2)
    
    print("\n【模型1统计】")
    print_statistics(stats1)
    
    print("\n【模型2统计】")
    print_statistics(stats2)
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 正确答案PPL对比
    if stats1['correct_ppls'] and stats2['correct_ppls']:
        axes[0, 0].hist([stats1['correct_ppls'], stats2['correct_ppls']], 
                       bins=50, label=['Model 1', 'Model 2'], alpha=0.7)
        axes[0, 0].set_xlabel('PPL')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Correct Answer PPL: Model 1 vs Model 2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 错误答案PPL对比
    if stats1['incorrect_ppls'] and stats2['incorrect_ppls']:
        axes[0, 1].hist([stats1['incorrect_ppls'], stats2['incorrect_ppls']], 
                       bins=50, label=['Model 1', 'Model 2'], alpha=0.7)
        axes[0, 1].set_xlabel('PPL')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Incorrect Answer PPL: Model 1 vs Model 2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # PPL gap对比
    if stats1['ppl_gaps'] and stats2['ppl_gaps']:
        axes[1, 0].hist([stats1['ppl_gaps'], stats2['ppl_gaps']], 
                       bins=50, label=['Model 1', 'Model 2'], alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('PPL Gap')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('PPL Gap: Model 1 vs Model 2')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 准确率对比
    acc1 = stats1['correct_is_min'] / stats1['total_samples'] * 100
    acc2 = stats2['correct_is_min'] / stats2['total_samples'] * 100
    axes[1, 1].bar(['Model 1', 'Model 2'], [acc1, acc2], alpha=0.7, color=['blue', 'orange'])
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, (label, acc) in enumerate(zip(['Model 1', 'Model 2'], [acc1, acc2])):
        axes[1, 1].text(i, acc + 1, f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PPL分布诊断工具')
    parser.add_argument('--result_file', type=str, required=True,
                       help='预测结果JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='./ppl_diagnosis',
                       help='输出目录')
    parser.add_argument('--compare_file', type=str, default=None,
                       help='用于对比的第二个模型的预测结果文件（可选）')
    parser.add_argument('--top_n', type=int, default=20,
                       help='保存的错误案例数量')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.result_file):
        print(f"错误: 文件不存在 - {args.result_file}")
        return
    
    print(f"加载预测结果: {args.result_file}")
    predictions = load_predictions(args.result_file)
    
    print("分析PPL分布...")
    stats = analyze_ppl_distribution(predictions)
    
    print_statistics(stats)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n绘制分布图...")
    plot_distributions(stats, args.output_dir)
    
    print("\n保存错误案例...")
    save_bad_cases(stats, args.output_dir, args.top_n)
    
    # 保存详细统计信息
    stats_file = os.path.join(args.output_dir, 'statistics.json')
    stats_to_save = {
        'total_samples': stats['total_samples'],
        'correct_is_min': stats['correct_is_min'],
        'accuracy': stats['correct_is_min'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0,
        'correct_ppl': {
            'mean': float(np.mean(stats['correct_ppls'])) if stats['correct_ppls'] else None,
            'median': float(np.median(stats['correct_ppls'])) if stats['correct_ppls'] else None,
            'std': float(np.std(stats['correct_ppls'])) if stats['correct_ppls'] else None,
        },
        'incorrect_ppl': {
            'mean': float(np.mean(stats['incorrect_ppls'])) if stats['incorrect_ppls'] else None,
            'median': float(np.median(stats['incorrect_ppls'])) if stats['incorrect_ppls'] else None,
            'std': float(np.std(stats['incorrect_ppls'])) if stats['incorrect_ppls'] else None,
        },
        'ppl_gap': {
            'mean': float(np.mean(stats['ppl_gaps'])) if stats['ppl_gaps'] else None,
            'median': float(np.median(stats['ppl_gaps'])) if stats['ppl_gaps'] else None,
            'std': float(np.std(stats['ppl_gaps'])) if stats['ppl_gaps'] else None,
        }
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
    print(f"统计信息已保存到: {stats_file}")
    
    # 如果提供了对比文件，进行对比分析
    if args.compare_file:
        if os.path.exists(args.compare_file):
            compare_two_models(args.result_file, args.compare_file, args.output_dir)
        else:
            print(f"警告: 对比文件不存在 - {args.compare_file}")
    
    print("\n✓ 诊断完成!")


if __name__ == '__main__':
    main()
