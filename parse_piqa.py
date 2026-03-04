import math
from statistics import mean
import json
import csv

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


JSON_PATH = 'outputs/default/20260228_123141/results/qwen1.5-moe-a2.7b-hf/piqa.json'
TSV_PATH = 'moe_routing_token_stats_piqa.tsv'


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return mean(values)


def pearson_corr(xs, ys):
    """简单 Pearson 相关系数实现，返回 None 表示无法计算。"""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    # 统一转为 float，避免 Decimal 混入导致类型错误
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    varx = sum((x - mx) ** 2 for x in xs)
    vary = sum((y - my) ** 2 for y in ys)
    if varx <= 0 or vary <= 0:
        return None
    return cov / math.sqrt(varx * vary)


def analyze():
    ppls = []

    top4_means = []         # 所有层、所有 token 的 top-4 概率质量均值
    top4_last_means = []    # 最后若干层的 top-4 概率质量均值

    delta_means = []        # 所有层、所有 token 的 (p4 - p5) 均值
    delta_last_means = []   # 最后若干层的 (p4 - p5) 均值

    entropy_means = []      # 所有层、所有 token 的熵均值
    entropy_last_means = [] # 最后若干层的熵均值

    sample_count = 0
    record_count = 0

    sample_limit = 1000  # 为了运行时间，先只抽样前若干条，可按需调大或去掉

    last_k_layers = 2  # 只看最后多少层的指标

    print('Start analyzing MoE routing vs PPL...')

    with open(JSON_PATH, 'r') as f:
        # 只遍历 details 下的键值对，避免一次性加载整个文件
        if HAS_IJSON:
            items = ijson.kvitems(f, 'details')
        else:
            data = json.load(f)
            items = data.get('details', {}).items()

        for sample_id, sample in items:
            if sample_id == 'type':
                continue
            sample_count += 1

            if sample_count > sample_limit:
                break

            origin_pred = sample.get('origin_prediction', {})
            # 只取正确选项（references 为字符串 "0" 或 "1"）
            correct_opt = str(sample.get('references', ''))
            if correct_opt not in ('0', '1'):
                continue
            opt_ids = [correct_opt]
            # 遍历选项
            for opt_id in opt_ids:
                option = origin_pred.get(opt_id)
                if not option:
                    continue

                ppl = option.get('PPL')
                if ppl is None:
                    continue

                probs = option.get('probs')
                if not probs:
                    continue

                # 解析层索引，找到最后几层
                layer_indices = []  # [(idx, layer_name), ...]
                for lname in probs.keys():
                    if lname.startswith('layer'):
                        try:
                            idx = int(lname[len('layer'):])
                            layer_indices.append((idx, lname))
                        except ValueError:
                            continue

                if not layer_indices:
                    continue

                layer_indices.sort(key=lambda x: x[0])
                last_layers = {idx for idx, _ in layer_indices[-last_k_layers:]}
                idx_to_name = {idx: lname for idx, lname in layer_indices}

                # token 级统计
                top4_all = []
                delta_all = []
                entropy_all = []

                top4_last = []
                delta_last = []
                entropy_last = []

                for idx, lname in layer_indices:
                    layer_probs = probs.get(lname)
                    if not layer_probs:
                        continue

                    for token_probs in layer_probs:
                        if not token_probs:
                            continue
                        # 排序得到 top-k
                        sorted_probs = sorted(token_probs, reverse=True)
                        # top-4 概率质量
                        s_top4 = sum(sorted_probs[:4]) if len(sorted_probs) >= 4 else sum(sorted_probs)
                        # 边缘差距 p4 - p5
                        if len(sorted_probs) >= 5:
                            delta_45 = sorted_probs[3] - sorted_probs[4]
                        else:
                            delta_45 = 0.0
                        # 熵
                        ent = 0.0
                        for p in token_probs:
                            p = float(p)
                            if p > 0.0:
                                ent -= p * math.log(p)

                        top4_all.append(s_top4)
                        delta_all.append(delta_45)
                        entropy_all.append(ent)

                        if idx in last_layers:
                            top4_last.append(s_top4)
                            delta_last.append(delta_45)
                            entropy_last.append(ent)

                if not top4_all:
                    continue

                ppls.append(ppl)
                top4_means.append(mean(top4_all))
                delta_means.append(mean(delta_all))
                entropy_means.append(mean(entropy_all))

                if top4_last:
                    top4_last_means.append(mean(top4_last))
                    delta_last_means.append(mean(delta_last))
                    entropy_last_means.append(mean(entropy_last))
                else:
                    top4_last_means.append(None)
                    delta_last_means.append(None)
                    entropy_last_means.append(None)

                record_count += 1

    # 将 None 过滤掉再计算相关性
    def corr_with_ppl(metric_list, name):
        xs = []
        ys = []
        for ppl, m in zip(ppls, metric_list):
            if m is None:
                continue
            xs.append(m)
            ys.append(ppl)
        r = pearson_corr(xs, ys)
        print(f'{name}: n={len(xs)}, pearson_corr={r}')

    print(f'Total samples (details entries): {sample_count}')
    print(f'Total records (sample-option with valid PPL & probs): {record_count}')

    corr_with_ppl(top4_means, 'top4_mean (all layers) vs PPL')
    corr_with_ppl(delta_means, 'delta_4_5_mean (all layers) vs PPL')
    corr_with_ppl(entropy_means, 'entropy_mean (all layers) vs PPL')

    corr_with_ppl(top4_last_means, f'top4_mean (last {last_k_layers} layers) vs PPL')
    corr_with_ppl(delta_last_means, f'delta_4_5_mean (last {last_k_layers} layers) vs PPL')
    corr_with_ppl(entropy_last_means, f'entropy_mean (last {last_k_layers} layers) vs PPL')

def _to_float(obj):
    """递归将 Decimal/数值类型统一转成 Python float，确保 JSON 可序列化。"""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [_to_float(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_float(v) for k, v in obj.items()}
    return obj


def extract_extreme_ppls(k=20):
    """提取 PPL 最小和最大的各 k 条数据，保存到 JSON。

    输出结构：
    {
        "min": [
            {"sample_id": ..., "option_id": ..., "ppl": ..., "probs": ...},
            ... 共 k 条
        ],
        "max": [ ... 共 k 条 ]
    }
    """
    print(f'Start extracting top-{k} min/max PPL records...')

    records = []  # (ppl, sample_id, opt_id, probs)

    with open(JSON_PATH, 'r') as f:
        if HAS_IJSON:
            items = ijson.kvitems(f, 'details')
        else:
            data = json.load(f)
            items = data.get('details', {}).items()

        for sample_id, sample in items:
            if sample_id == 'type':
                continue
            origin_pred = sample.get('origin_prediction', {})
            # 只取正确选项
            correct_opt = str(sample.get('references', ''))
            if correct_opt not in ('0', '1'):
                continue
            for opt_id in [correct_opt]:
                option = origin_pred.get(opt_id)
                if not option:
                    continue
                ppl = option.get('PPL')
                probs = option.get('probs')
                if ppl is None or probs is None:
                    continue
                records.append((float(ppl), sample_id, opt_id, probs))

    if not records:
        print('No valid records found.')
        return

    # 按 PPL 排序
    records.sort(key=lambda x: x[0])
    min_k = records[:k]
    max_k = records[-k:]

    result = {
        'min': [
            {
                'ppl': ppl,
                'sample_id': sample_id,
                'option_id': opt_id,
                'probs': _to_float(probs),
            }
            for ppl, sample_id, opt_id, probs in min_k
        ],
        'max': [
            {
                'ppl': ppl,
                'sample_id': sample_id,
                'option_id': opt_id,
                'probs': _to_float(probs),
            }
            for ppl, sample_id, opt_id, probs in max_k
        ],
    }

    out_path = 'outputs/default/20260228_123141/results/qwen1.5-moe-a2.7b-hf/piqa_extremes.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)

    print(f'Saved extremes to {out_path}')


def analyze_tsv_tokenwise(tsv_path: str = TSV_PATH):
    """基于评测生成的 TSV 文件，计算 token 级路由指标与 PPL(NLL) 的 Pearson 相关性。

    输出两部分：
    1. 汇总指标（all layers / last 2 layers）与 token_nll 的相关性
    2. 每一层的 top4 / delta / entropy 与 token_nll 的相关性
    """
    # 汇总指标
    summary_metrics = {
        'top4_all': ([], []),
        'delta_all': ([], []),
        'entropy_all': ([], []),
        'top4_last2': ([], []),
        'delta_last2': ([], []),
        'entropy_last2': ([], []),
    }
    # 每层指标：动态构建，key 形如 'top4_layer0'、'delta_layer3' 等
    layer_metrics = {}  # name -> (xs, ys)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        # 建立列名到索引的映射，避免列顺序变化影响
        col_idx = {name: i for i, name in enumerate(header)}
        nll_idx = col_idx.get('token_nll')
        if nll_idx is None:
            raise ValueError('TSV 中未找到 token_nll 列')

        # 从表头中找出所有 per-layer 列：top4_layerN / delta_layerN / entropy_layerN
        for col_name in header:
            if col_name.startswith(('top4_layer', 'delta_layer', 'entropy_layer')):
                layer_metrics[col_name] = ([], [])

        def add_point(metrics_dict, metric_name, x_str, nll):
            if metric_name not in metrics_dict:
                return
            if x_str is None or x_str == '':
                return
            x = float(x_str)
            xs, ys = metrics_dict[metric_name]
            xs.append(x)
            ys.append(nll)

        for row in reader:
            try:
                nll = float(row[nll_idx])
            except (ValueError, IndexError):
                continue

            # 汇总指标
            for name in summary_metrics:
                idx = col_idx.get(name)
                add_point(summary_metrics, name, row[idx] if idx is not None else None, nll)

            # 每层指标
            for name in layer_metrics:
                idx = col_idx.get(name)
                add_point(layer_metrics, name, row[idx] if idx is not None else None, nll)

    print(f'Read TSV from {tsv_path}')

    # --- 汇总指标相关性 ---
    print('\n[Summary metrics vs token_nll]')
    for name, (xs, ys) in summary_metrics.items():
        r = pearson_corr(xs, ys)
        print(f'  {name}: n={len(xs)}, pearson_corr={r}')

    # --- 每层指标相关性（按层号排序输出，每层 top4 / delta / entropy 并排）---
    if layer_metrics:
        # 提取所有出现的层编号
        layer_ids = set()
        for col_name in layer_metrics:
            # col_name 形如 'top4_layer5' -> 取 'layer5'
            parts = col_name.split('_', 1)  # ['top4', 'layer5']
            if len(parts) == 2:
                layer_ids.add(parts[1])  # 'layer5'
        # 按数字排序
        def _layer_sort_key(lk):
            try:
                return int(lk.replace('layer', ''))
            except ValueError:
                return -1
        sorted_layer_ids = sorted(layer_ids, key=_layer_sort_key)

        print('\n[Per-layer metrics vs token_nll]')
        for lk in sorted_layer_ids:
            results = []
            for prefix in ('top4', 'delta', 'entropy'):
                col_name = f'{prefix}_{lk}'
                if col_name in layer_metrics:
                    xs, ys = layer_metrics[col_name]
                    r = pearson_corr(xs, ys)
                    results.append(f'{prefix}={r:.4f}(n={len(xs)})')
            print(f'  {lk}: ' + '  '.join(results))


if __name__ == '__main__':
    # JSON 分析：基于原始 piqa.json 的样本级相关性
    # analyze()

    # TSV 分析：基于 hf_qwen_moe_custom 写出的 token 级路由统计
    analyze_tsv_tokenwise()

