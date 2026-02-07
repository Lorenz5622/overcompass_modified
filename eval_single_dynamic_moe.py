from mmengine.config import read_base
import sys
import torch

# 从命令行参数获取模型路径和名称
model_path = None
model_name = None

for i, arg in enumerate(sys.argv):
    if arg == '--model-path' and i + 1 < len(sys.argv):
        model_path = sys.argv[i + 1]
    if arg == '--model-name' and i + 1 < len(sys.argv):
        model_name = sys.argv[i + 1]

# 如果没有从命令行获取，使用默认值
if model_path is None:
    model_path = '/data/cyx/models/Predict_MoE_v29'
if model_name is None:
    model_name = 'Predict_MoE_v29'

with read_base():
    # 导入数据集配置
    from opencompass.configs.datasets.piqa.piqa_ppl import piqa_datasets

# 动态配置模型
from opencompass.models import HuggingFaceDynamicMoE

models = [
    dict(
        type=HuggingFaceDynamicMoE,
        abbr=model_name,
        path=model_path,
        tokenizer_path=model_path,
        max_out_len=64,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
        ),
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]

# 使用 PIQA 数据集
datasets = piqa_datasets
