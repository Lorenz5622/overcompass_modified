# configs/models/dynamic_moe/dynamic_moe_base.py
# 微调前的基础模型配置
from opencompass.models import HuggingFaceDynamicMoE
import torch

models = [
    dict(
        abbr='dynamic_moe_base',  # 基础模型简称
        type=HuggingFaceDynamicMoE,
        path='/data/cyx/models/Dynamic_MoE',  # 微调前的模型路径
        tokenizer_path='/data/cyx/models/Dynamic_MoE',
        moe_package_name='Dynamic_MoE',  # 指定MoE包名
        max_out_len=64,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
        ),
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
