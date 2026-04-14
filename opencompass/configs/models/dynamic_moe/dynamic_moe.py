# configs/models/dynamic_moe/dynamic_moe.py
# 原版 Predict_MoE 执行代码配置
from opencompass.models import HuggingFaceDynamicMoE
import torch

models = [
    dict(
        type=HuggingFaceDynamicMoE,
        abbr='dynamic_moe_predict',
        path='/data/cyx/models/out_piqa_lora_ori',
        tokenizer_path='/data/cyx/models/out_piqa_lora_ori',
        moe_package_name='Predict_MoE',
        moe_modeling_module='modeling_moe_ori',
        moe_config_module='configuration_moe',
        max_out_len=192,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
            enable_expert_stats=False,
        ),
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=8),
    )
]
