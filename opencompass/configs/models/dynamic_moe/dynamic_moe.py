# # configs/models/dynamic_moe.py
from opencompass.models import DynamicMoE,HuggingFaceDynamicMoE, HuggingFaceCausalLM
import torch
models = [
    dict(
        type=HuggingFaceDynamicMoE,
        # type=HuggingFaceCausalLM,
        path='/data/cyx/models/Dynamic_MoE',
        tokenizer_path='/data/cyx/models/Dynamic_MoE',
        max_out_len=512,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
            enable_expert_stats=False,
        ),
        batch_size=2,
        run_cfg=dict(num_gpus=1, num_procs=8),
    )
]


# configs/models/dynamic_moe_env.py
# import os
# from opencompass.models import HuggingFaceDynamicMoE
# import torch

# model_path = os.environ["OC_MODEL_PATH"]
# model_name = os.environ.get("OC_MODEL_NAME", "dynamic_moe_env")

# models = [
#     dict(
#         type=HuggingFaceDynamicMoE,
#         abbr=model_name,
#         path=model_path,
#         tokenizer_path=model_path,
#         max_out_len=192,
#         model_kwargs=dict(
#             trust_remote_code=True,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             enable_expert_stats=False,  # 启用专家统计（仅在 Predict MoE 中生效）
#             expert_stats_path='./expert_usage_stats.json',  # 统计结果保存路径
#         ),
#         batch_size=192,
#         run_cfg=dict(num_gpus=1, num_procs=16),
#     )
# ]