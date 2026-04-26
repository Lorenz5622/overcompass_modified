# configs/models/dynamic_moe/dynamic_moe_yuan.py
# Yuan 路由版 Qwen_MoE 执行代码配置
import os

from opencompass.models import HuggingFaceYuanMoE
import torch

default_model_path = '/data/cyx/models/your_dynamic_moe_yuan_ckpt'
model_path = os.environ.get('OC_MODEL_PATH', default_model_path)
tokenizer_path = os.environ.get('OC_TOKENIZER_PATH', model_path)
model_name = os.environ.get('OC_MODEL_NAME', 'dynamic_moe_yuan')

models = [
    dict(
        type=HuggingFaceYuanMoE,
        abbr=model_name,
        path=model_path,
        tokenizer_path=tokenizer_path,
        moe_package_name='Qwen_MoE',
        moe_modeling_module='modeling_moe_yuan',
        moe_config_module='configuration_moe_yuan',
        max_out_len=128,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
            enable_expert_stats=True,
            enable_kpredictor_entropy_stats=False,
            enable_theoretical_flops_stats=True,
            enable_routing_eval=False,
            enable_token_routing_tsv=False,
            routing_eval_with_layers=False,
            routing_eval_prob_eps=1e-8,
            routing_eval_top_p=0.7,
            routing_eval_store_token_counts=False,
        ),
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=16),
    )
]
