# configs/models/dynamic_moe/dynamic_moe_reinforce_layerk.py
import os

import torch
from opencompass.models import HuggingFaceDynamicMoE


default_model_path = '/data/cyx/models/your_reinforce_layerk_ckpt'
model_path = os.environ.get('OC_MODEL_PATH', default_model_path)
tokenizer_path = os.environ.get('OC_TOKENIZER_PATH', model_path)
model_name = os.environ.get('OC_MODEL_NAME', 'dynamic_moe_reinforce_layerk')

models = [
    dict(
        type=HuggingFaceDynamicMoE,
        abbr=model_name,
        path=model_path,
        tokenizer_path=tokenizer_path,
        moe_package_name='Predict_MoE',
        moe_modeling_module='modeling_moe_reinforce_layerk',
        moe_config_module='configuration_moe_reinforce_layerk',
        max_out_len=128,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
            enable_expert_stats=False,
            enable_routing_eval=False,
            routing_eval_with_layers=False,
            routing_eval_prob_eps=1e-8,
            routing_eval_top_p=0.7,
            routing_eval_store_token_counts=False,
        ),
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=16),
    )
]
