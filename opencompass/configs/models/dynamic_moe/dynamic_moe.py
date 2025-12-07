# configs/models/dynamic_moe.py
from opencompass.models import DynamicMoE,HuggingFaceDynamicMoE, HuggingFaceCausalLM
import torch
models = [
    dict(
        type=HuggingFaceDynamicMoE,
        # type=HuggingFaceCausalLM,
        path='/data/cyx/models/ADAK_MoE',
        tokenizer_path='/data/cyx/models/ADAK_MoE',
        max_out_len=1024,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16
        ),
        batch_size=32,
        run_cfg=dict(num_gpus=1),
    )
]
