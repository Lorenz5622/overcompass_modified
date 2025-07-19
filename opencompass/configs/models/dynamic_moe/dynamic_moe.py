# configs/models/dynamic_moe.py
from opencompass.models import DynamicMoE,HuggingFaceDynamicMoE

models = [
    dict(
        type=HuggingFaceDynamicMoE,
        path='/home/cyx/models/Dynamic_MoE',
        tokenizer_path='/home/cyx/models/Dynamic_MoE',
        max_out_len=1024,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype='auto'
        ),
        batch_size=32,
        run_cfg=dict(num_gpus=1),
    )
]
