from opencompass.models import HuggingFaceQwenMoeCustom
import torch

models = [
    dict(
        type=HuggingFaceQwenMoeCustom,
        abbr='qwen1.5-moe-a2.7b-hf',
        path='/data/cyx/models/Qwen1.5-MoE-A2.7B',
        max_out_len=512,
        batch_size=2,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
        ),
        run_cfg=dict(num_gpus=1),
    )
]
