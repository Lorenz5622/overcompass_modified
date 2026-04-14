from opencompass.models import HuggingFaceQwenMoeCustom
import torch

models = [
    dict(
        type=HuggingFaceQwenMoeCustom,
        abbr='qwen1.5-moe-a2.7b-hf',
        path='/data/cyx/models/out_piqa_qw_lowrank_4bit_v1',
        tokenizer_path='/data/cyx/models/out_piqa_qw_lowrank_4bit_v1',
        moe_modeling_module='Qwen_MoE.modeling.modeling_moe',
        moe_config_module='Qwen_MoE.modeling.configuration_moe',
        max_out_len=192,
        batch_size=2,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
        ),
        run_cfg=dict(num_gpus=1),
    )
]
