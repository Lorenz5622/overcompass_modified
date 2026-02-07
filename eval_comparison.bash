#!/bin/bash
# 对比微调前后模型的评测脚本

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 确保在judge环境中运行
# conda activate judge

python run.py \
    --datasets piqa_ppl \
    --models dynamic_moe_base predict_moe \
    --work-dir outputs/comparison
