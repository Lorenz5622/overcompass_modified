export CUDA_VISIBLE_DEVICES=0

python run.py \
    --datasets piqa_ppl \
    --models hf_qwen1_5_moe_a2_7b_local \
    --debug
