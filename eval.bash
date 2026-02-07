export CUDA_VISIBLE_DEVICES=0

python run.py \
    --datasets mmlu_ppl \
    --models dynamic_moe \
    --debug
