export CUDA_VISIBLE_DEVICES=2

python run.py \
    --datasets hellaswag_ppl \
    --models dynamic_moe \
    --debug
