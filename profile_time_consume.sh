#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

start_ts=$(date +%s)
start_hr=$(date '+%F %T')

echo "==> START: ${start_hr}"

python run.py \
  --datasets siqa_ppl \
  --models dynamic_moe \
  --debug |& tee run.log

end_ts=$(date +%s)
end_hr=$(date '+%F %T')

echo "==> END:   ${end_hr}"
echo "==> ELAPSED: $((end_ts - start_ts)) seconds"