#!/usr/bin/env bash
set -euo pipefail

# ========= 配置区 =========
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${RUN_PY:-run.py}"
MODEL_CFG="${MODEL_CFG:-dynamic_moe_yuan}"
DATASET="${DATASET:-piqa_ppl}"
LOG_DIR="${LOG_DIR:-./logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/dynamic_moe_yuan_batch}"
EXTRA_ARGS=(--debug)

# 每一项格式:
#   模型路径|tokenizer路径|数据集|结果名
# 其中 tokenizer路径、数据集、结果名都可以留空:
#   tokenizer路径为空时默认等于模型路径
#   数据集为空时默认使用上面的 DATASET
#   结果名为空时默认使用模型目录名
declare -a EVAL_SPECS=(
  "/data/cyx/models/out_piqa_yuan_lora|/data/cyx/models/out_piqa_yuan_lora|piqa_ppl|yuan_moe"
  "/data/cyx/models/out_arc_challenge_yuan_lora|/data/cyx/models/out_arc_challenge_yuan_lora|arc_c_ppl|yuan_moe"
  "/data/cyx/models/out_arc_easy_yuan_lora|/data/cyx/models/out_arc_easy_yuan_lora|arc_e_ppl|yuan_moe"
  "/data/cyx/models/out_oqa_yuan_lora|/data/cyx/models/out_oqa_yuan_lora|obqa_ppl|yuan_moe"
  # "/data/cyx/models/out_piqa_yuan_lora|/data/cyx/models/out_piqa_yuan_lora|piqa_ppl|yuan_moe"
  "/data/cyx/models/out_siqa_yuan_lora|/data/cyx/models/out_siqa_yuan_lora|siqa_ppl|yuan_moe"
  # "/data/cyx/models/your_dynamic_moe_yuan_ckpt|/data/cyx/models/your_dynamic_moe_yuan_ckpt|hellaswag_ppl|your_dynamic_moe_yuan_ckpt"
)
# ==========================

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

safe_name() {
  local value="$1"
  value="${value%/}"
  basename "$value"
}

run_one() {
  local model_path="$1"
  local tokenizer_path="$2"
  local dataset="$3"
  local model_name="$4"

  local log_file="${LOG_DIR}/eval_${model_name}_${dataset}.log"
  local work_dir="${OUTPUT_ROOT}/${model_name}_${dataset}"

  mkdir -p "$work_dir"

  echo "========================================"
  echo "开始评测: ${model_name}"
  echo "model_path     : ${model_path}"
  echo "tokenizer_path : ${tokenizer_path}"
  echo "dataset        : ${dataset}"
  echo "work_dir       : ${work_dir}"
  echo "log_file       : ${log_file}"
  echo "========================================"

  if CUDA_VISIBLE_DEVICES="$GPU_ID" \
    OC_MODEL_PATH="$model_path" \
    OC_TOKENIZER_PATH="$tokenizer_path" \
    OC_MODEL_NAME="$model_name" \
    "$PYTHON_BIN" "$RUN_PY" \
      --datasets "$dataset" \
      --models "$MODEL_CFG" \
      --work-dir "$work_dir" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$log_file"; then
    echo "[DONE] ${model_name} @ ${dataset}"
    return 0
  fi

  echo "[FAIL] ${model_name} @ ${dataset}，详情见 ${log_file}"
  return 1
}

total_start_ts=$(date +%s)
fail_count=0

for spec in "${EVAL_SPECS[@]}"; do
  IFS='|' read -r model_path tokenizer_path dataset model_name <<< "$spec"

  if [[ -z "$model_path" ]]; then
    echo "[SKIP] 空 model_path"
    continue
  fi
  [[ -n "$tokenizer_path" ]] || tokenizer_path="$model_path"
  [[ -n "$dataset" ]] || dataset="$DATASET"
  [[ -n "$model_name" ]] || model_name="$(safe_name "$model_path")"

  if ! run_one "$model_path" "$tokenizer_path" "$dataset" "$model_name"; then
    fail_count=$((fail_count + 1))
  fi
done

total_elapsed=$(( $(date +%s) - total_start_ts ))
echo ""
echo "批量评测完成，总耗时: ${total_elapsed}s，失败任务数: ${fail_count}"

[[ "$fail_count" -eq 0 ]]
