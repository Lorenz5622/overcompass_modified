#!/usr/bin/env bash
set -euo pipefail

# ========= 配置区 =========
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${RUN_PY:-run.py}"
MODEL_CFG="${MODEL_CFG:-dynamic_moe_dm}"
DATASET="${DATASET:-piqa_ppl}"
LOG_DIR="${LOG_DIR:-./logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/dynamic_moe_dm_batch}"
EXTRA_ARGS=(--debug)

# 每一项格式:
#   模型路径|tokenizer路径|数据集|结果名
# 其中 tokenizer路径、数据集、结果名都可以留空:
#   tokenizer路径为空时默认等于模型路径
#   数据集为空时默认使用上面的 DATASET
#   结果名为空时默认使用模型目录名
declare -a EVAL_SPECS=(
  # "/data/cyx/models/out_dm_cb_za02_r175_tp032_b22|/data/cyx/models/out_dm_cb_za02_r175_tp032_b22|piqa_ppl|out_dm_cb_za02_r175_tp032_b22"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp040_b22_highstart|/data/cyx/models/out_dm_cb_za02_r175_tp040_b22_highstart|piqa_ppl|out_dm_cb_za02_r175_tp040_b22_highstart"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp030_b22_later|/data/cyx/models/out_dm_cb_za02_r175_tp030_b22_later|piqa_ppl|out_dm_cb_za02_r175_tp030_b22_later"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late|/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late|piqa_ppl|out_dm_cb_za02_r175_tp032_b22_late"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp031_b22_late_t085|/data/cyx/models/out_dm_cb_za02_r175_tp031_b22_late_t085|piqa_ppl|out_dm_cb_za02_r175_tp031_b22_late_t085"
  "/data/cyx/models/CB_norestrict_abl_newlr01_ctx03|/data/cyx/models/CB_norestrict_abl_newlr01_ctx03|piqa_ppl|CB_norestrict_abl_newlr01_ctx03"
  "/data/cyx/models/CB_norestrict_abl_newlr005_ctx03|/data/cyx/models/CB_norestrict_abl_newlr005_ctx03|piqa_ppl|CB_norestrict_abl_newlr005_ctx03"
  # "/data/cyx/models/CB_norestrict_abl_newlr01_ctx01|/data/cyx/models/CB_norestrict_abl_newlr01_ctx01|piqa_ppl|CB_norestrict_abl_newlr01_ctx01"
  # "/data/cyx/models/CB_norestrict_abl_ctx01|/data/cyx/models/CB_norestrict_abl_ctx01|piqa_ppl|CB_norestrict_abl_ctx01"
  
  # "/data/cyx/models/CB_norestrict_abl_noctx|/data/cyx/models/CB_norestrict_abl_noctx|piqa_ppl|CB_norestrict_abl_noctx"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm112|/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm112|piqa_ppl|out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm112"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm115|/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm115|piqa_ppl|out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm115"
  # "/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm112|/data/cyx/models/out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm112|piqa_ppl|out_dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm112"
  
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
