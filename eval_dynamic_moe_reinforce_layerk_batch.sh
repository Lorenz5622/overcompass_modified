#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${RUN_PY:-run.py}"
MODEL_CFG="${MODEL_CFG:-dynamic_moe_reinforce_layerk}"
DATASET="${DATASET:-piqa_ppl}"
LOG_DIR="${LOG_DIR:-./logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/dynamic_moe_reinforce_layerk_batch}"
EXTRA_ARGS=(--debug)

declare -a EVAL_SPECS=(
  # "/mnt/data/models/reinforce_layerk_piqa_s2|/mnt/data/models/reinforce_layerk_piqa_s2|piqa_ppl|reinforce_layerk"
  # "/mnt/data/models/reinforce_layerk_siqa_s2|/mnt/data/models/reinforce_layerk_siqa_s2|siqa_ppl|reinforce_layerk"
  # "/data/cyx/models/reinforce_layerk_winogrande|/data/cyx/models/reinforce_layerk_winogrande|winograd_ppl|reinforce_layerk"
  # "/mnt/data/models/reinforce_layerk_arc_e_s2|/mnt/data/models/reinforce_layerk_arc_e_s2|arc_e_ppl|reinforce_layerk"
  "/mnt/data/models/reinforce_layerk_obqa_s2|/mnt/data/models/reinforce_layerk_obqa_s2|obqa_ppl|reinforce_layerk"
)

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

safe_name() {
  local value="$1"
  value="${value%/}"
  basename "$value"
}

is_hf_model_dir() {
  local dir="$1"
  [[ -f "${dir}/config.json" && -f "${dir}/tokenizer_config.json" ]]
}

resolve_checkpoint_dir() {
  local dir="$1"
  local default_tag="${CHECKPOINT_TAG:-best}"

  if [[ ! -d "$dir" ]]; then
    echo "$dir"
    return
  fi

  if is_hf_model_dir "$dir"; then
    echo "$dir"
    return
  fi

  if [[ -n "$default_tag" && -d "${dir}/${default_tag}" ]] && is_hf_model_dir "${dir}/${default_tag}"; then
    echo "${dir}/${default_tag}"
    return
  fi

  if [[ -d "${dir}/best" ]] && is_hf_model_dir "${dir}/best"; then
    echo "${dir}/best"
    return
  fi

  if [[ -d "${dir}/last" ]] && is_hf_model_dir "${dir}/last"; then
    echo "${dir}/last"
    return
  fi

  echo "$dir"
}

run_one() {
  local model_path="$1"
  local tokenizer_path="$2"
  local dataset="$3"
  local model_name="$4"
  local resolved_model_path
  local resolved_tokenizer_path

  local log_file="${LOG_DIR}/eval_${model_name}_${dataset}.log"
  local work_dir="${OUTPUT_ROOT}/${model_name}_${dataset}"

  mkdir -p "$work_dir"

  resolved_model_path="$(resolve_checkpoint_dir "$model_path")"
  resolved_tokenizer_path="$(resolve_checkpoint_dir "$tokenizer_path")"

  echo "========================================"
  echo "开始评测: ${model_name}"
  echo "model_path     : ${model_path}"
  echo "resolved_model : ${resolved_model_path}"
  echo "tokenizer_path : ${tokenizer_path}"
  echo "resolved_tok   : ${resolved_tokenizer_path}"
  echo "dataset        : ${dataset}"
  echo "work_dir       : ${work_dir}"
  echo "log_file       : ${log_file}"
  echo "========================================"

  if CUDA_VISIBLE_DEVICES="$GPU_ID" \
    OC_MODEL_PATH="$resolved_model_path" \
    OC_TOKENIZER_PATH="$resolved_tokenizer_path" \
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
