#!/usr/bin/env bash
set -uo pipefail

# ========= 配置区 =========
AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:-0}")
LOG_DIR="${LOG_DIR:-./logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${RUN_PY:-run.py}"
MODEL_CFG="${MODEL_CFG:-dynamic_moe}"
EXTRA_ARGS=(--debug)
# ==========================

mkdir -p "$LOG_DIR"

declare -a MODEL_DATASET_PAIRS=(
  # "/data/cyx/models/your_dynamic_moe_ori_ckpt|piqa_ppl"
)

total_start_ts=$(date +%s)
echo "==> BATCH START: $(date '+%F %T')"
echo "可用显卡: ${AVAILABLE_GPUS[*]}"
echo "日志目录: $LOG_DIR"
echo "模型配置: $MODEL_CFG"
echo ""

declare -A GPU_PID=()
declare -A PID_DESC=()
declare -A PID_LOG=()
declare -A PID_START=()

FAIL_COUNT=0

cleanup() {
  echo ""
  echo "==> 清理后台任务..."
  for pid in "${GPU_PID[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

safe_basename() {
  local p="$1"
  basename "${p%/}"
}

start_task_on_gpu() {
  local gpu_id="$1"
  local model_path="$2"
  local dataset="$3"

  local base
  base="$(safe_basename "$model_path")"
  local log_file="${LOG_DIR}/eval_${base}_${dataset}.log"

  echo "[SCHED] 分配 GPU ${gpu_id}: ${base} | dataset=${dataset} | log=${log_file}"

  (
    set -e
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export OC_MODEL_PATH="$model_path"
    export OC_TOKENIZER_PATH="$model_path"
    export OC_MODEL_NAME="$base"

    exec "$PYTHON_BIN" "$RUN_PY" --datasets "$dataset" --models "$MODEL_CFG" "${EXTRA_ARGS[@]}"
  ) >"$log_file" 2>&1 &

  local pid=$!
  GPU_PID["$gpu_id"]="$pid"
  PID_DESC["$pid"]="${base} | ${dataset} | GPU ${gpu_id}"
  PID_LOG["$pid"]="$log_file"
  PID_START["$pid"]="$(date +%s)"
}

find_free_gpu() {
  for g in "${AVAILABLE_GPUS[@]}"; do
    local pid="${GPU_PID[$g]:-}"
    if [[ -z "${pid}" ]]; then
      echo "$g"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

wait_one_finish() {
  local finished_pid=""
  local rc=0

  if wait -n -p finished_pid 2>/dev/null; then
    rc=0
  else
    rc=$?
    set +e
    wait -n
    rc=$?
    set -e 2>/dev/null || true

    for g in "${AVAILABLE_GPUS[@]}"; do
      local pid="${GPU_PID[$g]:-}"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        finished_pid="$pid"
        break
      fi
    done
  fi

  for g in "${AVAILABLE_GPUS[@]}"; do
    if [[ "${GPU_PID[$g]:-}" == "$finished_pid" ]]; then
      GPU_PID["$g"]=""
      break
    fi
  done

  local exit_code=0
  if [[ -n "$finished_pid" ]]; then
    set +e
    wait "$finished_pid"
    exit_code=$?
    set -e 2>/dev/null || true
  else
    exit_code=$rc
  fi

  local end_ts
  end_ts=$(date +%s)
  local start_ts="${PID_START[$finished_pid]:-$end_ts}"
  local elapsed=$((end_ts - start_ts))
  local desc="${PID_DESC[$finished_pid]:-UNKNOWN}"
  local log="${PID_LOG[$finished_pid]:-UNKNOWN}"

  if [[ "$exit_code" -eq 0 ]]; then
    echo "[DONE] OK    (${elapsed}s) ${desc} | log=${log}"
  else
    echo "[DONE] FAIL  (${elapsed}s) ${desc} | exit=${exit_code} | log=${log}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  unset "PID_DESC[$finished_pid]" "PID_LOG[$finished_pid]" "PID_START[$finished_pid]" 2>/dev/null || true
}

set -e
for pair in "${MODEL_DATASET_PAIRS[@]}"; do
  IFS='|' read -r model_path dataset <<< "$pair"

  if [[ -z "${model_path:-}" || -z "${dataset:-}" ]]; then
    echo "[SKIP] 非法配置: ${pair}"
    continue
  fi

  while ! free_gpu="$(find_free_gpu)"; do
    wait_one_finish
  done

  start_task_on_gpu "$free_gpu" "$model_path" "$dataset"
done

while :; do
  any_running=0
  for g in "${AVAILABLE_GPUS[@]}"; do
    pid="${GPU_PID[$g]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      any_running=1
      break
    fi
  done
  [[ "$any_running" -eq 0 ]] && break
  wait_one_finish
done

total_elapsed=$(( $(date +%s) - total_start_ts ))
echo ""
echo "========================================"
echo "dynamic_moe_ori 批量测评完成!"
echo "总耗时: ${total_elapsed} 秒"
echo "失败任务数: ${FAIL_COUNT}"
echo "========================================"

# [[ "$FAIL_COUNT" -eq 0 ]]
