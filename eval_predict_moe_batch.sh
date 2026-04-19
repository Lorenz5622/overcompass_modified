#!/usr/bin/env bash
set -uo pipefail

# ========= 配置区 =========
AVAILABLE_GPUS=(0)              # 例如 (0 1 2 3)
LOG_DIR="./logs"
PYTHON_BIN="python"
RUN_PY="run.py"
EXTRA_ARGS=(--debug)            # 你想固定带的额外参数
# ==========================

mkdir -p "$LOG_DIR"

declare -a MODEL_DATASET_PAIRS=(
  # "/data/cyx/models/out_oqa_lora_k1/|dynamic_moe|mmlu_ppl"
  # "/data/cyx/models/out_oqa_lora_k2/|dynamic_moe|mmlu_ppl"
  # "/data/cyx/models/out_oqa_lora/|dynamic_moe|mmlu_ppl"
  # "/data/cyx/models/Predict_MoE_oqa/|predict_moe|mmlu_ppl"
  # "/data/cyx/models/out_oqa_lora/|dynamic_moe|mmlu_ppl"
  # "/data/cyx/models/out_oqa_lora_k1/|dynamic_moe|obqa_ppl"
  # "/data/cyx/models/out_oqa_lora_k2/|dynamic_moe|obqa_ppl"
  # "/data/cyx/models/Predict_MoE_k19/|dynamic_moe|piqa_ppl"
  # "/data/cyx/models/Predict_MoE_k21/|dynamic_moe|piqa_ppl"
  # "/data/cyx/models/Predict_MoE_k23/|dynamic_moe|piqa_ppl"
  # "/data/cyx/models/Predict_MoE_k25/|dynamic_moe|piqa_ppl"
  "/data/cyx/models/Predict_MoE/|predict_moe|piqa_ppl"
  # "/data/cyx/models/out_hellaswag_lora/|dynamic_moe|hellaswag_ppl"
  # "/data/cyx/models/out_hellaswag_lora_k1/|dynamic_moe|hellaswag_ppl"
  # "/data/cyx/models/out_hellaswag_lora_k2/|dynamic_moe|hellaswag_ppl"
)

total_start_ts=$(date +%s)
echo "==> BATCH START: $(date '+%F %T')"
echo "可用显卡: ${AVAILABLE_GPUS[*]}"
echo "日志目录: $LOG_DIR"
echo ""

# GPU -> PID 映射（空表示空闲）
declare -A GPU_PID=()
# PID -> 描述/日志/开始时间
declare -A PID_DESC=()
declare -A PID_LOG=()
declare -A PID_START=()

# 记录失败数量
FAIL_COUNT=0

cleanup() {
  echo ""
  echo "==> 清理后台任务..."
  # 结束所有仍在跑的子任务
  for pid in "${GPU_PID[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 生成安全的日志文件名（处理末尾 /）
safe_basename() {
  local p="$1"
  basename "${p%/}"
}

start_task_on_gpu() {
  local gpu_id="$1"
  local model_path="$2"
  local model_name="$3"
  local dataset="$4"

  local base
  base="$(safe_basename "$model_path")"
  local log_file="${LOG_DIR}/eval_${base}_${dataset}.log"

  echo "[SCHED] 分配 GPU ${gpu_id}: ${base} | dataset=${dataset} | log=${log_file}"

  (
    set -e
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export OC_MODEL_PATH="$model_path"
    export OC_MODEL_NAME="$model_name"

    # 如果你希望 --models 跟着 model_name 变，用下一行替换固定 dynamic_moe 的写法：
    exec "$PYTHON_BIN" "$RUN_PY" --datasets "$dataset" --models "$model_name" "${EXTRA_ARGS[@]}"
    # exec "$PYTHON_BIN" "$RUN_PY" --datasets "$dataset" --models dynamic_moe "${EXTRA_ARGS[@]}"
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
      # pid 已经结束但尚未被 wait 回收，也认为空闲
      echo "$g"
      return 0
    fi
  done
  return 1
}

# 等待任一任务结束，并释放对应 GPU，打印结果
wait_one_finish() {
  local finished_pid=""
  local rc=0

  # bash5: wait -n -p var
  if wait -n -p finished_pid 2>/dev/null; then
    rc=0
  else
    rc=$?
    # 不支持 -p 的 fallback：等一个结束，然后通过轮询找出已结束 pid
    # 这里先 wait -n（可能返回非0）
    set +e
    wait -n
    rc=$?
    set -e 2>/dev/null || true

    # 找一个已经退出的 pid
    for g in "${AVAILABLE_GPUS[@]}"; do
      local pid="${GPU_PID[$g]:-}"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        finished_pid="$pid"
        break
      fi
    done
  fi

  # 找出它属于哪个 GPU 并释放
  for g in "${AVAILABLE_GPUS[@]}"; do
    if [[ "${GPU_PID[$g]:-}" == "$finished_pid" ]]; then
      GPU_PID["$g"]=""
      break
    fi
  done

  # 获取真实退出码：需要再 wait 一次具体 pid 才能拿到（如果刚才没回收）
  local exit_code=0
  if [[ -n "$finished_pid" ]]; then
    set +e
    wait "$finished_pid"
    exit_code=$?
    set -e 2>/dev/null || true
  else
    # 极少数情况下 fallback 找不到 finished pid
    exit_code=$rc
  fi

  local end_ts=$(date +%s)
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

  # 清理 pid 相关信息
  unset "PID_DESC[$finished_pid]" "PID_LOG[$finished_pid]" "PID_START[$finished_pid]" 2>/dev/null || true
}

# 主调度循环
set -e  # 仅主控启用 -e；任务内部也 set -e
for pair in "${MODEL_DATASET_PAIRS[@]}"; do
  IFS='|' read -r model_path model_name dataset <<< "$pair"

  # 等到有空闲 GPU
  while ! free_gpu="$(find_free_gpu)"; do
    wait_one_finish
  done

  start_task_on_gpu "$free_gpu" "$model_path" "$model_name" "$dataset"
done

# 等待剩余所有任务完成
while :; do
  # 检查是否还有在跑的 pid
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
echo "所有并行测评任务完成!"
echo "总耗时: ${total_elapsed} 秒"
echo "失败任务数: ${FAIL_COUNT}"
echo "========================================"

# 若你希望“有失败则整体返回非0”，可以取消注释：
# [[ "$FAIL_COUNT" -eq 0 ]]
