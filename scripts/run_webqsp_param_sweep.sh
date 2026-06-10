#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./scripts/train_deepspeed.sh}"

MODEL="${MODEL:-llama}"
TASK="${TASK:-QA}"
DATASET="${DATASET:-webqsp}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EMB="${EMB:-E5-base}"
case "${EMB}" in
  E5-base|e5-base|e5|intfloat/multilingual-e5-base)
    EMB_LABEL="E5-base"
    ;;
  Alibaba-NLP/gte-modernbert-base|gte-modernbert-base|GTE-modernbert-base|gte-modernbert|GTE-modernbert)
    EMB_LABEL="gte-modernbert-base"
    ;;
  *)
    EMB_LABEL="${EMB//\//_}"
    ;;
esac
PREFIX="grlprof-llama-2-7b-chat-${EMB_LABEL}-${DATASET}-projector"

MODEL_BASE="${MODEL_BASE:-./llama/llama-2-7b-chat-hf}"
CONV_MODE="${CONV_MODE:-llaga_llama_2}"
CACHE_DIR="${CACHE_DIR:-./checkpoints/hf_cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export LLAGA_SKIP_INITIAL_EVAL="${LLAGA_SKIP_INITIAL_EVAL:-1}"
export LLAGA_MAX_LEN="${LLAGA_MAX_LEN:-2600}"
export LLAGA_GRAD_ACCUM="${LLAGA_GRAD_ACCUM:-4}"
export LLAGA_DEEPSPEED_CONFIG="${LLAGA_DEEPSPEED_CONFIG:-./configs/deepspeed/zero3.json}"
export LLAGA_DATALOADER_NUM_WORKERS="${LLAGA_DATALOADER_NUM_WORKERS:-8}"
export LLAGA_LOGGING_STEPS="${LLAGA_LOGGING_STEPS:-2}"
export LLAGA_SAVE_STEPS="${LLAGA_SAVE_STEPS:-600}"
export LLAGA_REPORT_TO="${LLAGA_REPORT_TO:-none}"

EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_MAX_GRAPH_PATHS="${EVAL_MAX_GRAPH_PATHS:-256}"
EVAL_MAX_MEMORY_PER_GPU="${EVAL_MAX_MEMORY_PER_GPU:-20GiB}"
EVAL_SPLITS="${EVAL_SPLITS:-validation test}"

RUN_QWEN_SEMANTIC="${RUN_QWEN_SEMANTIC:-1}"
SEMANTIC_SPLITS="${SEMANTIC_SPLITS:-validation}"
QWEN_PROXY_MODE="${QWEN_PROXY_MODE:-direct}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-5}"
QWEN_MAX_WORKERS="${QWEN_MAX_WORKERS:-8}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-768}"
QWEN_DISABLE_THINKING="${QWEN_DISABLE_THINKING:-0}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
FORCE_QWEN="${FORCE_QWEN:-0}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-3}"

LOG_DIR="${ROOT_DIR}/checkpoints/webqsp/_batch_logs"
mkdir -p "${LOG_DIR}"

# Current signal:
# - 2 epochs + warmup 0.05 is better than the 1-epoch runs.
# - lr=1.5e-3 is currently the best tested point.
# So the default queue does a small local search around that setup.
EXPERIMENTS=(
  "lr1.75e-3warmup0.05epochs2|1.75e-3|0.05|2|0."
  "lr1.25e-3warmup0.05epochs2|1.25e-3|0.05|2|0."
  "lr1.5e-3warmup0.03epochs2|1.5e-3|0.03|2|0."
  "lr1.5e-3warmup0.05epochs2wd0.01|1.5e-3|0.05|2|0.01"
)


timestamp() {
  date "+%F %T"
}


log() {
  echo "[$(timestamp)] $*"
}


run_logged() {
  local log_file="$1"
  shift
  log "$*" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
}


latest_checkpoint_name() {
  local output_dir="$1"
  local latest_step
  latest_step="$(
    find "${output_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
      | sed 's/^checkpoint-//' \
      | sort -n \
      | tail -n 1
  )"

  if [ -n "${latest_step}" ]; then
    echo "checkpoint-${latest_step}"
    return 0
  fi

  if [ -f "${output_dir}/config.json" ] && [ -f "${output_dir}/mm_projector.bin" ]; then
    echo "final"
    return 0
  fi

  return 1
}


should_skip_qwen() {
  local extracted_json="$1"
  local backup_json="${extracted_json%.json}.backup.json"
  [ "${FORCE_QWEN}" != "1" ] && [ -f "${backup_json}" ]
}


train_one() {
  local suffix="$1"
  local lr="$2"
  local warmup="$3"
  local epochs="$4"
  local weight_decay="$5"
  local output_dir="$6"
  local log_file="$7"

  if [ "${FORCE_TRAIN}" != "1" ] && [ -f "${output_dir}/trainer_state.json" ]; then
    log "Skip training for ${suffix}: ${output_dir}/trainer_state.json already exists." | tee -a "${log_file}"
    return 0
  fi

  run_logged "${log_file}" \
    env \
      LLAGA_LR="${lr}" \
      LLAGA_WARMUP_RATIO="${warmup}" \
      LLAGA_EPOCHS="${epochs}" \
      LLAGA_WEIGHT_DECAY="${weight_decay}" \
      LLAGA_OUTPUT_SUFFIX="${suffix}" \
      "${TRAIN_SCRIPT}" "${MODEL}" "${TASK}" "${DATASET}" "${BATCH_SIZE}" "${EMB}"
}


eval_one() {
  local output_dir="$1"
  local log_file="$2"
  local summary_path="${output_dir}/eval_results/webqsp_checkpoint_eval/summary.json"

  if [ "${FORCE_EVAL}" != "1" ] && [ -f "${summary_path}" ]; then
    log "Skip checkpoint eval for ${output_dir}: summary already exists." | tee -a "${log_file}"
    return 0
  fi

  read -r -a eval_splits_array <<< "${EVAL_SPLITS}"
  local cmd=(
    "${PYTHON_BIN}" "eval/eval_webqsp_checkpoints.py"
    --output_dir "${output_dir}"
    --model_base "${MODEL_BASE}"
    --conv_mode "${CONV_MODE}"
    --pretrained_embedding_type "${EMB}"
    --cache_dir "${CACHE_DIR}"
    --splits "${eval_splits_array[@]}"
    --device "${EVAL_DEVICE}"
    --max_graph_paths "${EVAL_MAX_GRAPH_PATHS}"
    --max_memory_per_gpu "${EVAL_MAX_MEMORY_PER_GPU}"
  )
  run_logged "${log_file}" "${cmd[@]}"
}


extract_and_semantic_one() {
  local output_dir="$1"
  local log_file="$2"
  local results_dir="${output_dir}/eval_results/webqsp_checkpoint_eval"
  local checkpoint_name

  checkpoint_name="$(latest_checkpoint_name "${output_dir}")"
  log "Using ${checkpoint_name} as the last checkpoint for ${output_dir}." | tee -a "${log_file}"

  read -r -a semantic_splits_array <<< "${SEMANTIC_SPLITS}"
  local split details_jsonl extracted_json semantic_cmd

  for split in "${semantic_splits_array[@]}"; do
    details_jsonl="${results_dir}/${checkpoint_name}_${split}_details.jsonl"
    if [ ! -f "${details_jsonl}" ]; then
      log "Skip ${split} extraction for ${output_dir}: details file not found at ${details_jsonl}." | tee -a "${log_file}"
      continue
    fi

    extracted_json="${results_dir}/${checkpoint_name}_${split}_details_extracted.json"
    if [ "${FORCE_EXTRACT}" != "1" ] && [ -f "${extracted_json}" ]; then
      log "Skip extraction for ${split}: ${extracted_json} already exists." | tee -a "${log_file}"
    else
      run_logged "${log_file}" \
        "${PYTHON_BIN}" "checkpoints/webqsp/extract_fields_to_json.py" \
        "${details_jsonl}" \
        --output "${extracted_json}"
    fi

    if [ "${RUN_QWEN_SEMANTIC}" != "1" ]; then
      continue
    fi

    if should_skip_qwen "${extracted_json}"; then
      log "Skip Qwen semantic match for ${split}: backup file already exists." | tee -a "${log_file}"
      continue
    fi

    semantic_cmd=(
      "${PYTHON_BIN}" "checkpoints/webqsp/semantic_match_with_qwen.py"
      "${extracted_json}"
      --proxy-mode "${QWEN_PROXY_MODE}"
      --batch-size "${QWEN_BATCH_SIZE}"
      --max-workers "${QWEN_MAX_WORKERS}"
      --max-tokens "${QWEN_MAX_TOKENS}"
      --force
    )
    if [ "${QWEN_DISABLE_THINKING}" = "1" ]; then
      semantic_cmd+=(--disable-thinking)
    fi
    run_logged "${log_file}" "${semantic_cmd[@]}"
  done
}


main() {
  if [ "${RUN_QWEN_SEMANTIC}" = "1" ] && [ -z "${DASHSCOPE_API_KEY:-${BAILIAN_API_KEY:-}}" ]; then
    echo "RUN_QWEN_SEMANTIC=1 but DASHSCOPE_API_KEY / BAILIAN_API_KEY is not set." >&2
    exit 1
  fi

  local index=0
  local entry suffix lr warmup epochs weight_decay output_dir log_file
  for entry in "${EXPERIMENTS[@]}"; do
    index=$((index + 1))
    if [ "${index}" -gt "${MAX_EXPERIMENTS}" ]; then
      break
    fi

    IFS='|' read -r suffix lr warmup epochs weight_decay <<< "${entry}"
    output_dir="${ROOT_DIR}/checkpoints/${DATASET}/${PREFIX}-${suffix}"
    log_file="${LOG_DIR}/${suffix}.log"

    log "========== Experiment ${index}/${MAX_EXPERIMENTS}: ${suffix} ==========" | tee -a "${log_file}"
    log "Output dir: ${output_dir}" | tee -a "${log_file}"
    log "Params: lr=${lr}, warmup=${warmup}, epochs=${epochs}, weight_decay=${weight_decay}" | tee -a "${log_file}"

    train_one "${suffix}" "${lr}" "${warmup}" "${epochs}" "${weight_decay}" "${output_dir}" "${log_file}"
    eval_one "${output_dir}" "${log_file}"
    extract_and_semantic_one "${output_dir}" "${log_file}"
  done

  log "All requested experiments finished."
}


main "$@"
