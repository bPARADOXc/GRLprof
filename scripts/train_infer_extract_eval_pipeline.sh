#!/bin/bash
set -euo pipefail

# End-to-end QA pipeline:
#   1. train with scripts/train_deepspeed.sh
#   2. run checkpoint inference with eval/eval_webqsp_checkpoints.py
#   3. convert *_details.jsonl with tools/extract_fields_to_json.py
#   4. score extracted JSON files with eval.py
#
# Positional args:
#   1: MODEL       llama or llama3, default llama3
#   2: DATASET     cwq or webqsp, default cwq
#   3: DATA_DIR    merged data dir under this repo, default <dataset>_may_merged
#   4: BATCH_SIZE  train batch size, default 1
#   5: EMB         embedding type, default Alibaba-NLP/gte-modernbert-base
#
# Common overrides:
#   OUTPUT_BASE=./checkpoints  # weights go under ${OUTPUT_BASE}/${DATASET}
#   RUN_PREFLIGHT=0|1          # validate merged JSON/path embeddings first
#   RUN_TRAIN=0|1 RUN_INFER=0|1 RUN_EXTRACT=0|1 RUN_EVAL_PY=0|1
#   EVAL_SPLITS="validation test"     # default: all checkpoints validation, final test
#   EVAL_CHECKPOINTS="final 300"      # optional, otherwise eval script discovers all
#   EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS=1
#   EVAL_TORCH_DTYPE=auto|bfloat16|float16
#   EVAL_CUDA_VISIBLE_DEVICES=1,2     # optional: use different GPUs for inference
#   NO_GRAPH_TOKEN=1
#   LLAGA_* variables accepted by scripts/train_deepspeed.sh are passed through.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2 \
#   LLAGA_MAX_LEN=2600 \
#   LLAGA_GRAD_ACCUM=4 \
#   LLAGA_DEEPSPEED_CONFIG=./scripts/zero3.json \
#   scripts/train_infer_extract_eval_pipeline.sh \
#     llama3 cwq data/cwq_may_merged 1 Alibaba-NLP/gte-modernbert-base

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./scripts/train_deepspeed.sh}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-scripts/preflight_core_pipeline.py}"
INFER_SCRIPT="${INFER_SCRIPT:-eval/eval_webqsp_checkpoints.py}"
EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-tools/extract_fields_to_json.py}"
EVAL_PY_SCRIPT="${EVAL_PY_SCRIPT:-eval.py}"

MODEL="${1:-${MODEL:-llama3}}"
DATASET="${2:-${DATASET:-cwq}}"
DATA_DIR_INPUT="${3:-${DATA_DIR:-data/${DATASET}_may_merged}}"
BATCH_SIZE="${4:-${BATCH_SIZE:-1}}"
EMB="${5:-${EMB:-Alibaba-NLP/gte-modernbert-base}}"
TASK="${TASK:-QA}"

OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_EVAL_PY="${RUN_EVAL_PY:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_SKIP_EMBEDDING_CONTENTS="${PREFLIGHT_SKIP_EMBEDDING_CONTENTS:-0}"

EVAL_SPLITS="${EVAL_SPLITS:-validation test}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_MAX_GRAPH_PATHS="${EVAL_MAX_GRAPH_PATHS:-${LLAGA_MAX_GRAPH_PATHS:-32}}"
EVAL_MAX_MEMORY_PER_GPU="${EVAL_MAX_MEMORY_PER_GPU:-20GiB}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-}"
EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS="${EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS:-0}"
EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-auto}"
EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-}"
NO_GRAPH_TOKEN="${NO_GRAPH_TOKEN:-0}"
CACHE_DIR="${CACHE_DIR:-./checkpoints/hf_cache}"

EXTRACT_PATTERN="${EXTRACT_PATTERN:-*_details.jsonl}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
EVAL_PY_SPLITS="${EVAL_PY_SPLITS:-${EVAL_SPLITS}}"
EVAL_ALIAS_DIR="${EVAL_ALIAS_DIR:-}"

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

shell_join() {
  local joined=""
  local arg
  for arg in "$@"; do
    printf -v arg "%q" "${arg}"
    joined+="${arg} "
  done
  echo "${joined}"
}

run_logged() {
  local log_file="$1"
  shift
  log "Command: $(shell_join "$@")" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
}

abs_path() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "${ROOT_DIR}/$1" ;;
  esac
}

canonical_split() {
  case "$1" in
    val) echo "validation" ;;
    *) echo "$1" ;;
  esac
}

resolve_model_defaults() {
  case "${MODEL}" in
    llama)
      DEFAULT_MODEL_BASE="${ROOT_DIR}/llama/llama-2-7b-chat-hf"
      DEFAULT_CONV_MODE="llaga_llama_2"
      ;;
    llama3)
      DEFAULT_MODEL_BASE="${ROOT_DIR}/llama/Meta-Llama-3-8B-Instruct"
      if [ ! -d "${DEFAULT_MODEL_BASE}" ]; then
        DEFAULT_MODEL_BASE="meta-llama/Meta-Llama-3-8B-Instruct"
      fi
      DEFAULT_CONV_MODE="llama_3"
      ;;
    *)
      DEFAULT_MODEL_BASE="${MODEL_BASE:-}"
      DEFAULT_CONV_MODE="${CONV_MODE:-}"
      if [ -z "${DEFAULT_MODEL_BASE}" ] || [ -z "${DEFAULT_CONV_MODE}" ]; then
        echo "Unsupported MODEL=${MODEL}. Set MODEL_BASE and CONV_MODE, or use llama/llama3." >&2
        exit 1
      fi
      ;;
  esac

  MODEL_BASE="${MODEL_BASE:-${DEFAULT_MODEL_BASE}}"
  CONV_MODE="${CONV_MODE:-${DEFAULT_CONV_MODE}}"
}

parse_output_dir_from_log() {
  local log_file="$1"
  awk -F'OUTPUT_DIR: ' '/^OUTPUT_DIR:/ {print $2}' "${log_file}" | tail -n 1
}

find_latest_output_dir() {
  local search_dir="${OUTPUT_ROOT%/}/${DATASET}"
  [ -d "${search_dir}" ] || return 1
  find "${search_dir}" -maxdepth 1 -mindepth 1 -type d ! -name "pipeline_logs" -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

DATA_DIR_ABS="$(abs_path "${DATA_DIR_INPUT}")"
resolve_model_defaults

PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${OUTPUT_ROOT%/}/${DATASET}/pipeline_logs}"
mkdir -p "${PIPELINE_LOG_ROOT}"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_ROOT}/${PIPELINE_ID}"
mkdir -p "${PIPELINE_LOG_DIR}"

log "ROOT_DIR=${ROOT_DIR}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"
log "MODEL=${MODEL}, DATASET=${DATASET}, DATA_DIR=${DATA_DIR_ABS}, EMB=${EMB}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"
log "OUTPUT_ROOT=${OUTPUT_ROOT}, PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  preflight_cmd=(
    "${PYTHON_BIN}" "${PREFLIGHT_SCRIPT}"
    --dataset "${DATASET}"
    --data-dir "${DATA_DIR_ABS}"
    --embedding-type "${EMB}"
  )
  if [ "${PREFLIGHT_SKIP_EMBEDDING_CONTENTS}" = "1" ]; then
    preflight_cmd+=(--skip-embedding-contents)
  fi
  run_logged "${PIPELINE_LOG_DIR}/preflight.log" "${preflight_cmd[@]}"
fi

if [ "${RUN_TRAIN}" = "1" ]; then
  export LLAGA_OUTPUT_ROOT="${OUTPUT_ROOT}"
  export LLAGA_DATA_DIR="${DATA_DIR_ABS}"

  train_log="${PIPELINE_LOG_DIR}/train.log"
  run_logged "${train_log}" \
    "${TRAIN_SCRIPT}" \
    "${MODEL}" \
    "${TASK}" \
    "${DATASET}" \
    "${BATCH_SIZE}" \
    "${EMB}" \
    "${DATA_DIR_ABS}"

  parsed_output_dir="$(parse_output_dir_from_log "${train_log}")"
  if [ -n "${parsed_output_dir}" ]; then
    OUTPUT_DIR="$(abs_path "${parsed_output_dir}")"
  fi
fi

if [ -z "${OUTPUT_DIR}" ]; then
  latest_dir="$(find_latest_output_dir || true)"
  if [ -n "${latest_dir}" ]; then
    OUTPUT_DIR="${latest_dir}"
  else
    echo "Could not determine OUTPUT_DIR. Set OUTPUT_DIR explicitly when RUN_TRAIN=0." >&2
    exit 1
  fi
fi

if [ "${NO_GRAPH_TOKEN}" = "1" ]; then
  RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_DIR}/eval_results/${DATASET}_checkpoint_eval_no_graph}"
else
  RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_DIR}/eval_results/${DATASET}_checkpoint_eval}"
fi
mkdir -p "${RESULTS_DIR}"

cat > "${RESULTS_DIR}/pipeline_config.json" <<EOF
{
  "pipeline_id": "${PIPELINE_ID}",
  "model": "${MODEL}",
  "dataset": "${DATASET}",
  "data_dir": "${DATA_DIR_ABS}",
  "embedding": "${EMB}",
  "output_dir": "${OUTPUT_DIR}",
  "results_dir": "${RESULTS_DIR}",
  "model_base": "${MODEL_BASE}",
  "conv_mode": "${CONV_MODE}",
  "eval_splits": "${EVAL_SPLITS}",
  "eval_torch_dtype": "${EVAL_TORCH_DTYPE}",
  "eval_cuda_visible_devices": "${EVAL_CUDA_VISIBLE_DEVICES}",
  "no_graph_token": "${NO_GRAPH_TOKEN}"
}
EOF

log "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"
log "RESULTS_DIR=${RESULTS_DIR}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"

if [ "${RUN_INFER}" = "1" ]; then
  infer_log="${PIPELINE_LOG_DIR}/infer.log"
  infer_cmd=(
    "${PYTHON_BIN}" "${INFER_SCRIPT}"
    --dataset "${DATASET}"
    --data_dir "${DATA_DIR_ABS}"
    --output_dir "${OUTPUT_DIR}"
    --model_base "${MODEL_BASE}"
    --conv_mode "${CONV_MODE}"
    --cache_dir "${CACHE_DIR}"
    --pretrained_embedding_type "${EMB}"
    --splits
  )
  for split in ${EVAL_SPLITS}; do
    infer_cmd+=("$(canonical_split "${split}")")
  done
  infer_cmd+=(
    --device "${EVAL_DEVICE}"
    --torch_dtype "${EVAL_TORCH_DTYPE}"
    --max_graph_paths "${EVAL_MAX_GRAPH_PATHS}"
    --max_memory_per_gpu "${EVAL_MAX_MEMORY_PER_GPU}"
    --results_dir "${RESULTS_DIR}"
  )
  if [ -n "${EVAL_DEVICE_MAP}" ]; then
    infer_cmd+=(--device_map "${EVAL_DEVICE_MAP}")
  fi
  if [ "${EVAL_LIMIT}" != "-1" ]; then
    infer_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  if [ -n "${EVAL_CHECKPOINTS}" ]; then
    infer_cmd+=(--checkpoints)
    for checkpoint in ${EVAL_CHECKPOINTS}; do
      infer_cmd+=("${checkpoint}")
    done
  fi
  if [ "${EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS}" = "1" ]; then
    infer_cmd+=(--all-splits-for-all-checkpoints)
  fi
  if [ "${NO_GRAPH_TOKEN}" = "1" ]; then
    infer_cmd+=(--no_graph_token)
  fi

  if [ -n "${EVAL_CUDA_VISIBLE_DEVICES}" ]; then
    run_logged "${infer_log}" env CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${infer_cmd[@]}"
  else
    run_logged "${infer_log}" "${infer_cmd[@]}"
  fi
fi

if [ "${RUN_EXTRACT}" = "1" ]; then
  extract_log="${PIPELINE_LOG_DIR}/extract.log"
  extract_cmd=(
    "${PYTHON_BIN}" "${EXTRACT_SCRIPT}"
    "${RESULTS_DIR}"
    --pattern "${EXTRACT_PATTERN}"
    --recursive
  )
  if [ "${FORCE_EXTRACT}" = "1" ]; then
    extract_cmd+=(--overwrite)
  fi
  run_logged "${extract_log}" "${extract_cmd[@]}"
fi

if [ "${RUN_EVAL_PY}" = "1" ]; then
  eval_py_log="${PIPELINE_LOG_DIR}/eval_py.log"
  for split in ${EVAL_PY_SPLITS}; do
    split="$(canonical_split "${split}")"
    eval_py_cmd=(
      "${PYTHON_BIN}" "${EVAL_PY_SCRIPT}"
      --dataset "${DATASET}"
      --checkpoint_eval_dir "${RESULTS_DIR}"
      --split "${split}"
      --summary_file "${RESULTS_DIR}/eval_py_summary_${split}.json"
    )
    if [ -n "${EVAL_ALIAS_DIR}" ]; then
      eval_py_cmd+=(--alias_dir "${EVAL_ALIAS_DIR}")
    fi
    run_logged "${eval_py_log}" "${eval_py_cmd[@]}"
  done
fi

log "Pipeline finished. Results: ${RESULTS_DIR}" | tee -a "${PIPELINE_LOG_DIR}/pipeline.log"
