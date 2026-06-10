#!/bin/bash
set -euo pipefail
#
# Example:
# DASHSCOPE_API_KEY=... \
# SEMANTIC_CHECKPOINTS="final checkpoint-678" \
# SEMANTIC_SPLITS="validation test" \
# scripts/train_eval_webqsp_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./scripts/train_deepspeed.sh}"

MODEL="${MODEL:-llama3}"
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
SAMPLE_SIZE="${SAMPLE_SIZE:-${LLAGA_SAMPLE_SIZE:--1}}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX-${LLAGA_OUTPUT_SUFFIX-llama3-rag-path16-lr1.25e-3warmup0.05epochs3}}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_SEMANTIC="${RUN_SEMANTIC:-1}"

SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"
SKIP_EVAL_IF_DONE="${SKIP_EVAL_IF_DONE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
FORCE_SEMANTIC="${FORCE_SEMANTIC:-0}"

EVAL_SPLITS="${EVAL_SPLITS:-validation test}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_MAX_GRAPH_PATHS="${EVAL_MAX_GRAPH_PATHS:-16}"
EVAL_MAX_MEMORY_PER_GPU="${EVAL_MAX_MEMORY_PER_GPU:-20GiB}"
EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS="${EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS:-1}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
NO_GRAPH_TOKEN="${NO_GRAPH_TOKEN:-0}"
CACHE_DIR="${CACHE_DIR:-../../checkpoint}"

SEMANTIC_CHECKPOINTS="${SEMANTIC_CHECKPOINTS:-latest}"
SEMANTIC_SPLITS="${SEMANTIC_SPLITS:-test}"
SEMANTIC_ALLOW_MISSING="${SEMANTIC_ALLOW_MISSING:-0}"

QWEN_PROXY_MODE="${QWEN_PROXY_MODE:-direct}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-5}"
QWEN_MAX_WORKERS="${QWEN_MAX_WORKERS:-8}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-768}"
QWEN_DISABLE_THINKING="${QWEN_DISABLE_THINKING:-0}"
QWEN_ONLY_TRUNCATED_PREDICTION_RAW="${QWEN_ONLY_TRUNCATED_PREDICTION_RAW:-0}"
QWEN_ONLY_GENERATION_FAILED="${QWEN_ONLY_GENERATION_FAILED:-0}"
QWEN_NO_BACKUP="${QWEN_NO_BACKUP:-0}"
QWEN_MAX_ITEMS="${QWEN_MAX_ITEMS:-}"
QWEN_SLEEP_SECONDS="${QWEN_SLEEP_SECONDS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export LLAGA_OUTPUT_SUFFIX="${OUTPUT_SUFFIX}"

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

resolve_model_defaults() {
  case "${MODEL}" in
    vicuna)
      PREFIX="grlprof-vicuna-7b-${EMB_LABEL}-2-${SAMPLE_SIZE}-linear-projector"
      DEFAULT_MODEL_BASE="lmsys/vicuna-7b-v1.5-16k"
      DEFAULT_CONV_MODE="v1"
      ;;
    vicuna_2layer)
      PREFIX="grlprof-vicuna-7b-${EMB_LABEL}-2-${SAMPLE_SIZE}-2-layer-mlp-projector"
      DEFAULT_MODEL_BASE="lmsys/vicuna-7b-v1.5-16k"
      DEFAULT_CONV_MODE="v1"
      ;;
    vicuna_4hop)
      PREFIX="grlprof-vicuna-7b-${EMB_LABEL}-4-hop-token-linear-projector"
      DEFAULT_MODEL_BASE="lmsys/vicuna-7b-v1.5-16k"
      DEFAULT_CONV_MODE="v1"
      ;;
    vicuna_4hop_2layer)
      PREFIX="grlprof-vicuna-7b-${EMB_LABEL}-4-hop-token-2-layer-mlp-projector"
      DEFAULT_MODEL_BASE="lmsys/vicuna-7b-v1.5-16k"
      DEFAULT_CONV_MODE="v1"
      ;;
    llama)
      PREFIX="grlprof-llama-2-7b-chat-${EMB_LABEL}-${DATASET}-projector"
      DEFAULT_MODEL_BASE="${ROOT_DIR}/llama/llama-2-7b-chat-hf"
      DEFAULT_CONV_MODE="llaga_llama_2"
      ;;
    llama3)
      PREFIX="grlprof-llama-3-8b-instruct-${EMB_LABEL}-${DATASET}-projector"
      DEFAULT_MODEL_BASE="${ROOT_DIR}/llama/Meta-Llama-3-8B-Instruct"
      if [ ! -d "${DEFAULT_MODEL_BASE}" ]; then
        DEFAULT_MODEL_BASE="meta-llama/Meta-Llama-3-8B-Instruct"
      fi
      DEFAULT_CONV_MODE="llama_3"
      ;;
    llama_4hop)
      PREFIX="grlprof-llama-2-7b-hf-${EMB_LABEL}-4-hop-token-linear-projector"
      DEFAULT_MODEL_BASE="meta-llama/Llama-2-7b-hf"
      DEFAULT_CONV_MODE="llaga_llama_2"
      ;;
    opt_2.7b)
      PREFIX="grlprof-opt-2.7b-${EMB_LABEL}-2-${SAMPLE_SIZE}-linear-projector"
      DEFAULT_MODEL_BASE="facebook/opt-2.7b"
      DEFAULT_CONV_MODE="v1"
      ;;
    opt_2.7b_4hop)
      PREFIX="grlprof-opt-2.7b-${EMB_LABEL}-4-hop-token-linear-projector-only-train-pretrain"
      DEFAULT_MODEL_BASE="facebook/opt-2.7b"
      DEFAULT_CONV_MODE="v1"
      ;;
    *)
      echo "Unsupported MODEL=${MODEL}. Update resolve_model_defaults in $0." >&2
      exit 1
      ;;
  esac
}

resolve_output_dir() {
  if [ -n "${OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$(abs_path "${OUTPUT_DIR}")"
    return
  fi

  OUTPUT_DIR="${ROOT_DIR}/checkpoints/${DATASET}/${PREFIX}"
  if [ -n "${OUTPUT_SUFFIX}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-${OUTPUT_SUFFIX}"
  fi
}

resolve_results_dir() {
  if [ -n "${RESULTS_DIR:-}" ]; then
    RESULTS_DIR="$(abs_path "${RESULTS_DIR}")"
    return
  fi

  local results_name="webqsp_checkpoint_eval"
  if [ "${NO_GRAPH_TOKEN}" = "1" ]; then
    results_name="webqsp_checkpoint_eval_no_graph"
  fi
  RESULTS_DIR="${OUTPUT_DIR}/eval_results/${results_name}"
}

latest_checkpoint_name() {
  local split
  for split in ${SEMANTIC_SPLITS}; do
    if [ -f "${RESULTS_DIR}/final_${split}_details.jsonl" ]; then
      echo "final"
      return 0
    fi
  done

  local latest_step=""
  local checkpoint_dir
  while IFS= read -r checkpoint_dir; do
    local name="${checkpoint_dir##*/}"
    local step="${name#checkpoint-}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && { [ -z "${latest_step}" ] || [ "${step}" -gt "${latest_step}" ]; }; then
      latest_step="${step}"
    fi
  done < <(find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" 2>/dev/null)

  if [ -n "${latest_step}" ]; then
    echo "checkpoint-${latest_step}"
    return 0
  fi

  return 1
}

resolve_semantic_checkpoints() {
  RESOLVED_SEMANTIC_CHECKPOINTS=()
  local token
  declare -A seen=()

  for token in ${SEMANTIC_CHECKPOINTS}; do
    if [ "${token}" = "all" ]; then
      local split file base checkpoint_name suffix
      for split in ${SEMANTIC_SPLITS}; do
        suffix="_${split}_details.jsonl"
        for file in "${RESULTS_DIR}"/*"${suffix}"; do
          [ -e "${file}" ] || continue
          base="${file##*/}"
          checkpoint_name="${base%"${suffix}"}"
          if [ -n "${checkpoint_name}" ] && [ -z "${seen[${checkpoint_name}]:-}" ]; then
            RESOLVED_SEMANTIC_CHECKPOINTS+=("${checkpoint_name}")
            seen["${checkpoint_name}"]=1
          fi
        done
      done
      continue
    fi

    if [ "${token}" = "latest" ]; then
      token="$(latest_checkpoint_name)"
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      token="checkpoint-${token}"
    fi

    if [ -n "${token}" ] && [ -z "${seen[${token}]:-}" ]; then
      RESOLVED_SEMANTIC_CHECKPOINTS+=("${token}")
      seen["${token}"]=1
    fi
  done

  if [ "${#RESOLVED_SEMANTIC_CHECKPOINTS[@]}" -eq 0 ]; then
    echo "No semantic checkpoints resolved from SEMANTIC_CHECKPOINTS=${SEMANTIC_CHECKPOINTS}" >&2
    exit 1
  fi
}

train_stage() {
  if [ "${RUN_TRAIN}" != "1" ]; then
    log "Skip train stage because RUN_TRAIN=${RUN_TRAIN}."
    return 0
  fi

  if [ "${SKIP_TRAIN_IF_DONE}" = "1" ] && [ -f "${OUTPUT_DIR}/trainer_state.json" ]; then
    log "Skip train stage because ${OUTPUT_DIR}/trainer_state.json already exists."
    return 0
  fi

  run_logged "${LOG_FILE}" "${TRAIN_SCRIPT}" "${MODEL}" "${TASK}" "${DATASET}" "${BATCH_SIZE}" "${EMB}"
}

eval_stage() {
  if [ "${RUN_EVAL}" != "1" ]; then
    log "Skip eval stage because RUN_EVAL=${RUN_EVAL}."
    return 0
  fi

  if [ "${SKIP_EVAL_IF_DONE}" = "1" ] && [ "${FORCE_EVAL}" != "1" ] && [ -f "${RESULTS_DIR}/summary.json" ]; then
    log "Skip eval stage because ${RESULTS_DIR}/summary.json already exists."
    return 0
  fi

  read -r -a eval_splits_array <<< "${EVAL_SPLITS}"
  local cmd=(
    "${PYTHON_BIN}" "eval/eval_webqsp_checkpoints.py"
    --output_dir "${OUTPUT_DIR}"
    --model_base "${MODEL_BASE}"
    --conv_mode "${CONV_MODE}"
    --pretrained_embedding_type "${EMB}"
    --cache_dir "${CACHE_DIR}"
    --splits "${eval_splits_array[@]}"
    --device "${EVAL_DEVICE}"
    --max_graph_paths "${EVAL_MAX_GRAPH_PATHS}"
    --max_memory_per_gpu "${EVAL_MAX_MEMORY_PER_GPU}"
    --results_dir "${RESULTS_DIR}"
  )

  if [ "${EVAL_ALL_SPLITS_FOR_ALL_CHECKPOINTS}" = "1" ]; then
    cmd+=(--all-splits-for-all-checkpoints)
  fi
  if [ "${EVAL_LIMIT}" -gt 0 ]; then
    cmd+=(--limit "${EVAL_LIMIT}")
  fi
  if [ "${NO_GRAPH_TOKEN}" = "1" ]; then
    cmd+=(--no_graph_token)
  fi

  run_logged "${LOG_FILE}" "${cmd[@]}"
}

extract_and_semantic_stage() {
  if [ "${RUN_EXTRACT}" != "1" ] && [ "${RUN_SEMANTIC}" != "1" ]; then
    log "Skip extract and semantic stages."
    return 0
  fi

  resolve_semantic_checkpoints
  read -r -a semantic_splits_array <<< "${SEMANTIC_SPLITS}"

  local checkpoint_name split details_jsonl extracted_json
  for checkpoint_name in "${RESOLVED_SEMANTIC_CHECKPOINTS[@]}"; do
    for split in "${semantic_splits_array[@]}"; do
      details_jsonl="${RESULTS_DIR}/${checkpoint_name}_${split}_details.jsonl"
      extracted_json="${RESULTS_DIR}/${checkpoint_name}_${split}_details_extracted.json"

      if [ ! -f "${details_jsonl}" ]; then
        local message="Missing inference details: ${details_jsonl}"
        if [ "${SEMANTIC_ALLOW_MISSING}" = "1" ]; then
          log "Skip ${checkpoint_name}/${split}: ${message}."
          continue
        fi
        echo "${message}" >&2
        exit 1
      fi

      if [ "${RUN_EXTRACT}" = "1" ]; then
        if [ "${FORCE_EXTRACT}" != "1" ] && [ -f "${extracted_json}" ]; then
          log "Skip extraction for ${checkpoint_name}/${split}: ${extracted_json} already exists."
        else
          run_logged "${LOG_FILE}" \
            "${PYTHON_BIN}" "checkpoints/extract_fields_to_json.py" \
            "${details_jsonl}" \
            --output "${extracted_json}"
        fi
      fi

      if [ "${RUN_SEMANTIC}" != "1" ]; then
        continue
      fi

      if [ ! -f "${extracted_json}" ]; then
        echo "Missing extracted JSON for semantic match: ${extracted_json}" >&2
        exit 1
      fi

      local semantic_cmd=(
        "${PYTHON_BIN}" "checkpoints/semantic_match_with_qwen.py"
        "${extracted_json}"
        --proxy-mode "${QWEN_PROXY_MODE}"
        --batch-size "${QWEN_BATCH_SIZE}"
        --max-workers "${QWEN_MAX_WORKERS}"
        --max-tokens "${QWEN_MAX_TOKENS}"
      )
      if [ "${FORCE_SEMANTIC}" = "1" ]; then
        semantic_cmd+=(--force)
      fi
      if [ "${QWEN_DISABLE_THINKING}" = "1" ]; then
        semantic_cmd+=(--disable-thinking)
      fi
      if [ "${QWEN_ONLY_TRUNCATED_PREDICTION_RAW}" = "1" ]; then
        semantic_cmd+=(--only-truncated-prediction-raw)
      fi
      if [ "${QWEN_ONLY_GENERATION_FAILED}" = "1" ]; then
        semantic_cmd+=(--only-generation-failed)
      fi
      if [ "${QWEN_NO_BACKUP}" = "1" ]; then
        semantic_cmd+=(--no-backup)
      fi
      if [ -n "${QWEN_MAX_ITEMS}" ]; then
        semantic_cmd+=(--max-items "${QWEN_MAX_ITEMS}")
      fi
      if [ -n "${QWEN_SLEEP_SECONDS}" ]; then
        semantic_cmd+=(--sleep-seconds "${QWEN_SLEEP_SECONDS}")
      fi

      run_logged "${LOG_FILE}" "${semantic_cmd[@]}"
    done
  done
}

main() {
  resolve_model_defaults
  MODEL_BASE="${MODEL_BASE:-${DEFAULT_MODEL_BASE}}"
  CONV_MODE="${CONV_MODE:-${DEFAULT_CONV_MODE}}"
  resolve_output_dir
  resolve_results_dir

  LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/pipeline_logs}"
  mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_eval_webqsp_$(date +%Y%m%d_%H%M%S).log}"

  if [ "${RUN_SEMANTIC}" = "1" ] && [ -z "${DASHSCOPE_API_KEY:-${BAILIAN_API_KEY:-}}" ]; then
    echo "RUN_SEMANTIC=1 but DASHSCOPE_API_KEY / BAILIAN_API_KEY is not set." >&2
    exit 1
  fi

  log "Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
  log "Results dir: ${RESULTS_DIR}" | tee -a "${LOG_FILE}"
  log "Model=${MODEL}, model_base=${MODEL_BASE}, conv_mode=${CONV_MODE}, emb=${EMB}" | tee -a "${LOG_FILE}"
  log "Eval splits=${EVAL_SPLITS}, semantic checkpoints=${SEMANTIC_CHECKPOINTS}, semantic splits=${SEMANTIC_SPLITS}" | tee -a "${LOG_FILE}"

  train_stage
  eval_stage
  extract_and_semantic_stage

  log "Pipeline finished. Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
}

main "$@"
