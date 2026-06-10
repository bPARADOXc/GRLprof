#!/bin/bash
set -euo pipefail

# Run train -> inference -> extract -> eval.py pipelines in order:
#   1. CWQ    + llama3
#   2. WebQSP + llama2
#   3. WebQSP + llama3
#
# Defaults follow the requested command shape:
#   LLAGA_DEEPSPEED_INCLUDE="localhost:1,2"
#   LLAGA_MAX_LEN=2600
#   LLAGA_GRAD_ACCUM=2
#   LLAGA_DEEPSPEED_CONFIG=./scripts/zero3.json
#   EMB=Alibaba-NLP/gte-modernbert-base
#   EVAL_TORCH_DTYPE=auto
#   EVAL_CUDA_VISIBLE_DEVICES=1,2
#
# Optional controls:
#   START_AT=1..3          start from a later run; use START_AT=2 if CWQ+llama3 is already done
#   ONLY_STEP=1..3         run only one step
#   RUN_EXPAND_CWQ=1       rebuild CWQ merged-data JSON files
#   RUN_CWQ_EMBED=1        rebuild CWQ graph embedding .pt files
#   CHECK_CWQ_INPUTS=0     skip existing CWQ json/pt checks
#   RUN_TRAIN=0            skip training and use OUTPUT_DIR if provided to child pipeline
#   OUTPUT_BASE=./checkpoints
#   CWQ_DATA_SUBDIR=cwq_may_merged

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-./scripts/train_infer_extract_eval_pipeline.sh}"
EXPAND_SCRIPT="${EXPAND_SCRIPT:-scripts/expand_webqsp_scored_triples.py}"
EMBED_SCRIPT="${EMBED_SCRIPT:-scripts/process_webqsp_e5.py}"
GTE_PYTHON_BIN="${GTE_PYTHON_BIN:-python}"

export LLAGA_DEEPSPEED_INCLUDE="${LLAGA_DEEPSPEED_INCLUDE:-localhost:1,2}"
export LLAGA_MAX_LEN="${LLAGA_MAX_LEN:-2600}"
export LLAGA_GRAD_ACCUM="${LLAGA_GRAD_ACCUM:-2}"
export LLAGA_DEEPSPEED_CONFIG="${LLAGA_DEEPSPEED_CONFIG:-./configs/deepspeed/zero3.json}"
export EMB="${EMB:-Alibaba-NLP/gte-modernbert-base}"
export OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"
export EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-auto}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1,2}"
HF_CACHE_ROOT="${LLAGA_HF_CACHE_ROOT:-${ROOT_DIR}/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}}"

CWQ_EXPAND_INPUT_ROOT="${CWQ_EXPAND_INPUT_ROOT:-${ROOT_DIR}/retrieval/runs/cwq}"
CWQ_DATA_SUBDIR="${CWQ_DATA_SUBDIR:-cwq_may_merged}"
CWQ_DATA_DIR="${CWQ_DATA_DIR:-${CWQ_EXPAND_INPUT_ROOT%/}/${CWQ_DATA_SUBDIR}}"
WEBQSP_DATA_DIR="${WEBQSP_DATA_DIR:-${ROOT_DIR}/data/webqsp_may_merged}"
RUN_EXPAND_CWQ="${RUN_EXPAND_CWQ:-0}"
RUN_CWQ_EMBED="${RUN_CWQ_EMBED:-0}"
CHECK_CWQ_INPUTS="${CHECK_CWQ_INPUTS:-1}"
FORCE_CWQ_EMBED="${FORCE_CWQ_EMBED:-${RUN_EXPAND_CWQ}}"
CWQ_EXPAND_OUTPUT_ROOT="${CWQ_EXPAND_OUTPUT_ROOT:-${CWQ_DATA_DIR}}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-512}"

START_AT="${START_AT:-1}"
ONLY_STEP="${ONLY_STEP:-}"

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

run_cwq_expand() {
  if [ "${RUN_EXPAND_CWQ}" != "1" ]; then
    log "Skip CWQ data creation because RUN_EXPAND_CWQ=${RUN_EXPAND_CWQ}"
    return
  fi

  if [ "$(readlink -f "${CWQ_EXPAND_INPUT_ROOT}")" = "$(readlink -f "${CWQ_EXPAND_OUTPUT_ROOT}")" ]; then
    echo "CWQ_EXPAND_INPUT_ROOT and CWQ_EXPAND_OUTPUT_ROOT point to the same directory." >&2
    echo "Use the raw retrieve directory as input and a child directory as output." >&2
    exit 1
  fi

  log "Create CWQ merged data: input=${CWQ_EXPAND_INPUT_ROOT}, output=${CWQ_EXPAND_OUTPUT_ROOT}"
  "${GTE_PYTHON_BIN}" "${EXPAND_SCRIPT}" \
    --dataset cwq \
    --input-root "${CWQ_EXPAND_INPUT_ROOT}" \
    --output-root "${CWQ_EXPAND_OUTPUT_ROOT}"
}

embedding_prefix_for_cwq() {
  case "${EMB}" in
    E5-base|e5-base|e5|intfloat/multilingual-e5-base)
      echo "cwq_paths_e5"
      ;;
    Alibaba-NLP/gte-modernbert-base|gte-modernbert-base|GTE-modernbert-base|gte-modernbert|GTE-modernbert)
      echo "cwq_paths_gte_modernbert_base"
      ;;
    *)
      echo "Unsupported EMB=${EMB}. Use E5-base or Alibaba-NLP/gte-modernbert-base." >&2
      exit 1
      ;;
  esac
}

cwq_embedding_files_exist() {
  local prefix="$1"
  local split
  for split in train validation test; do
    [ -s "${CWQ_DATA_DIR}/${prefix}_${split}.pt" ] || return 1
  done
  return 0
}

validate_cwq_existing_inputs() {
  local prefix
  local split
  local missing=0
  prefix="$(embedding_prefix_for_cwq)"

  for split in train validation test; do
    if [ ! -s "${CWQ_DATA_DIR}/${split}.json" ]; then
      echo "Missing CWQ split file: ${CWQ_DATA_DIR}/${split}.json" >&2
      missing=1
    fi
    if [ ! -s "${CWQ_DATA_DIR}/${prefix}_${split}.pt" ]; then
      echo "Missing CWQ embedding file: ${CWQ_DATA_DIR}/${prefix}_${split}.pt" >&2
      missing=1
    fi
  done

  if [ "${missing}" != "0" ]; then
    echo "Existing CWQ inputs are incomplete. Set RUN_EXPAND_CWQ=1 and/or RUN_CWQ_EMBED=1 to rebuild them." >&2
    exit 1
  fi

  log "Use existing CWQ json/pt files from ${CWQ_DATA_DIR}"
}

validate_cwq_reasoning_paths() {
  "${GTE_PYTHON_BIN}" -c "import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); total=0; nonempty=0
for split in ('train','validation','test'):
    data=json.load(open(root / f'{split}.json'))
    split_nonempty=sum(1 for item in data if item.get('reasoning_paths'))
    print(f'[cwq data] {split}: records={len(data)}, nonempty_reasoning_paths={split_nonempty}')
    total += len(data)
    nonempty += split_nonempty
if total == 0 or nonempty == 0:
    sys.exit('CWQ merged data has no reasoning_paths. Check CWQ_EXPAND_INPUT_ROOT; it must be the raw retrieve directory.')" \
    "${CWQ_DATA_DIR}"
}

run_cwq_embeddings() {
  if [ "${RUN_CWQ_EMBED}" != "1" ]; then
    log "Skip CWQ graph embedding creation because RUN_CWQ_EMBED=${RUN_CWQ_EMBED}"
    return
  fi

  validate_cwq_reasoning_paths

  local prefix
  prefix="$(embedding_prefix_for_cwq)"
  if [ "${FORCE_CWQ_EMBED}" != "1" ] && cwq_embedding_files_exist "${prefix}"; then
    log "Skip CWQ graph embedding creation because ${prefix}_{train,validation,test}.pt already exist"
    return
  fi

  log "Create CWQ graph embeddings: data_dir=${CWQ_DATA_DIR}, emb=${EMB}, HF_HOME=${HF_HOME}"
  "${GTE_PYTHON_BIN}" "${EMBED_SCRIPT}" \
    --dataset cwq \
    --base-dir "${CWQ_DATA_DIR}" \
    --embedding-type "${EMB}" \
    --batch-size "${EMBED_BATCH_SIZE}" \
    --max-length "${EMBED_MAX_LENGTH}"

  if ! cwq_embedding_files_exist "${prefix}"; then
    echo "CWQ graph embedding generation finished without required ${prefix}_{train,validation,test}.pt files." >&2
    exit 1
  fi
}

step_selected() {
  local step="$1"
  if [ -n "${ONLY_STEP}" ] && [ "${step}" != "${ONLY_STEP}" ]; then
    return 1
  fi
  if [ "${step}" -lt "${START_AT}" ]; then
    return 1
  fi
  return 0
}

run_step() {
  local step="$1"
  local model="$2"
  local dataset="$3"
  local data_dir="$4"

  if ! step_selected "${step}"; then
    if [ -n "${ONLY_STEP}" ] && [ "${step}" != "${ONLY_STEP}" ]; then
      log "Skip step ${step}: ${dataset}/${model} because ONLY_STEP=${ONLY_STEP}"
    elif [ "${step}" -lt "${START_AT}" ]; then
      log "Skip step ${step}: ${dataset}/${model} because START_AT=${START_AT}"
    fi
    return
  fi

  log "Start step ${step}: dataset=${dataset}, model=${model}, data_dir=${data_dir}, emb=${EMB}"
  "${PIPELINE_SCRIPT}" "${model}" "${dataset}" "${data_dir}" 1 "${EMB}"
  log "Finished step ${step}: dataset=${dataset}, model=${model}"
}


if step_selected 1; then
  if [ "${RUN_EXPAND_CWQ}" = "1" ]; then
    run_cwq_expand
  fi
  if [ "${RUN_CWQ_EMBED}" = "1" ]; then
    run_cwq_embeddings
  elif [ "${CHECK_CWQ_INPUTS}" = "1" ]; then
    validate_cwq_existing_inputs
  fi
fi

run_step 1 llama3 cwq    "${CWQ_DATA_DIR}"
run_step 2 llama  webqsp "${WEBQSP_DATA_DIR}"
run_step 3 llama3 webqsp "${WEBQSP_DATA_DIR}"

log "All requested pipelines finished."
