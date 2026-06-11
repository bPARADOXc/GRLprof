#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-webqsp}"
MODEL="${MODEL:-llama3}"
QUESTION="${QUESTION:-}"

FREEBASE_TRIPLES="${FREEBASE_TRIPLES:-}"
FREEBASE_DELIMITER="${FREEBASE_DELIMITER:-$'\t'}"
FREEBASE_CVT_ENTITIES="${FREEBASE_CVT_ENTITIES:-}"
FREEBASE_ENTITY_NAMES="${FREEBASE_ENTITY_NAMES:-}"
MAX_FREEBASE_TRIPLES="${MAX_FREEBASE_TRIPLES:--1}"

LINKER_CHECKPOINT="${LINKER_CHECKPOINT:-}"
ENTITY_CATALOG="${ENTITY_CATALOG:-}"
ALIAS_FILE="${ALIAS_FILE:-}"
START_ENTITIES="${START_ENTITIES:-}"
QUESTION_ENTITIES="${QUESTION_ENTITIES:-}"

RETRIEVER_CHECKPOINT="${RETRIEVER_CHECKPOINT:-}"
RETRIEVER_TEXT_ENCODER="${RETRIEVER_TEXT_ENCODER:-Alibaba-NLP/gte-large-en-v1.5}"
ENTITY_IDENTIFIERS="${ENTITY_IDENTIFIERS:-}"
SKIP_OPEN_SCORING="${SKIP_OPEN_SCORING:-0}"

PATH_EMBEDDING_TYPE="${PATH_EMBEDDING_TYPE:-Alibaba-NLP/gte-modernbert-base}"
PATH_EMBEDDING_MODEL="${PATH_EMBEDDING_MODEL:-}"
OPEN_OUTPUT_DIR="${OPEN_OUTPUT_DIR:-outputs/open_kgqa}"

if [[ -z "$FREEBASE_TRIPLES" ]]; then
  echo "FREEBASE_TRIPLES must point to a Freebase triples file." >&2
  exit 2
fi

if [[ ! -f "$FREEBASE_TRIPLES" ]]; then
  echo "FREEBASE_TRIPLES does not exist: $FREEBASE_TRIPLES" >&2
  exit 2
fi

if [[ -z "$ALIAS_FILE" ]]; then
  if [[ "$DATASET" == "webqsp" && -f "data/aliases/WQSP_aliase_data.json" ]]; then
    ALIAS_FILE="data/aliases/WQSP_aliase_data.json"
  elif [[ "$DATASET" == "cwq" && -f "data/aliases/CWQ_aliase_data.json" ]]; then
    ALIAS_FILE="data/aliases/CWQ_aliase_data.json"
  fi
fi

if [[ -z "$ENTITY_IDENTIFIERS" ]]; then
  default_identifiers="retrieval/data_files/${DATASET}/entity_identifiers.txt"
  if [[ -f "$default_identifiers" ]]; then
    ENTITY_IDENTIFIERS="$default_identifiers"
  fi
fi

cmd=(
  "$PYTHON_BIN" "eval/webqsp_cli_infer.py"
  --dataset "$DATASET"
  --model "$MODEL"
  --open_fallback
  --freebase_triples "$FREEBASE_TRIPLES"
  --freebase_delimiter "$FREEBASE_DELIMITER"
  --max_freebase_triples "$MAX_FREEBASE_TRIPLES"
  --retriever_text_encoder "$RETRIEVER_TEXT_ENCODER"
  --path_embedding_type "$PATH_EMBEDDING_TYPE"
  --open_output_dir "$OPEN_OUTPUT_DIR"
)

if [[ -n "$QUESTION" ]]; then
  cmd+=(--question "$QUESTION")
else
  cmd+=(--interactive)
fi

if [[ -n "$FREEBASE_CVT_ENTITIES" ]]; then
  cmd+=(--freebase_cvt_entities "$FREEBASE_CVT_ENTITIES")
fi
if [[ -n "$FREEBASE_ENTITY_NAMES" ]]; then
  cmd+=(--freebase_entity_names "$FREEBASE_ENTITY_NAMES")
fi
if [[ -n "$LINKER_CHECKPOINT" ]]; then
  cmd+=(--linker_checkpoint "$LINKER_CHECKPOINT")
fi
if [[ -n "$ENTITY_CATALOG" ]]; then
  IFS=":" read -r -a catalog_paths <<< "$ENTITY_CATALOG"
  cmd+=(--entity_catalog "${catalog_paths[@]}")
fi
if [[ -n "$ALIAS_FILE" ]]; then
  IFS=":" read -r -a alias_paths <<< "$ALIAS_FILE"
  cmd+=(--alias_file "${alias_paths[@]}")
fi
if [[ -n "$START_ENTITIES" ]]; then
  cmd+=(--start_entities "$START_ENTITIES")
fi
if [[ -n "$QUESTION_ENTITIES" ]]; then
  cmd+=(--question_entities "$QUESTION_ENTITIES")
fi
if [[ -n "$ENTITY_IDENTIFIERS" ]]; then
  cmd+=(--entity_identifiers "$ENTITY_IDENTIFIERS")
fi
if [[ -n "$PATH_EMBEDDING_MODEL" ]]; then
  cmd+=(--path_embedding_model "$PATH_EMBEDDING_MODEL")
fi

if [[ "$SKIP_OPEN_SCORING" == "1" ]]; then
  cmd+=(--skip_open_scoring)
else
  if [[ -z "$RETRIEVER_CHECKPOINT" ]]; then
    echo "RETRIEVER_CHECKPOINT is required unless SKIP_OPEN_SCORING=1." >&2
    exit 2
  fi
  cmd+=(--retriever_checkpoint "$RETRIEVER_CHECKPOINT")
fi

exec "${cmd[@]}" "$@"
