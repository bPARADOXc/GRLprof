#!/usr/bin/env bash
# Usage:
# DASHSCOPE_API_KEY=... bash tools/run_qwen_semantic.sh \
#   checkpoints/webqsp/experiment/eval_results/webqsp_checkpoint_eval/final_validation_details.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_PATH="${1:-}"
OUTPUT_JSON="${2:-}"

if [ -z "${INPUT_PATH}" ]; then
  echo "Usage: $0 <details.jsonl|extracted.json> [output.json]" >&2
  exit 2
fi

if [ ! -f "${INPUT_PATH}" ]; then
  echo "Input file not found: ${INPUT_PATH}" >&2
  exit 1
fi

case "${INPUT_PATH}" in
  *.jsonl)
    if [ -z "${OUTPUT_JSON}" ]; then
      OUTPUT_JSON="${INPUT_PATH%.jsonl}_extracted.json"
    fi

    if [ "${FORCE_EXTRACT:-0}" = "1" ] || [ ! -f "${OUTPUT_JSON}" ]; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_fields_to_json.py" \
        "${INPUT_PATH}" \
        --output "${OUTPUT_JSON}"
    else
      echo "Use existing extracted JSON: ${OUTPUT_JSON}"
    fi
    ;;
  *.json)
    OUTPUT_JSON="${INPUT_PATH}"
    ;;
  *)
    echo "Input must be a .jsonl details file or an extracted .json file: ${INPUT_PATH}" >&2
    exit 1
    ;;
esac

SEMANTIC_CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/semantic_match_with_qwen.py"
  "${OUTPUT_JSON}"
  --proxy-mode "${QWEN_PROXY_MODE:-direct}"
  --batch-size "${QWEN_BATCH_SIZE:-5}"
  --max-workers "${QWEN_MAX_WORKERS:-8}"
  --max-tokens "${QWEN_MAX_TOKENS:-768}"
)

if [ "${QWEN_FORCE:-0}" = "1" ]; then
  SEMANTIC_CMD+=(--force)
fi
if [ "${QWEN_DISABLE_THINKING:-0}" = "1" ]; then
  SEMANTIC_CMD+=(--disable-thinking)
fi
if [ "${QWEN_ONLY_TRUNCATED_PREDICTION_RAW:-0}" = "1" ]; then
  SEMANTIC_CMD+=(--only-truncated-prediction-raw)
fi
if [ "${QWEN_ONLY_GENERATION_FAILED:-0}" = "1" ]; then
  SEMANTIC_CMD+=(--only-generation-failed)
fi
if [ "${QWEN_NO_BACKUP:-0}" = "1" ]; then
  SEMANTIC_CMD+=(--no-backup)
fi
if [ -n "${QWEN_MAX_ITEMS:-}" ]; then
  SEMANTIC_CMD+=(--max-items "${QWEN_MAX_ITEMS}")
fi
if [ -n "${QWEN_SLEEP_SECONDS:-}" ]; then
  SEMANTIC_CMD+=(--sleep-seconds "${QWEN_SLEEP_SECONDS}")
fi

echo "Semantic input: ${OUTPUT_JSON}"
"${SEMANTIC_CMD[@]}"
