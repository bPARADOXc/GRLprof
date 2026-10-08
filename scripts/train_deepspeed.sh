#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Example:
# CUDA_VISIBLE_DEVICES=0,1,2 LLAGA_MAX_LEN=2600 LLAGA_GRAD_ACCUM=4 \
#   LLAGA_DEEPSPEED_CONFIG=./configs/deepspeed/zero3.json \
#   ./scripts/train_deepspeed.sh llama3 QA webqsp 1 E5-base data/webqsp_may_merged
#CUDA_VISIBLE_DEVICES=0,1,2 LLAGA_SKIP_INITIAL_EVAL=1 LLAGA_MAX_LEN=2600 LLAGA_GRAD_ACCUM=4 LLAGA_DEEPSPEED_CONFIG=./scripts/zero3.json ./scripts/train_deepspeed.sh llama QA webqsp 1 E5-base
#LLAGA_DEEPSPEED_INCLUDE="localhost:1,2"  LLAGA_MAX_LEN=2600 LLAGA_GRAD_ACCUM=2 LLAGA_DEEPSPEED_CONFIG=./scripts/zero3.json ./scripts/train_deepspeed.sh llama3 QA webqsp 1 E5-base
max_len=${LLAGA_MAX_LEN:-2600}
sample_size=${LLAGA_SAMPLE_SIZE:--1}
grad_accum=${LLAGA_GRAD_ACCUM:-2}
bits=${LLAGA_BITS:-16}
report_to=${LLAGA_REPORT_TO:-none}
deepspeed_config=${LLAGA_DEEPSPEED_CONFIG:-./configs/deepspeed/zero3.json}
master_port=${LLAGA_MASTER_PORT:-61000}
train_entry=${LLAGA_TRAIN_ENTRY:-train/train_mem.py}
dataloader_workers=${LLAGA_DATALOADER_NUM_WORKERS:-8}
epochs=${LLAGA_EPOCHS:-1}
max_steps=${LLAGA_MAX_STEPS:--1}
max_samples_per_split=${LLAGA_MAX_SAMPLES_PER_SPLIT:--1}
learning_rate=${LLAGA_LR:-2e-3}
weight_decay=${LLAGA_WEIGHT_DECAY:-0.}
warmup_ratio=${LLAGA_WARMUP_RATIO:-0.03}
lr_scheduler_type=${LLAGA_LR_SCHEDULER_TYPE:-cosine}
save_steps=${LLAGA_SAVE_STEPS:-2000}
save_total_limit=${LLAGA_SAVE_TOTAL_LIMIT:-}
logging_steps=${LLAGA_LOGGING_STEPS:-2}
max_grad_norm=${LLAGA_MAX_GRAD_NORM:-1.0}
seed=${LLAGA_SEED:-42}
output_suffix=${LLAGA_OUTPUT_SUFFIX:-rag-path32-lr2e-3-warmup0.03-epoch1-bs2.2.1}
output_root=${LLAGA_OUTPUT_ROOT:-./checkpoints}
explicit_output_dir=${LLAGA_OUTPUT_DIR:-}
resume_mm_projector_from=${LLAGA_RESUME_MM_PROJECTOR_FROM:-}
word_embed_proj_dim=${LLAGA_WORD_EMBED_PROJ_DIM:-4096}
cache_dir=${LLAGA_CACHE_DIR:-./checkpoints/hf_cache}
triton_cache_dir=${TRITON_CACHE_DIR:-${cache_dir%/}/triton}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.9}"
export TRITON_CACHE_DIR="${triton_cache_dir}"
export LLAGA_MAX_GRAPH_PATHS="${LLAGA_MAX_GRAPH_PATHS:-32}"
export LLAGA_GRAPH_PATH_BATCH="${LLAGA_GRAPH_PATH_BATCH:-32}"
export LLAGA_GRAPH_ENCODER_CHECKPOINT="${LLAGA_GRAPH_ENCODER_CHECKPOINT:-0}"

model=${1:-"llama"}
task=${2:-"QA"}
dataset=${3:-"webqsp"}
bs=${4:-1}
emb=${5:-"E5-base"}
data_dir=${6:-${LLAGA_DATA_DIR:-}}
if [ -z "${data_dir}" ]; then
  case "${dataset}" in
    cwq)
      data_dir="data/cwq_may_merged"
      ;;
    webqsp)
      data_dir="data/webqsp_may_merged"
      ;;
    *)
      data_dir="data/${dataset}_may_merged"
      ;;
  esac
fi
case "${emb}" in
  E5-base|e5-base|e5|intfloat/multilingual-e5-base)
    emb_label="E5-base"
    ;;
  Alibaba-NLP/gte-modernbert-base|gte-modernbert-base|GTE-modernbert-base|gte-modernbert|GTE-modernbert)
    emb_label="gte-modernbert-base"
    ;;
  *)
    emb_label="${emb//\//_}"
    ;;
esac

if [ "${LLAGA_DISABLE_FLASH_ATTN:-0}" = "1" ]; then
  train_entry="train/train.py"
fi


if [ ${model} = "vicuna" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=grlprof-vicuna-7b-${emb_label}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  use_hop=2
  template="ND"
  projector_type="2-layer-mlp"
  prefix=grlprof-vicuna-7b-${emb_label}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=grlprof-vicuna-7b-${emb_label}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop_2layer" ]; then
  use_hop=4
  template="HO"
  projector_type="2-layer-mlp"
  prefix=grlprof-vicuna-7b-${emb_label}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  use_hop=0
  template="QA"
  projector_type="mlp"
  prefix=grlprof-llama-2-7b-chat-${emb_label}-${dataset}-projector
  default_llama2_model_base="./llama/llama-2-7b-chat-hf"
  if [ ! -d "${default_llama2_model_base}" ]; then
    default_llama2_model_base="meta-llama/Llama-2-7b-chat-hf"
  fi
  model_base=${LLAGA_LLAMA2_MODEL_BASE:-${default_llama2_model_base}}
  mode="llaga_llama_2"
elif [ ${model} = "llama3" ]; then
  use_hop=0
  template="QA"
  projector_type="mlp"
  prefix=grlprof-llama-3-8b-instruct-${emb_label}-${dataset}-projector
  default_llama3_model_base="./llama/Meta-Llama-3-8B-Instruct"
  if [ ! -d "${default_llama3_model_base}" ]; then
    default_llama3_model_base="meta-llama/Meta-Llama-3-8B-Instruct"
  fi
  model_base=${LLAGA_LLAMA3_MODEL_BASE:-${default_llama3_model_base}}
  mode="llama_3"
elif [ ${model} = "llama_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=grlprof-llama-2-7b-hf-${emb_label}-${use_hop}-hop-token-${projector_type}-projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "opt_2.7b" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=grlprof-opt-2.7b-${emb_label}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
elif [ ${model} = "opt_2.7b_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=grlprof-opt-2.7b-${emb_label}-${use_hop}-hop-token-${projector_type}-only-train-pretrain
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
fi

if [ -n "${explicit_output_dir}" ]; then
  output_dir="${explicit_output_dir}"
else
  output_dir="${output_root%/}/${dataset}/${prefix}"
  if [ -n "${output_suffix}" ]; then
    output_dir="${output_dir}-${output_suffix}"
  fi
fi


echo "PREFIX:  ${prefix}"
echo "PRETRAINED_EMBEDDING_TYPE: ${emb}"
echo "EMBEDDING_LABEL: ${emb_label}"
echo "DATA_DIR: ${data_dir}"
echo "OUTPUT_DIR: ${output_dir}"
echo "LLAGA_MAX_GRAPH_PATHS: ${LLAGA_MAX_GRAPH_PATHS}"
echo "LLAGA_GRAPH_PATH_BATCH: ${LLAGA_GRAPH_PATH_BATCH}"
echo "LLAGA_GRAPH_ENCODER_CHECKPOINT: ${LLAGA_GRAPH_ENCODER_CHECKPOINT}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
if [ "${report_to}" = "wandb" ] && command -v wandb >/dev/null 2>&1; then
  wandb online
fi

cmd=(deepspeed)
if [ -n "${LLAGA_DEEPSPEED_INCLUDE:-}" ]; then
  # DeepSpeed must enumerate physical slots before applying --include.
  cmd=(env -u CUDA_VISIBLE_DEVICES deepspeed)
  cmd+=(--include "${LLAGA_DEEPSPEED_INCLUDE}")
elif [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  # DeepSpeed 0.14 otherwise mistakes physical GPU2 for a nonexistent slot
  # after torch has already reduced the visible device count to one.
  cmd=(env -u CUDA_VISIBLE_DEVICES deepspeed --include "localhost:${CUDA_VISIBLE_DEVICES}")
fi
cmd+=(
  --master_port "${master_port}"
  "${train_entry}"
  --deepspeed "${deepspeed_config}"
  --model_name_or_path "${model_base}"
  --version "${mode}"
  --cache_dir "${cache_dir}"
  --pretrained_embedding_type "${emb}"
  --data_dir "${data_dir}"
  --tune_mm_mlp_adapter True
  --mm_use_graph_start_end True
  --mm_use_graph_patch_token False
  --bf16 True
  --bits "${bits}"
  --output_dir "${output_dir}"
  --num_train_epochs "${epochs}"
  --max_steps "${max_steps}"
  --max_samples_per_split "${max_samples_per_split}"
  --per_device_train_batch_size "${bs}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${grad_accum}"
  --evaluation_strategy no
  --save_strategy steps
  --save_steps "${save_steps}"
  --learning_rate "${learning_rate}"
  --weight_decay "${weight_decay}"
  --warmup_ratio "${warmup_ratio}"
  --lr_scheduler_type "${lr_scheduler_type}"
  --logging_steps "${logging_steps}"
  --max_grad_norm "${max_grad_norm}"
  --seed "${seed}"
  --tf32 True
  --model_max_length "${max_len}"
  --gradient_checkpointing True
  --lazy_preprocess True
  --dataloader_num_workers "${dataloader_workers}"
  --word_embed_proj_dim "${word_embed_proj_dim}"
  --report_to "${report_to}"
  --use_hop "${use_hop}"
  --sample_neighbor_size "${sample_size}"
  --mm_projector_type "${projector_type}"
  --use_task "${task}"
  --use_dataset "${dataset}"
  --template "${template}"
)

if [ -n "${save_total_limit}" ]; then
  cmd+=(--save_total_limit "${save_total_limit}")
fi

if [ -n "${resume_mm_projector_from}" ]; then
  cmd+=(--resume_mm_projector_only_from "${resume_mm_projector_from}")
fi

printf '%q ' "${cmd[@]}"
echo
if [ "${LLAGA_DRY_RUN:-0}" = "1" ]; then
  echo "LLAGA_DRY_RUN=1, command not executed."
  exit 0
fi
mkdir -p "${TRITON_CACHE_DIR}"
"${cmd[@]}"
