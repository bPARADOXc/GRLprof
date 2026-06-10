#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

max_len=${LLAGA_MAX_LEN:-2048}
sample_size=${LLAGA_SAMPLE_SIZE:--1}
grad_accum=${LLAGA_GRAD_ACCUM:-4}
bits=${LLAGA_BITS:-16}
report_to=${LLAGA_REPORT_TO:-none}
train_entry=${LLAGA_TRAIN_ENTRY:-train/train_mem.py}
dataloader_workers=${LLAGA_DATALOADER_NUM_WORKERS:-8}
resume_mm_projector_from=${LLAGA_RESUME_MM_PROJECTOR_FROM:-}
cache_dir=${LLAGA_CACHE_DIR:-./checkpoints/hf_cache}
output_root=${LLAGA_OUTPUT_ROOT:-./checkpoints}

model=${1:-"llama"}
task=${2:-"QA"}
dataset=${3-"webqsp"}
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
else
  echo "Unsupported model=${model}" >&2
  exit 1
fi

#用图的start和end
#原本max4096
#eval的逻辑改一下

echo "PREFIX:  ${prefix}"
echo "PRETRAINED_EMBEDDING_TYPE: ${emb}"
echo "EMBEDDING_LABEL: ${emb_label}"
echo "DATA_DIR: ${data_dir}"

if [ "${report_to}" = "wandb" ] && command -v wandb >/dev/null 2>&1; then
  wandb online
fi

cmd=(
  python "${train_entry}"
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
  --output_dir "${output_root%/}/${dataset}/${prefix}"
  --num_train_epochs 1
  --per_device_train_batch_size "${bs}"
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps "${grad_accum}"
  --evaluation_strategy no
  --save_strategy steps
  --save_steps 200
  --learning_rate 2e-3
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 20
  --tf32 True
  --model_max_length "${max_len}"
  --gradient_checkpointing True
  --lazy_preprocess True
  --dataloader_num_workers "${dataloader_workers}"
  --report_to "${report_to}"
  --word_embed_proj_dim 4096
  --use_hop "${use_hop}"
  --sample_neighbor_size "${sample_size}"
  --mm_projector_type "${projector_type}"
  --use_task "${task}"
  --use_dataset "${dataset}"
  --template "${template}"
)

if [ -n "${resume_mm_projector_from}" ]; then
  cmd+=(--resume_mm_projector_only_from "${resume_mm_projector_from}")
fi

printf '%q ' "${cmd[@]}"
echo
if [ "${LLAGA_DRY_RUN:-0}" = "1" ]; then
  echo "LLAGA_DRY_RUN=1, command not executed."
  exit 0
fi
"${cmd[@]}"
