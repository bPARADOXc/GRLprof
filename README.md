# GRLprof

GRLprof is a structure-aware KG-RAG system for complex knowledge-graph question answering. It retrieves a compact evidence subgraph from a KG, recovers high-scoring reasoning paths, vectorizes those paths, and injects them as soft prompts into a frozen local Llama-family model.

The code in this repository is organized for release and reuse. Some internal module and checkpoint symbols still use `llaga` names because the adapter model type and tokenizer conventions are inherited from the LLaGA codebase; user-facing scripts, docs, and output prefixes use `grlprof`.

## Method Overview

Pipeline:

```text
question -> anchor entity and hop estimate -> candidate subgraph
-> edge-level triple scoring -> compact subgraph selection
-> high-score path recovery -> path graph vectorization
-> knowledge adapter soft prompts -> frozen Llama answer generation
```

Core ideas:

- Structure-aware edge retriever: scores candidate triples with question/entity/relation semantics plus directional distance features.
- Budgeted subgraph selection: keeps evidence connected to anchor entities and adds bridge edges when needed.
- Path recovery: converts high-score subgraphs into de-duplicated reasoning paths.
- Knowledge adapter: projects path structure into the LLM embedding space as soft prompts instead of serializing all triples as long text.
- Frozen local LLM: trains the retriever and graph adapter while keeping Llama2-7B or Llama3-8B frozen.

See [docs/method.md](docs/method.md) for the full method description.

## Repository Layout

```text
GRLprof/
  model/                 Llama/OPT graph-adapter model code
  train/                 Adapter training entrypoints and trainer
  eval/                  Checkpoint, text-path, and CLI evaluation
  retrieval/             KG subgraph retriever training/inference code
  reasoning/             Optional text-based KGQA reasoning baseline
  scripts/               Data conversion, embedding, training pipelines
  tools/                 Evaluation post-processing utilities
  configs/               DeepSpeed and example runtime configs
  requirements/          Dependency sets by stage
  data/                  Local datasets and generated path embeddings
  checkpoints/           Local adapter checkpoints and eval outputs
  docs/                  Method, data, and release notes
```

Large datasets, model weights, generated embeddings, checkpoint binaries, and logs are intentionally excluded from git.

## Installation

Use separate environments for the adapter, retriever, and path-embedding stages. The adapter code follows the original LLaGA dependency range, while ModernBERT path embeddings require a newer Hugging Face stack.

Create the adapter environment:

```bash
conda create -n grlprof python=3.10 -y
conda activate grlprof
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Create the retriever environment:

```bash
conda create -n grlprof-retriever python=3.10 -y
conda activate grlprof-retriever
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/retriever.txt
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

Create the reasoning-path embedding environment:

```bash
conda create -n grlprof-embed python=3.10 -y
conda activate grlprof-embed
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/embedding.txt
```

Optional text-reasoning baseline environment:

```bash
conda create -n grlprof-reasoning python=3.10 -y
conda activate grlprof-reasoning
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/reasoning.txt
```

## Data Preparation

Expected merged KGQA data layout:

```text
data/webqsp_may_merged/
  train.json
  validation.json
  test.json
  webqsp_paths_gte_modernbert_base_train.pt
  webqsp_paths_gte_modernbert_base_validation.pt
  webqsp_paths_gte_modernbert_base_test.pt

data/cwq_may_merged/
  train.json
  validation.json
  test.json
  cwq_paths_gte_modernbert_base_train.pt
  cwq_paths_gte_modernbert_base_validation.pt
  cwq_paths_gte_modernbert_base_test.pt
```

Run path recovery from scored retrieval outputs:

```bash
python scripts/expand_webqsp_scored_triples.py \
  --dataset webqsp \
  --input-root retrieval/webqsp_<run-id> \
  --output-root data/webqsp_may_merged
```

Pre-compute reasoning-path embeddings:

```bash
python scripts/process_webqsp_e5.py \
  --dataset webqsp \
  --base-dir data/webqsp_may_merged \
  --embedding-type Alibaba-NLP/gte-modernbert-base
```

See [docs/data.md](docs/data.md) for required fields and artifact policy.

## Retriever Stage

From `retrieval/`:

```bash
cd retrieval
python emb.py -d webqsp
python train.py -d webqsp
python inference.py -p webqsp_<run-id>/cpt.pth
python eval.py -d webqsp -p webqsp_<run-id>/retrieval_result.pth
```

The retriever writes split-level scored triples that can be expanded into GRLprof reasoning paths with `scripts/expand_webqsp_scored_triples.py`.

## Adapter Training

Single-machine DeepSpeed example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
LLAGA_MAX_LEN=2600 \
LLAGA_GRAD_ACCUM=4 \
LLAGA_DEEPSPEED_CONFIG=./configs/deepspeed/zero3.json \
./scripts/train_deepspeed.sh \
  llama3 QA webqsp 1 Alibaba-NLP/gte-modernbert-base data/webqsp_may_merged
```

Check the command expansion without launching DeepSpeed:

```bash
LLAGA_DRY_RUN=1 ./scripts/train_deepspeed.sh \
  llama3 QA webqsp 1 Alibaba-NLP/gte-modernbert-base data/webqsp_may_merged
```

Useful environment variables:

- `LLAGA_LLAMA3_MODEL_BASE`: local or Hugging Face Llama3 base model path.
- `LLAGA_LLAMA2_MODEL_BASE`: local or Hugging Face Llama2 base model path.
- `LLAGA_OUTPUT_ROOT`: checkpoint root, default `./checkpoints`.
- `LLAGA_REPORT_TO`: set `wandb` to enable Weights & Biases, default `none`.
- `LLAGA_MAX_GRAPH_PATHS`: maximum soft-prompt path tokens per sample.
- `LLAGA_GRAPH_PATH_BATCH`: path encoder micro-batch size.

## Evaluation

Evaluate graph-soft-prompt checkpoints:

```bash
python eval/eval_webqsp_checkpoints.py \
  --output_dir checkpoints/webqsp/grlprof-llama-3-8b-instruct-gte-modernbert-base-webqsp-projector-rag-path32-lr2e-3-warmup0.03-epoch1-bs2.2.1 \
  --model_base meta-llama/Meta-Llama-3-8B-Instruct \
  --conv_mode llama_3 \
  --dataset webqsp \
  --data_dir data/webqsp_may_merged \
  --pretrained_embedding_type Alibaba-NLP/gte-modernbert-base \
  --splits validation test \
  --checkpoints final
```

Run the text-path baseline:

```bash
python eval/eval_text_paths_checkpoints.py \
  --dataset webqsp \
  --data_dir data/webqsp_may_merged \
  --output_dir checkpoints/webqsp/text_paths_llama3 \
  --model_base meta-llama/Meta-Llama-3-8B-Instruct \
  --conv_mode llama_3 \
  --splits test
```

Interactive single-question CLI requires adapter paths via environment variables:

```bash
export GRLPROF_WEBQSP_LLAMA3_ADAPTER=checkpoints/webqsp/grlprof-llama3
python eval/webqsp_cli_infer.py --dataset webqsp --model llama3 --interactive
```

Open-question inference can be enabled for questions not found in the local
dataset cache. This runs entity prediction, Freebase rough subgraph sampling,
triple scoring, path recovery, dynamic path embedding, and then the GRLprof
adapter:

```bash
export GRLPROF_WEBQSP_LLAMA3_ADAPTER=checkpoints/webqsp/grlprof-llama3
FREEBASE_TRIPLES=/path/to/freebase.tsv \
LINKER_CHECKPOINT=checkpoints/entity_linker/webqsp_roberta/best \
RETRIEVER_CHECKPOINT=checkpoints/retriever/webqsp/cpt.pth \
QUESTION="where was Barack Obama born" \
./scripts/run_open_kgqa_cli.sh --show_all_matched
```

For debugging the retrieval bridge without loading the GRLprof adapter, run:

```bash
python scripts/run_freebase_pipeline.py \
  --question "where was Barack Obama born" \
  --start-entities /m/02mjmr \
  --question-entities /m/02mjmr \
  --freebase-triples /path/to/freebase.tsv \
  --skip-scoring \
  --output-json outputs/freebase_pipeline_obama.json
```

Entity linker training entrypoints are:

```bash
python scripts/train_elq_start_linker.py \
  --train-file data/webqsp_may_merged/train.json \
  --valid-file data/webqsp_may_merged/validation.json \
  --alias-file data/aliases/WQSP_aliase_data.json \
  --model-name-or-path roberta-base \
  --output-dir checkpoints/entity_linker/webqsp_roberta
```

## Release Notes

- The repository contains code and small config files only.
- Put local base LLMs under `llama/` or set `LLAGA_LLAMA*_MODEL_BASE`.
- Put generated merged datasets under `data/`.
- Put adapter checkpoints under `checkpoints/`.
- Do not commit generated `.pt`, `.pth`, `.bin`, `.safetensors`, `.jsonl`, or raw dataset cache files.

## Upstream Acknowledgements

This project derives adapter-side implementation details from LLaGA and retriever-side components from SubgraphRAG. License copies are preserved in [licenses/](licenses/), and provenance details are in [NOTICE.md](NOTICE.md).
