# Data and Artifact Policy

This repository does not include full WebQSP/CWQ data, Hugging Face dataset caches, base LLM weights, generated path embeddings, or trained adapter checkpoints.

## Stage 1: Retriever Input and Caches

The first stage uses `ml1996/webqsp` or `rmanluo/RoG-cwq` records with
`id`, `question`, `q_entity`, `a_entity`, `answer`, and `graph`.
`graph` is the candidate graph as a list of `[head, relation, tail]` triples.
It is not the recovered-path dataset used to train the adapter.

`retrieval/emb.py` turns these records into local entity/relation indices in
`data_files/<dataset>/processed/{train,val,test}.pkl`, and caches GTE-large
embeddings in `data_files/<dataset>/emb/gte-large-en-v1.5/{train,val,test}.pth`.
Each embedding record has `q_emb` (1 × 1024), `entity_embs` (N × 1024), and
`relation_embs` (R × 1024). Training derives binary triple labels from shortest
topic-to-answer paths. Inference exports scored triples, which are then passed
to path recovery.

## Stage 2: Merged KGQA Split Format

This is the boundary between path recovery and the graph knowledge adapter. The core
pipeline does not run entity linking or acquire a candidate graph from
Freebase; those inputs come from the prepared WebQSP/CWQ dataset.

The existing `webqsp_may_merged` and `llaga_cwq_may_merged` directories are
second-stage data. In the original configuration, their path embeddings use
GTE-ModernBERT (768 dimensions), independently of the first-stage GTE-large
encoder (1024 dimensions).

Each split file is a JSON list:

```json
[
  {
    "id": "WebQTrn-0",
    "question": "what did ...?",
    "answer": ["answer text"],
    "q_entity": ["anchor entity"],
    "a_entity": ["answer entity"],
    "reasoning_paths": [
      [
        ["head entity", "relation", "tail entity"]
      ]
    ]
  }
]
```

The training code is tolerant to several answer/entity field variants used by KGQA preprocessing, but new datasets should keep the fields above.

## Path Embedding Files

Path embedding files are PyTorch dictionaries keyed by sample id:

```text
{sample_id: List[List[Tensor(3, hidden_dim)]]}
```

Expected file names:

- `webqsp_paths_e5_train.pt`
- `webqsp_paths_e5_validation.pt`
- `webqsp_paths_e5_test.pt`
- `webqsp_paths_gte_modernbert_base_train.pt`
- `webqsp_paths_gte_modernbert_base_validation.pt`
- `webqsp_paths_gte_modernbert_base_test.pt`

For CWQ, replace the prefix `webqsp` with `cwq`.

Before training or evaluation, validate all three splits and their embedding
ID alignment:

```bash
python scripts/preflight_core_pipeline.py \
  --dataset webqsp \
  --data-dir data/webqsp_may_merged \
  --embedding-type Alibaba-NLP/gte-modernbert-base
```

## Local Directory Layout

Use these paths after cloning:

```text
data/webqsp_may_merged/
data/cwq_may_merged/
checkpoints/webqsp/
checkpoints/cwq/
llama/Meta-Llama-3-8B-Instruct/
llama/llama-2-7b-chat-hf/
```

These directories are ignored by git. Share large files through a model/data hosting service and document the download source in your release notes.

## Evaluation Alias Resources

The original exact-match/F1 evaluator uses the benchmark alias resources
`WQSP_aliase_data.json`, `CWQ_aliase_data31158.json`,
`ComplexWebQuestions_test_wans.json`, and `cwq_aname_dict.json`. Put them in
`data/aliases/`, or pass their directory with `eval.py --alias_dir ...` (the
`GRLPROF_ALIAS_DIR` environment variable is also supported). These files are
evaluation resources only; they are not inputs to entity linking in the core
pipeline.
