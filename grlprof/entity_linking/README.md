# KG-ELQ-Start

This package adds an isolated training path for question entity linking and
start-entity selection. It does not modify the existing retriever, path
recovery, or GRLprof adapter training code.

The first version implements:

- ELQ-style span enumeration over a question.
- Mention detection from span representations.
- Biencoder entity linking with dot-product ranking.
- Optional cross-encoder entity reranker training for top-K candidates.
- KG-aware start-entity ranking over linked candidates.
- Optional relation/path auxiliary prediction from `reasoning_paths` or
  `inferential_chain`.

Example:

```bash
python scripts/train_elq_start_linker.py \
  --train-file data/webqsp_may_merged/train.json \
  --valid-file data/webqsp_may_merged/validation.json \
  --alias-file data/aliases/WQSP_aliase_data.json \
  --model-name-or-path roberta-base \
  --output-dir checkpoints/entity_linker/webqsp_roberta
```

Cross-encoder reranker:

```bash
python scripts/train_entity_reranker.py \
  --train-file data/webqsp_may_merged/train.json \
  --valid-file data/webqsp_may_merged/validation.json \
  --alias-file data/aliases/WQSP_aliase_data.json \
  --model-name-or-path roberta-base \
  --output-dir checkpoints/entity_reranker/webqsp_roberta
```

Use a local `--model-name-or-path` when network access is unavailable. Current
merged data contains entity names but not Freebase MIDs, so this code can train
with entity names as weak IDs. When MID-level WebQSP/SPARQL annotations are
available, provide samples with `mentions[].entity_id` or `q_entity_id_list`;
the same training code will use those IDs directly.

Accepted entity catalog formats:

- JSONL records with `entity_id`, `name`, `aliases`, `types`,
  `description`, `relations`, `degree`, and `popularity`.
- JSON list or dict of the same records.
- Alias dicts such as `data/aliases/WQSP_aliase_data.json`, mapping entity name
  to aliases.

Checkpoint layout:

```text
output-dir/
  best/
    pytorch_model.bin
    tokenizer/
    entity_catalog.jsonl
    relation_vocab.json
    training_config.json
  final/
  metrics_history.json
```

The saved catalog and model are intended to be reused by a later inference
stage that emits:

```json
{
  "linked_entities": [],
  "start_entities": []
}
```

Those start entities are the anchors for the planned Freebase subgraph
extraction stage before feeding GRLprof retriever scoring and path recovery.
