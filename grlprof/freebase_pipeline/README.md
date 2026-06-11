# Freebase Pipeline

This package adds a non-invasive end-to-end path for open-question inference:

```text
question
-> entity linking / start entity selection
-> difficulty-aware relation-gated Freebase rough subgraph
-> GRLprof triple scorer
-> reasoning path recovery
```

It does not modify the original GRLprof training/evaluation scripts.

## Rough Subgraph Strategy

The sampler implements:

- rule-based difficulty estimation;
- start entity plus constraint entity multi-source BFS;
- relation-gated expansion;
- CVT/mediator weak filtering;
- Freebase metadata/noise relation filtering;
- multi-entity connection preference;
- per-entity, per-relation, and global triple budgets.

## Input Freebase Format

`--freebase-triples` supports:

- TSV/plain text: `head<TAB>relation<TAB>tail`
- JSONL records with `h/r/t`, `head/relation/tail`, or
  `subject/predicate/object`
- JSON lists/dicts containing `triples` or `graph`
- pickle lists of triples

Optional files:

- `--freebase-cvt-entities`: one CVT entity id per line or JSON list.
- `--freebase-entity-names`: JSON object or TSV mapping `entity_id` to name.

## Example: Rough Subgraph Only

```bash
python scripts/run_freebase_pipeline.py \
  --question "where was Barack Obama born" \
  --start-entities /m/02mjmr \
  --question-entities /m/02mjmr \
  --freebase-triples /path/to/freebase.tsv \
  --skip-scoring \
  --output-json outputs/freebase_pipeline_obama.json
```

## Example: Full Pipeline

```bash
python scripts/run_freebase_pipeline.py \
  --question "where was Barack Obama born" \
  --linker-checkpoint checkpoints/entity_linker/webqsp_roberta/best \
  --freebase-triples /path/to/freebase.tsv \
  --freebase-cvt-entities /path/to/cvt_entities.txt \
  --retriever-checkpoint checkpoints/retriever/webqsp/cpt.pth \
  --entity-identifiers retrieval/data_files/webqsp/entity_identifiers.txt \
  --text-encoder Alibaba-NLP/gte-large-en-v1.5 \
  --output-json outputs/freebase_pipeline_obama.json \
  --output-pth outputs/retrieval_result_obama.pth
```

If no trained linker checkpoint is available, the script can use alias matching:

```bash
python scripts/run_freebase_pipeline.py \
  --question "what movies did Tom Hanks star in with Meg Ryan" \
  --alias-file data/aliases/WQSP_aliase_data.json \
  --freebase-triples /path/to/freebase.tsv \
  --skip-scoring \
  --output-json outputs/freebase_pipeline_tom_meg.json
```

The `output_pth` file has the same top-level shape as GRLprof retrieval
outputs:

```python
{
  sample_id: {
    "question": ...,
    "scored_triples": ...,
    "q_entity": ...,
    "q_entity_in_graph": ...,
    "max_path_length": ...,
  }
}
```

The JSON output also includes recovered `reasoning_paths`, so it can be used as
the bridge to the existing GRLprof path embedding and QA stages.

## GRLprof CLI Open-Question Fallback

`eval/webqsp_cli_infer.py` now keeps the old dataset-cache path for known
questions. If a question is not found in `train/validation/test.json`, enable
`--open_fallback` to run:

```text
question
-> entity prediction
-> Freebase rough subgraph
-> GRLprof triple scoring
-> path recovery
-> dynamic path embedding
-> GRLprof answer generation
```

Single question:

```bash
python eval/webqsp_cli_infer.py \
  --dataset webqsp \
  --model llama3 \
  --question "where was Barack Obama born" \
  --open_fallback \
  --freebase_triples /path/to/freebase.tsv \
  --linker_checkpoint checkpoints/entity_linker/webqsp_roberta/best \
  --retriever_checkpoint checkpoints/retriever/webqsp/cpt.pth \
  --entity_identifiers retrieval/data_files/webqsp/entity_identifiers.txt \
  --open_output_dir outputs/open_kgqa
```

Shell wrapper:

```bash
FREEBASE_TRIPLES=/path/to/freebase.tsv \
LINKER_CHECKPOINT=checkpoints/entity_linker/webqsp_roberta/best \
RETRIEVER_CHECKPOINT=checkpoints/retriever/webqsp/cpt.pth \
QUESTION="where was Barack Obama born" \
bash scripts/run_open_kgqa_cli.sh --show_all_matched
```

Omit `QUESTION` to start the interactive CLI. The wrapper accepts extra
`eval/webqsp_cli_infer.py` arguments after the script command, so retrieval
budgets can be tuned without editing the file:

```bash
FREEBASE_TRIPLES=/path/to/freebase.tsv \
SKIP_OPEN_SCORING=1 \
bash scripts/run_open_kgqa_cli.sh \
  --start_entities /m/02mjmr \
  --question_entities /m/02mjmr \
  --max_physical_hop 2 \
  --rough_max_triples 3000
```
