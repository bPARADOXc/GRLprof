# Data and Artifact Policy

This repository does not include full WebQSP/CWQ data, Hugging Face dataset caches, base LLM weights, generated path embeddings, or trained adapter checkpoints.

## Merged KGQA Split Format

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

