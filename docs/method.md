# Method

GRLprof is a structure-aware retrieval and soft-prompt framework for complex KGQA. The main line is:

```text
prepared question/topic entities/candidate graph -> edge-level retrieval
-> reasoning-path recovery -> structured path encoding
-> local Llama reasoning
```

## System Boundary

The thesis experiments use the prepared WebQSP and CWQ records as the system input. Each record already supplies the natural-language question, topic entities, answer supervision, and a candidate knowledge graph. Entity linking and online candidate-subgraph acquisition are therefore outside the evaluated method. Their implementation remains in the repository as optional code, but the core training and inference entrypoints do not call it.

## 1. Edge-Level Retrieval

The retriever scores candidate triples instead of enumerating and scoring complete paths with a large language model.

For each triple `(h, r, t)`, the model builds:

- Semantic features: question embedding, head entity text embedding, relation text embedding, tail entity text embedding.
- Structural features: directional distance features from anchor entities to the triple head and tail.

A lightweight MLP scores candidate edges in parallel. This is cheaper than path-level LLM scoring and better suited to local deployment.

## 2. Retrieved Triple Selection and Post-Processing

The retriever ranks candidate triples. The inference artifact retains the highest-scoring triples, and path recovery applies the configured relative score threshold before expansion. Duplicate triples are removed before path construction.

## 3. High-Score Path Recovery

The retrieved triples are converted into reasoning paths. Starting from a high-score seed triple, the system backtracks over the prepared candidate graph toward a topic entity, keeps shortest admissible prefixes, and removes duplicate, reverse-conflicting, and strict-prefix paths.

Each sample then contains the question, answers, question entities, answer entities, reasoning hop count, and a set of reasoning paths.

## 4. Path Graph Vectorization

Recovered paths are still discrete symbols. GRLprof encodes entity names and relation names into vectors, while preserving triple role order and path order:

- head role
- relation role
- tail role
- triple order inside a path
- relation direction
- multiple-path structure

This avoids collapsing the path into a simple average vector and keeps graph structure available to the adapter.

## 5. Knowledge Adapter and Soft-Prompt Injection

The knowledge adapter fuses textual semantics, topology, and directional information, then projects the result into the frozen language model embedding space. Generation uses:

```text
soft prompt + hard instruction prompt + original question
```

The soft prompt carries KG evidence. The hard prompt specifies the task and output format. The original question remains visible to the language model.

## 6. Training and Inference

Training has two stages:

1. Retriever training: weak supervision marks edges on question-to-answer paths as positives and other candidate edges as negatives.
2. Adapter training: retrieved paths are encoded as soft prompts, and the frozen LLM is optimized through answer generation loss by updating only the adapter-side parameters.

At inference time there are no answer labels. For each prepared benchmark sample, the system performs edge scoring, path recovery, soft-prompt construction, and answer generation. It consumes the topic entities and candidate graph supplied by the dataset rather than invoking the optional entity-linking/Freebase front end.

## Difference from Text RAG

Text RAG:

```text
question -> text retrieval -> concatenate passages -> LLM generation
```

GRLprof:

```text
prepared KGQA sample -> edge-level retrieval -> path recovery
-> structural encoding -> soft-prompt injection -> local LLM generation
```

The system uses entities, relations, direction, hop count, and path structure instead of only retrieved text.
