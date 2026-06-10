# Method

GRLprof is a structure-aware KG-RAG framework for complex KGQA. The main line is:

```text
question processing -> compact subgraph retrieval -> structured knowledge encoding
-> local small-model reasoning
```

## 1. Question Processing and Candidate Subgraph Construction

Given a natural-language question, the system first identifies anchor entities in the knowledge graph and estimates the required reasoning hop count. It then expands from anchor entities with a maximum hop budget and fan-out cap. This bounded expansion keeps likely evidence in the candidate set while avoiding full BFS neighbor explosion.

## 2. Structure-Aware Edge-Level Retrieval

The retriever scores candidate triples instead of enumerating and scoring complete paths with a large language model.

For each triple `(h, r, t)`, the model builds:

- Semantic features: question embedding, head entity text embedding, relation text embedding, tail entity text embedding.
- Structural features: directional distance features from anchor entities to the triple head and tail.

A lightweight MLP scores candidate edges in parallel. This is cheaper than path-level LLM scoring and better suited to local deployment.

## 3. Budgeted Subgraph Selection and Post-Processing

High-scoring edges are not used as a raw top-k bag. The system checks whether evidence edges stay connected to question anchors, supplements bridge edges when needed, and removes duplicate or redundant edges. The result is a compact evidence subgraph with a controlled size and more coherent structure.

## 4. High-Score Path Recovery

The compact subgraph is converted into reasoning paths. Starting from high-score triples, the system backtracks toward question entities, joins prefix paths, and filters by path length, edge score, connectivity, and overlap. It removes duplicate paths, reverse duplicates, strict prefixes, and highly overlapping paths.

Each sample then contains the question, answers, question entities, answer entities, reasoning hop count, and a set of reasoning paths.

## 5. Path Graph Vectorization

Recovered paths are still discrete symbols. GRLprof encodes entity names and relation names into vectors, while preserving triple role order and path order:

- head role
- relation role
- tail role
- triple order inside a path
- relation direction
- multiple-path structure

This avoids collapsing the path into a simple average vector and keeps graph structure available to the adapter.

## 6. Knowledge Adapter and Soft-Prompt Injection

The knowledge adapter fuses textual semantics, topology, and directional information, then projects the result into the frozen language model embedding space. Generation uses:

```text
soft prompt + hard instruction prompt + original question
```

The soft prompt carries KG evidence. The hard prompt specifies the task and output format. The original question remains visible to the language model.

## 7. Training and Inference

Training has two stages:

1. Retriever training: weak supervision marks edges on question-to-answer paths as positives and other candidate edges as negatives.
2. Adapter training: retrieved paths are encoded as soft prompts, and the frozen LLM is optimized through answer generation loss by updating only the adapter-side parameters.

At inference time there are no labels. The system performs entity recognition, candidate subgraph expansion, edge scoring, path recovery, soft-prompt construction, and answer generation.

## Difference from Text RAG

Text RAG:

```text
question -> text retrieval -> concatenate passages -> LLM generation
```

GRLprof:

```text
question -> KG subgraph retrieval -> path recovery -> structural encoding
-> soft-prompt injection -> local LLM generation
```

The system uses entities, relations, direction, hop count, and path structure instead of only retrieved text.

