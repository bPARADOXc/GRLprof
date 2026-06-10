#!/usr/bin/env python3
"""Expand high-scoring KGQA triples into paths from the local HF cache.

This script reads scored triples from an input directory and expands each
high-scoring triple backward until its head entity reaches `q_entity_in_graph`.
The graph source is the local Hugging Face cache for WebQSP or CWQ.
  python scripts/expand_webqsp_scored_triples.py \
    --input-root retrieval/runs/webqsp \
    --output-root data/webqsp_may_merged
  python scripts/expand_webqsp_scored_triples.py \
    --dataset cwq \
    --input-root retrieval/runs/cwq \
    --output-root data/cwq_may_merged
Supported input layouts:
1. Split JSON files:
   - train.json
   - val.json
   - test.json
2. Split directories with one JSON per case:
   - train/*.json
   - val/*.json
   - test/*.json

Output files follow the `webqsp_merged/*.json` list format:
   - train.json
   - validation.json
   - test.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, concatenate_datasets
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "This script requires the `datasets` package. "
        "Use a Python env that can read the local Hugging Face cache."
    ) from exc


Triple = Tuple[str, str, str]
PathTriples = Tuple[Triple, ...]

SPLIT_TO_INPUT_NAME = {
    "train": "train",
    "val": "val",
    "test": "test",
}

SPLIT_TO_OUTPUT_FILE = {
    "train": "train.json",
    "val": "validation.json",
    "test": "test.json",
}

SPLIT_TO_HF_NAME = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

DATASET_SPECS = {
    "webqsp": {
        "dataset_dir_glob": "ml1996___webqsp/default/*/*",
        "split_file_globs": {
            "train": ("webqsp-train-*.arrow",),
            "validation": ("webqsp-validation.arrow", "webqsp-validation-*.arrow"),
            "test": ("webqsp-test.arrow", "webqsp-test-*.arrow"),
        },
        "default_input_root": Path("retrieval/runs/webqsp"),
        "default_output_root": Path("data/webqsp_may_merged"),
        "default_hf_cache_roots": (
            Path.home() / ".cache" / "huggingface" / "datasets",
        ),
    },
    "cwq": {
        "dataset_dir_glob": "rmanluo___ro_g-cwq/default/*/*",
        "split_file_globs": {
            "train": ("ro_g-cwq-train-*.arrow",),
            "validation": ("ro_g-cwq-validation-*.arrow",),
            "test": ("ro_g-cwq-test-*.arrow",),
        },
        "default_input_root": Path("retrieval/runs/cwq"),
        "default_output_root": Path("data/cwq_may_merged"),
        "default_hf_cache_roots": (
            Path.home() / ".cache" / "huggingface" / "datasets",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default="webqsp",
        help="Dataset layout to use for resolving the HF graph cache.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Directory containing split JSON files or split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for merged output JSON files.",
    )
    parser.add_argument(
        "--hf-cache-root",
        type=Path,
        default=None,
        help=(
            "HF datasets cache root, or the resolved dataset cache directory "
            "that directly contains *.arrow files."
        ),
    )
    parser.add_argument(
        "--score-ratio",
        type=float,
        default=1.0 / 3.0,
        help="Only expand triples with score >= max_score * score_ratio.",
    )
    parser.add_argument(
        "--max-prefix-expansions",
        type=int,
        default=3,
        help="Maximum number of backward expansion steps before the seed triple.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Which splits to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to process per split for debugging.",
    )
    return parser.parse_args()


def _matching_split_files(dataset_dir: Path, dataset: str, hf_split_name: str) -> List[Path]:
    spec = DATASET_SPECS[dataset]
    files: List[Path] = []
    for pattern in spec["split_file_globs"][hf_split_name]:
        files.extend(sorted(dataset_dir.glob(pattern)))
    seen = set()
    unique_files = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    return unique_files


def _looks_like_dataset_dir(dataset_dir: Path, dataset: str) -> bool:
    return all(
        _matching_split_files(dataset_dir, dataset, hf_split_name)
        for hf_split_name in ("train", "validation", "test")
    )


def resolve_hf_dataset_dir(dataset: str, hf_cache_root: Optional[Path]) -> Path:
    spec = DATASET_SPECS[dataset]
    search_roots = [hf_cache_root] if hf_cache_root is not None else spec["default_hf_cache_roots"]

    for search_root in search_roots:
        if search_root is None or not search_root.exists():
            continue
        if _looks_like_dataset_dir(search_root, dataset):
            return search_root

        for candidate in sorted(search_root.glob(spec["dataset_dir_glob"])):
            if _looks_like_dataset_dir(candidate, dataset):
                return candidate

    searched = ", ".join(str(root) for root in search_roots if root is not None)
    raise FileNotFoundError(
        f"Could not resolve the local HF cache directory for {dataset} "
        f"under: {searched}."
    )


def load_arrow_files(paths: Sequence[Path]) -> Dataset:
    if not paths:
        raise ValueError("No arrow files provided.")
    datasets = [Dataset.from_file(str(path)) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def load_hf_splits(dataset: str, hf_dataset_dir: Path) -> Dict[str, Dataset]:
    hf_splits = {}
    for hf_split_name in ("train", "validation", "test"):
        files = _matching_split_files(hf_dataset_dir, dataset, hf_split_name)
        if not files:
            raise FileNotFoundError(
                f"No {hf_split_name} arrow files found for {dataset} in {hf_dataset_dir}."
            )
        hf_splits[hf_split_name] = load_arrow_files(files)
    return hf_splits


def build_id_to_index(dataset: Dataset) -> Dict[str, int]:
    return {sample_id: idx for idx, sample_id in enumerate(dataset["id"])}


def load_input_samples(input_root: Path, split: str) -> List[dict]:
    split_name = SPLIT_TO_INPUT_NAME[split]
    split_file = input_root / f"{split_name}.json"
    split_dir = input_root / split_name

    samples: List[dict] = []

    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            for sample_id, sample in payload.items():
                if not isinstance(sample, dict):
                    continue
                sample_copy = dict(sample)
                sample_copy.setdefault("id", sample_id)
                samples.append(sample_copy)
        elif isinstance(payload, list):
            for sample in payload:
                if isinstance(sample, dict):
                    samples.append(dict(sample))
        else:
            raise ValueError(f"Unsupported JSON payload type in {split_file}.")
        return samples

    if split_dir.is_dir():
        for case_file in sorted(split_dir.rglob("*.json")):
            with case_file.open("r", encoding="utf-8") as f:
                sample = json.load(f)
            if not isinstance(sample, dict):
                continue
            sample = dict(sample)
            sample.setdefault("id", case_file.stem)
            samples.append(sample)
        return samples

    raise FileNotFoundError(
        f"Could not find either {split_file} or split directory {split_dir}."
    )


def normalize_triple(raw_triple: Sequence[object]) -> Triple:
    if len(raw_triple) < 3:
        raise ValueError(f"Invalid triple: {raw_triple}")
    return (str(raw_triple[0]), str(raw_triple[1]), str(raw_triple[2]))


def deduplicate_graph_triples(graph: Iterable[Sequence[object]]) -> List[Triple]:
    seen = set()
    deduped: List[Triple] = []
    for raw_triple in graph:
        triple = normalize_triple(raw_triple)
        if triple in seen:
            continue
        seen.add(triple)
        deduped.append(triple)
    return deduped


def select_high_score_triples(
    scored_triples: Sequence[Sequence[object]],
    score_ratio: float,
) -> List[Tuple[Triple, float]]:
    parsed: Dict[Triple, float] = {}
    max_score = None

    for item in scored_triples:
        if len(item) < 4:
            continue
        triple = normalize_triple(item[:3])
        score = float(item[3])
        prev_score = parsed.get(triple)
        if prev_score is None or score > prev_score:
            parsed[triple] = score
        if max_score is None or score > max_score:
            max_score = score

    if max_score is None:
        return []

    threshold = max_score * score_ratio
    selected = [
        (triple, score)
        for triple, score in parsed.items()
        if score >= threshold
    ]
    selected.sort(key=lambda item: (-item[1], item[0]))
    return selected


def build_predecessors_by_tail(graph_triples: Sequence[Triple]) -> Dict[str, List[Triple]]:
    preds = defaultdict(list)
    for triple in graph_triples:
        preds[triple[2]].append(triple)
    return preds


def find_shortest_prefix_paths(
    seed_head: str,
    q_entities: Sequence[str],
    preds_by_tail: Dict[str, List[Triple]],
    max_prefix_expansions: int,
) -> List[PathTriples]:
    q_entity_set = set(q_entities)
    if seed_head in q_entity_set:
        return [tuple()]

    dist: Dict[str, int] = {seed_head: 0}
    queue: deque[str] = deque([seed_head])

    while queue:
        current = queue.popleft()
        current_dist = dist[current]
        if current_dist >= max_prefix_expansions:
            continue

        for pred in preds_by_tail.get(current, []):
            prev_head = pred[0]
            if prev_head in dist:
                continue
            dist[prev_head] = current_dist + 1
            queue.append(prev_head)

    candidate_q_entities = [
        q_entity
        for q_entity in q_entity_set
        if q_entity in dist and dist[q_entity] <= max_prefix_expansions
    ]
    if not candidate_q_entities:
        return []

    best_dist = min(dist[q_entity] for q_entity in candidate_q_entities)
    results: List[PathTriples] = []

    def dfs(current_tail: str, reversed_path: List[Triple]) -> None:
        current_dist = dist[current_tail]
        if current_tail in q_entity_set and current_dist == best_dist:
            results.append(tuple(reversed(reversed_path)))
            return

        for pred in preds_by_tail.get(current_tail, []):
            prev_head = pred[0]
            if dist.get(prev_head) != current_dist + 1:
                continue
            if dist[prev_head] > best_dist:
                continue
            dfs(prev_head, reversed_path + [pred])

    dfs(seed_head, [])

    deduped = []
    seen = set()
    for path in results:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def is_prefix_path(shorter: PathTriples, longer: PathTriples) -> bool:
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def convert_path_to_lists(path: PathTriples) -> List[List[str]]:
    return [[h, r, t] for h, r, t in path]


def path_support_score(path: PathTriples, triple_scores: Dict[Triple, float]) -> float:
    return sum(triple_scores.get(triple, 0.0) for triple in path)


def build_reasoning_paths_for_sample(
    sample: dict,
    dataset_row: dict,
    score_ratio: float,
    max_prefix_expansions: int,
) -> List[List[List[str]]]:
    q_entities = sample.get("q_entity_in_graph") or dataset_row.get("q_entity") or sample.get("q_entity") or []
    graph_triples = deduplicate_graph_triples(dataset_row["graph"])
    preds_by_tail = build_predecessors_by_tail(graph_triples)
    selected_triples = select_high_score_triples(sample.get("scored_triples", []), score_ratio)
    triple_score_lookup = {triple: score for triple, score in selected_triples}

    if not selected_triples:
        return []

    prefix_cache: Dict[str, List[PathTriples]] = {}
    seed_to_records: Dict[Triple, List[dict]] = defaultdict(list)

    for seed_triple, score in selected_triples:
        seed_head = seed_triple[0]
        if seed_head not in prefix_cache:
            prefix_cache[seed_head] = find_shortest_prefix_paths(
                seed_head=seed_head,
                q_entities=q_entities,
                preds_by_tail=preds_by_tail,
                max_prefix_expansions=max_prefix_expansions,
            )

        prefix_paths = prefix_cache[seed_head]
        if prefix_paths:
            candidate_paths = [prefix + (seed_triple,) for prefix in prefix_paths]
            best_path = min(
                candidate_paths,
                key=lambda path: (
                    len(path),
                    -path_support_score(path[:-1], triple_score_lookup),
                    path,
                ),
            )
        else:
            best_path = (seed_triple,)

        seed_to_records[seed_triple].append(
            {
                "path": best_path,
                "score": score,
                "seed": seed_triple,
                "removed": False,
            }
        )

    selected_seed_list = [seed for seed, _ in selected_triples]
    for i, seed_a in enumerate(selected_seed_list):
        if seed_a not in seed_to_records:
            continue
        for seed_b in selected_seed_list[i + 1 :]:
            if seed_a[0] != seed_b[2] or seed_a[2] != seed_b[0]:
                continue
            active_a = [record for record in seed_to_records[seed_a] if not record["removed"]]
            active_b = [record for record in seed_to_records[seed_b] if not record["removed"]]
            if not active_a or not active_b:
                continue

            min_len_a = min(len(record["path"]) for record in active_a)
            min_len_b = min(len(record["path"]) for record in active_b)
            if min_len_a == min_len_b:
                continue

            loser_seed = seed_a if min_len_a > min_len_b else seed_b
            for record in seed_to_records[loser_seed]:
                record["removed"] = True

    unique_records: Dict[PathTriples, dict] = {}
    for records in seed_to_records.values():
        for record in records:
            if record["removed"]:
                continue
            existing = unique_records.get(record["path"])
            if existing is None or record["score"] > existing["score"]:
                unique_records[record["path"]] = record

    kept_records = list(unique_records.values())
    removed_paths = set()
    for record_a in kept_records:
        path_a = record_a["path"]
        for record_b in kept_records:
            if record_a is record_b:
                continue
            path_b = record_b["path"]
            if is_prefix_path(path_a, path_b):
                removed_paths.add(path_a)
                break

    final_records = [
        record
        for record in kept_records
        if record["path"] not in removed_paths
    ]
    final_records.sort(key=lambda record: (-record["score"], -len(record["path"]), record["path"]))

    return [convert_path_to_lists(record["path"]) for record in final_records]


def build_output_item(sample: dict, dataset_row: dict, reasoning_paths: List[List[List[str]]]) -> dict:
    hop = sample.get("max_path_length")
    return {
        "id": sample["id"],
        "question": dataset_row.get("question", sample.get("question", "")),
        "answer": dataset_row.get("answer", sample.get("a_entity", [])),
        "q_entity": dataset_row.get("q_entity", sample.get("q_entity", [])),
        "a_entity": dataset_row.get("a_entity", sample.get("a_entity", [])),
        "hop": hop,
        "num_reasoning_paths": len(reasoning_paths),
        "reasoning_paths": reasoning_paths,
    }


def find_dataset_row(
    sample_id: str,
    split: str,
    hf_splits: Dict[str, Dataset],
    hf_indices: Dict[str, Dict[str, int]],
) -> dict:
    preferred_hf_split = SPLIT_TO_HF_NAME[split]
    if sample_id in hf_indices[preferred_hf_split]:
        return hf_splits[preferred_hf_split][hf_indices[preferred_hf_split][sample_id]]

    for hf_split_name, split_indices in hf_indices.items():
        if sample_id in split_indices:
            return hf_splits[hf_split_name][split_indices[sample_id]]

    raise KeyError(f"Could not find sample {sample_id} in local HF cache.")


def process_split(
    split: str,
    samples: Sequence[dict],
    hf_splits: Dict[str, Dataset],
    hf_indices: Dict[str, Dict[str, int]],
    score_ratio: float,
    max_prefix_expansions: int,
) -> List[dict]:
    output = []
    for sample in samples:
        dataset_row = find_dataset_row(
            sample_id=sample["id"],
            split=split,
            hf_splits=hf_splits,
            hf_indices=hf_indices,
        )
        reasoning_paths = build_reasoning_paths_for_sample(
            sample=sample,
            dataset_row=dataset_row,
            score_ratio=score_ratio,
            max_prefix_expansions=max_prefix_expansions,
        )
        output.append(build_output_item(sample, dataset_row, reasoning_paths))
    return output


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    input_root = args.input_root or spec["default_input_root"]
    output_root = args.output_root or spec["default_output_root"]

    hf_dataset_dir = resolve_hf_dataset_dir(args.dataset, args.hf_cache_root)
    print(f"[{args.dataset}] using HF dataset cache: {hf_dataset_dir}")
    hf_splits = load_hf_splits(args.dataset, hf_dataset_dir)
    hf_indices = {
        split_name: build_id_to_index(dataset)
        for split_name, dataset in hf_splits.items()
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        samples = load_input_samples(input_root, split)
        if args.limit is not None:
            samples = samples[: args.limit]

        merged_samples = process_split(
            split=split,
            samples=samples,
            hf_splits=hf_splits,
            hf_indices=hf_indices,
            score_ratio=args.score_ratio,
            max_prefix_expansions=args.max_prefix_expansions,
        )

        output_path = output_root / SPLIT_TO_OUTPUT_FILE[split]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged_samples, f, ensure_ascii=False, indent=2)

        print(
            f"[{split}] wrote {len(merged_samples)} samples to {output_path}"
        )


if __name__ == "__main__":
    main()
