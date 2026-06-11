from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .freebase_kg import Triple

PathTriples = Tuple[Triple, ...]


def recover_reasoning_paths(
    graph_triples: Sequence[Triple],
    scored_triples: Sequence[Sequence[object]],
    q_entities: Sequence[str],
    score_ratio: float = 1.0 / 3.0,
    max_prefix_expansions: int = 3,
    max_paths: int = 256,
) -> List[List[List[str]]]:
    selected = select_high_score_triples(scored_triples, score_ratio)
    if not selected:
        return []
    preds_by_tail = build_predecessors_by_tail(graph_triples)
    score_lookup = {triple: score for triple, score in selected}
    prefix_cache: Dict[str, List[PathTriples]] = {}
    records = []

    for seed_triple, score in selected:
        seed_head = seed_triple.h
        if seed_head not in prefix_cache:
            prefix_cache[seed_head] = find_shortest_prefix_paths(
                seed_head=seed_head,
                q_entities=q_entities,
                preds_by_tail=preds_by_tail,
                max_prefix_expansions=max_prefix_expansions,
            )
        prefixes = prefix_cache[seed_head]
        if prefixes:
            candidates = [prefix + (seed_triple,) for prefix in prefixes]
            best_path = min(
                candidates,
                key=lambda path: (
                    len(path),
                    -path_support_score(path[:-1], score_lookup),
                    tuple(triple.as_tuple() for triple in path),
                ),
            )
        else:
            best_path = (seed_triple,)
        records.append({"path": best_path, "score": score})

    unique: Dict[PathTriples, dict] = {}
    for record in records:
        existing = unique.get(record["path"])
        if existing is None or record["score"] > existing["score"]:
            unique[record["path"]] = record

    kept = list(unique.values())
    removed = set()
    for record_a in kept:
        path_a = record_a["path"]
        for record_b in kept:
            if record_a is record_b:
                continue
            path_b = record_b["path"]
            if is_prefix_path(path_a, path_b):
                removed.add(path_a)
                break

    final_records = [record for record in kept if record["path"] not in removed]
    final_records.sort(
        key=lambda record: (
            -record["score"],
            -len(record["path"]),
            tuple(triple.as_tuple() for triple in record["path"]),
        )
    )
    return [path_to_lists(record["path"]) for record in final_records[:max_paths]]


def select_high_score_triples(
    scored_triples: Sequence[Sequence[object]],
    score_ratio: float,
) -> List[Tuple[Triple, float]]:
    parsed: Dict[Triple, float] = {}
    max_score = None
    for item in scored_triples:
        if len(item) < 4:
            continue
        triple = Triple(str(item[0]), str(item[1]), str(item[2]))
        score = float(item[3])
        parsed[triple] = max(score, parsed.get(triple, float("-inf")))
        max_score = score if max_score is None else max(max_score, score)
    if max_score is None:
        return []
    threshold = max_score * score_ratio
    selected = [(triple, score) for triple, score in parsed.items() if score >= threshold]
    selected.sort(key=lambda item: (-item[1], item[0].as_tuple()))
    return selected


def build_predecessors_by_tail(graph_triples: Sequence[Triple]) -> Dict[str, List[Triple]]:
    preds = defaultdict(list)
    for triple in graph_triples:
        preds[triple.t].append(triple)
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
            prev_head = pred.h
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
            prev_head = pred.h
            if dist.get(prev_head) != current_dist + 1:
                continue
            if dist[prev_head] > best_dist:
                continue
            dfs(prev_head, reversed_path + [pred])

    dfs(seed_head, [])
    seen = set()
    deduped = []
    for path in results:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def path_support_score(path: PathTriples, score_lookup: Dict[Triple, float]) -> float:
    return sum(score_lookup.get(triple, 0.0) for triple in path)


def is_prefix_path(shorter: PathTriples, longer: PathTriples) -> bool:
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def path_to_lists(path: PathTriples) -> List[List[str]]:
    return [triple.as_list() for triple in path]
