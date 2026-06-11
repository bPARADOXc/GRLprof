from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .freebase_kg import InMemoryFreebaseKG, Triple


NOISE_RELATION_PREFIXES = (
    "common.topic.image",
    "common.topic.webpage",
    "common.topic.article",
    "type.object.key",
    "type.object.permission",
    "freebase.",
    "base.",
    "dataworld.",
    "user.",
)

NOISE_RELATION_SUBSTRINGS = (
    "image",
    "webpage",
    "website",
    "url",
    "uri",
    "identifier",
    "external",
    "source",
    "imported",
    "thumbnail",
)

AUXILIARY_RELATION_PREFIXES = (
    "type.object.type",
    "common.topic.notable_types",
    "time.",
    "measurement_unit.",
)

TEMPORAL_HINTS = (
    "when",
    "date",
    "year",
    "born",
    "birth",
    "died",
    "death",
    "first",
    "last",
    "earliest",
    "latest",
    "oldest",
    "youngest",
)

COMPARISON_HINTS = (
    "largest",
    "smallest",
    "highest",
    "lowest",
    "most",
    "least",
    "earliest",
    "latest",
    "older",
    "younger",
    "more",
    "less",
)


@dataclass
class Difficulty:
    name: str
    hop_level: int
    constraint_level: int
    aggregation: bool = False
    comparison: bool = False
    temporal: bool = False
    multi_topic: bool = False

    @property
    def max_physical_hop(self) -> int:
        if self.name == "hard":
            return 4
        if self.name == "medium":
            return 3
        return 2

    @property
    def relation_top_m(self) -> int:
        if self.name == "hard":
            return 80
        if self.name == "medium":
            return 40
        return 20

    @property
    def max_triples(self) -> int:
        if self.name == "hard":
            return 50000
        if self.name == "medium":
            return 12000
        return 2000

    @property
    def beam_width(self) -> int:
        if self.name == "hard":
            return 1000
        if self.name == "medium":
            return 350
        return 150


def estimate_difficulty(question: str, num_question_entities: int) -> Difficulty:
    q = question.lower()
    aggregation = bool(re.search(r"\b(count|number of|how many)\b", q))
    comparison = any(hint in q for hint in COMPARISON_HINTS)
    temporal = any(hint in q for hint in TEMPORAL_HINTS)
    multi_topic = num_question_entities > 1
    constraint_level = max(0, num_question_entities - 1)

    if aggregation or comparison or (multi_topic and temporal) or num_question_entities >= 3:
        name = "hard"
        hop = 3
    elif multi_topic or temporal or re.search(r"\bwhich|what .* with |where .* in \b", q):
        name = "medium"
        hop = 2
    else:
        name = "easy"
        hop = 1
    return Difficulty(
        name=name,
        hop_level=hop,
        constraint_level=constraint_level,
        aggregation=aggregation,
        comparison=comparison,
        temporal=temporal,
        multi_topic=multi_topic,
    )


def relation_tokens(relation: str) -> Set[str]:
    cleaned = relation.replace(".", " ").replace("_", " ").replace("/", " ")
    return {tok for tok in re.findall(r"[a-zA-Z0-9]+", cleaned.lower()) if len(tok) > 1}


class LexicalRelationPredictor:
    """Lightweight relation predictor used before a trained relation head exists."""

    def __init__(self, relation_descriptions: Optional[Dict[str, str]] = None):
        self.relation_descriptions = relation_descriptions or {}

    def score(self, question: str, relation: str) -> float:
        q_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(tok) > 1}
        r_text = relation
        if relation in self.relation_descriptions:
            r_text += " " + self.relation_descriptions[relation]
        r_tokens = relation_tokens(r_text)
        if not q_tokens or not r_tokens:
            return 0.0
        overlap = len(q_tokens & r_tokens)
        if overlap == 0:
            return 0.0
        return overlap / math.sqrt(len(q_tokens) * len(r_tokens))

    def top_relations(self, question: str, relations: Iterable[str], top_m: int) -> Dict[str, float]:
        scored = [(rel, self.score(question, rel)) for rel in relations]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return {rel: score for rel, score in scored[:top_m] if score > 0.0}


def is_noise_relation(relation: str) -> bool:
    rel = relation.strip().lower().lstrip("/")
    if any(rel.startswith(prefix) for prefix in NOISE_RELATION_PREFIXES):
        return True
    return any(part in rel for part in NOISE_RELATION_SUBSTRINGS)


def is_auxiliary_relation(relation: str) -> bool:
    rel = relation.strip().lower().lstrip("/")
    return any(rel.startswith(prefix) for prefix in AUXILIARY_RELATION_PREFIXES)


def is_temporal_or_numeric_relation(relation: str) -> bool:
    rel = relation.lower()
    return any(
        key in rel
        for key in ("date", "time", "year", "birth", "death", "number", "count", "population", "area")
    )


@dataclass
class RoughSubgraphResult:
    question: str
    difficulty: Difficulty
    start_entities: List[str]
    question_entities: List[str]
    triples: List[Triple]
    relation_scores: Dict[str, float]
    entity_distances: Dict[str, int]

    def graph_as_lists(self) -> List[List[str]]:
        return [triple.as_list() for triple in self.triples]


class RoughSubgraphSampler:
    def __init__(
        self,
        kg: InMemoryFreebaseKG,
        relation_predictor: Optional[LexicalRelationPredictor] = None,
        per_entity_relation_cap: int = 25,
        per_relation_tail_cap: int = 120,
        relation_threshold: float = 0.08,
    ):
        self.kg = kg
        self.relation_predictor = relation_predictor or LexicalRelationPredictor()
        self.per_entity_relation_cap = per_entity_relation_cap
        self.per_relation_tail_cap = per_relation_tail_cap
        self.relation_threshold = relation_threshold

    def build(
        self,
        question: str,
        start_entities: Sequence[str],
        question_entities: Sequence[str],
        difficulty: Optional[Difficulty] = None,
        max_triples: Optional[int] = None,
        max_physical_hop: Optional[int] = None,
        relation_top_m: Optional[int] = None,
    ) -> RoughSubgraphResult:
        q_entities = unique_keep_order([*start_entities, *question_entities])
        starts = unique_keep_order(start_entities or q_entities)
        difficulty = difficulty or estimate_difficulty(question, len(q_entities))
        budget = max_triples or difficulty.max_triples
        max_hop = max_physical_hop or difficulty.max_physical_hop
        top_m = relation_top_m or difficulty.relation_top_m

        candidate_relations = self._candidate_relations(q_entities)
        relation_scores = self.relation_predictor.top_relations(question, candidate_relations, top_m)

        frontier = {entity for entity in q_entities if self.kg.has_entity(entity)}
        if not frontier:
            frontier = set(q_entities)
        visited = set(frontier)
        entity_distances = {entity: 0 for entity in frontier}
        candidate_triples: Dict[Triple, float] = {}
        paths_by_entity: Dict[str, Tuple[Triple, ...]] = {entity: tuple() for entity in frontier}

        for hop in range(1, max_hop + 1):
            next_frontier_scores: Dict[str, float] = {}
            for entity in sorted(frontier):
                triples = self.kg.get_in_out_triples(entity)
                if self.kg.is_cvt(entity):
                    filtered = self._filter_cvt_triples(triples, question, difficulty)
                else:
                    filtered = self._relation_gate(triples, relation_scores, question, difficulty)
                filtered = self._cap_triples(entity, filtered)
                for triple in filtered:
                    score = self._coarse_score(triple, question, starts, q_entities, relation_scores, entity_distances)
                    current = candidate_triples.get(triple)
                    if current is None or score > current:
                        candidate_triples[triple] = score
                    next_entity = triple.t if triple.h == entity else triple.h
                    if next_entity not in visited:
                        next_frontier_scores[next_entity] = max(next_frontier_scores.get(next_entity, float("-inf")), score)
                        paths_by_entity[next_entity] = paths_by_entity.get(entity, tuple()) + (triple,)
                        entity_distances[next_entity] = min(entity_distances.get(next_entity, hop), hop)

            self._add_connecting_paths(candidate_triples, paths_by_entity, q_entities)
            if len(candidate_triples) > budget:
                candidate_triples = self._topk_triples(candidate_triples, budget)

            frontier = {
                entity
                for entity, _ in sorted(next_frontier_scores.items(), key=lambda item: (-item[1], item[0]))[
                    : difficulty.beam_width
                ]
            }
            visited.update(frontier)
            if not frontier:
                break

        final_triples = [
            triple
            for triple, _ in sorted(candidate_triples.items(), key=lambda item: (-item[1], item[0].as_tuple()))
            [:budget]
        ]
        return RoughSubgraphResult(
            question=question,
            difficulty=difficulty,
            start_entities=starts,
            question_entities=q_entities,
            triples=final_triples,
            relation_scores=relation_scores,
            entity_distances=entity_distances,
        )

    def _candidate_relations(self, entities: Sequence[str]) -> Set[str]:
        relations: Set[str] = set()
        for entity in entities:
            for triple in self.kg.get_in_out_triples(entity):
                if not is_noise_relation(triple.r):
                    relations.add(triple.r)
        if len(relations) < 100:
            relations.update(rel for rel in self.kg.relations if not is_noise_relation(rel))
        return relations

    def _relation_gate(
        self,
        triples: Sequence[Triple],
        relation_scores: Dict[str, float],
        question: str,
        difficulty: Difficulty,
    ) -> List[Triple]:
        output = []
        for triple in triples:
            if is_noise_relation(triple.r):
                continue
            rel_score = relation_scores.get(triple.r, 0.0)
            keep = rel_score >= self.relation_threshold or is_auxiliary_relation(triple.r)
            if difficulty.temporal or difficulty.comparison or difficulty.aggregation:
                keep = keep or is_temporal_or_numeric_relation(triple.r)
            if keep:
                output.append(triple)
        if output:
            return output
        # Back off for lexical misses; keep a small number of low-noise edges.
        scored = [
            (self.relation_predictor.score(question, triple.r), triple)
            for triple in triples
            if not is_noise_relation(triple.r)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].as_tuple()))
        return [triple for _, triple in scored[: self.per_entity_relation_cap]]

    def _filter_cvt_triples(
        self,
        triples: Sequence[Triple],
        question: str,
        difficulty: Difficulty,
    ) -> List[Triple]:
        useful = [triple for triple in triples if not is_noise_relation(triple.r)]
        if difficulty.temporal or difficulty.comparison or difficulty.aggregation:
            temporal = [triple for triple in useful if is_temporal_or_numeric_relation(triple.r)]
            useful = unique_triples([*temporal, *useful])
        return useful[: max(self.per_entity_relation_cap * 3, self.per_relation_tail_cap)]

    def _cap_triples(self, entity: str, triples: Sequence[Triple]) -> List[Triple]:
        by_relation: Dict[str, List[Triple]] = defaultdict(list)
        for triple in triples:
            by_relation[triple.r].append(triple)

        output: List[Triple] = []
        for rel, rel_triples in sorted(by_relation.items()):
            rel_triples = sorted(
                rel_triples,
                key=lambda triple: (
                    self.kg.degree(triple.t if triple.h == entity else triple.h),
                    triple.as_tuple(),
                ),
            )
            output.extend(rel_triples[: self.per_relation_tail_cap])
            if len(output) >= self.per_entity_relation_cap * max(len(by_relation), 1):
                break
        return output

    def _coarse_score(
        self,
        triple: Triple,
        question: str,
        start_entities: Sequence[str],
        question_entities: Sequence[str],
        relation_scores: Dict[str, float],
        entity_distances: Dict[str, int],
    ) -> float:
        rel_sim = relation_scores.get(triple.r, self.relation_predictor.score(question, triple.r))
        min_distance = min(entity_distances.get(triple.h, 99), entity_distances.get(triple.t, 99))
        distance_score = 1.0 / (1.0 + min_distance)
        endpoint_bonus = 0.4 if triple.h in question_entities or triple.t in question_entities else 0.0
        start_bonus = 0.2 if triple.h in start_entities or triple.t in start_entities else 0.0
        aux_bonus = 0.1 if is_auxiliary_relation(triple.r) else 0.0
        temporal_bonus = 0.15 if is_temporal_or_numeric_relation(triple.r) else 0.0
        noise_penalty = 1.0 if is_noise_relation(triple.r) else 0.0
        return rel_sim + 0.5 * distance_score + endpoint_bonus + start_bonus + aux_bonus + temporal_bonus - noise_penalty

    @staticmethod
    def _add_connecting_paths(
        candidate_triples: Dict[Triple, float],
        paths_by_entity: Dict[str, Tuple[Triple, ...]],
        question_entities: Sequence[str],
    ) -> None:
        if len(question_entities) <= 1:
            return
        for entity in question_entities:
            for triple in paths_by_entity.get(entity, tuple()):
                candidate_triples[triple] = max(candidate_triples.get(triple, float("-inf")), 10.0)

    @staticmethod
    def _topk_triples(candidate_triples: Dict[Triple, float], k: int) -> Dict[Triple, float]:
        return dict(
            sorted(candidate_triples.items(), key=lambda item: (-item[1], item[0].as_tuple()))[:k]
        )


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def unique_triples(values: Iterable[Triple]) -> List[Triple]:
    seen = set()
    output: List[Triple] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
