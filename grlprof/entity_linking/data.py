from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .catalog import EntityCatalog, normalize_text, unique_keep_order


@dataclass
class GoldMention:
    text: str
    entity_id: str
    entity_name: str
    char_span: Optional[Tuple[int, int]] = None
    is_start: bool = False


@dataclass
class LinkerExample:
    qid: str
    question: str
    mentions: List[GoldMention] = field(default_factory=list)
    positive_start_entity_ids: List[str] = field(default_factory=list)
    relation_labels: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def load_json_or_jsonl(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        output = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    output.append(json.loads(line))
        return output
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [
            {"id": key, **value} if isinstance(value, dict) else {"id": key, "value": value}
            for key, value in data.items()
        ]
    raise ValueError(f"Unsupported sample file format: {path}")


def build_relation_vocab(paths: Sequence[Path], max_size: int = 2048) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in paths:
        for item in load_json_or_jsonl(path):
            for rel in extract_relation_labels(item):
                counts[rel] = counts.get(rel, 0) + 1
    sorted_rels = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if max_size > 0:
        sorted_rels = sorted_rels[:max_size]
    return {rel: idx for idx, (rel, _) in enumerate(sorted_rels)}


def save_relation_vocab(vocab: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_relation_vocab(path: Optional[Path]) -> Dict[str, int]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def extract_relation_labels(item: dict) -> List[str]:
    rels: List[str] = []
    for key in ("inferential_chain", "path_relations", "relation_labels", "relations"):
        value = item.get(key)
        if isinstance(value, str):
            rels.append(value)
        elif isinstance(value, list):
            for rel in value:
                if isinstance(rel, str):
                    rels.append(rel)
    for path in item.get("reasoning_paths") or []:
        if not isinstance(path, list):
            continue
        for triple in path:
            if isinstance(triple, (list, tuple)) and len(triple) >= 2:
                rels.append(str(triple[1]))
    return unique_keep_order(rels)


def _normalize_with_mapping(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    mapping: List[int] = []
    last_space = True
    for idx, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            mapping.append(idx)
            last_space = False
        else:
            if not last_space:
                chars.append(" ")
                mapping.append(idx)
                last_space = True
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return "".join(chars), mapping


def find_mention_span(question: str, aliases: Iterable[str]) -> Optional[Tuple[int, int]]:
    normalized_question, mapping = _normalize_with_mapping(question)
    if not normalized_question:
        return None

    alias_variants = []
    for alias in aliases:
        norm = normalize_text(alias)
        if norm:
            alias_variants.append(norm)
    alias_variants = sorted(set(alias_variants), key=lambda value: (-len(value), value))

    for alias in alias_variants:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        match = pattern.search(normalized_question)
        if not match:
            continue
        norm_start, norm_end = match.span()
        if norm_start >= len(mapping) or norm_end - 1 >= len(mapping):
            continue
        char_start = mapping[norm_start]
        char_end = mapping[norm_end - 1] + 1
        return char_start, char_end
    return None


def _as_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_entity_pairs(item: dict, catalog: EntityCatalog) -> List[Tuple[str, str]]:
    explicit_mentions = item.get("mentions")
    pairs: List[Tuple[str, str]] = []
    if isinstance(explicit_mentions, list):
        for mention in explicit_mentions:
            if not isinstance(mention, dict):
                continue
            entity_id = (
                mention.get("entity_id")
                or mention.get("mid")
                or mention.get("id")
                or mention.get("entity")
            )
            name = mention.get("entity_name") or mention.get("name") or mention.get("text") or entity_id
            ensured = catalog.ensure_entity(entity_id, name)
            if ensured:
                pairs.append((ensured, str(name)))
        if pairs:
            return pairs

    names = _as_list(item.get("q_entity") or item.get("q_entity_in_graph") or item.get("topic_entity"))
    ids = _as_list(item.get("q_entity_id_list") or item.get("q_entity_ids") or item.get("topic_entity_id"))
    if ids and len(ids) == len(names):
        for entity_id, name in zip(ids, names):
            ensured = catalog.ensure_entity(entity_id, name)
            if ensured:
                pairs.append((ensured, str(name)))
    else:
        for name in names:
            ensured = catalog.ensure_entity(name, name)
            if ensured:
                pairs.append((ensured, str(name)))
    return pairs


def _extract_positive_starts(item: dict, entity_pairs: Sequence[Tuple[str, str]]) -> List[str]:
    for key in (
        "positive_start_entities",
        "positive_start_entity_ids",
        "start_entities",
        "start_entity_ids",
    ):
        values = _as_list(item.get(key))
        if values:
            return unique_keep_order(values)
    for key in ("start_entity", "start_entity_id", "topic_entity_id", "topic_entity"):
        value = item.get(key)
        if value:
            return unique_keep_order(_as_list(value))
    return unique_keep_order(entity_id for entity_id, _ in entity_pairs)


def build_catalog_from_samples(catalog: EntityCatalog, paths: Sequence[Path]) -> None:
    for path in paths:
        for item in load_json_or_jsonl(path):
            for entity_id, name in _extract_entity_pairs(item, catalog):
                catalog.ensure_entity(entity_id, name)
            for name in _as_list(item.get("a_entity") or item.get("answer")):
                if isinstance(name, dict):
                    entity_id = name.get("entity_id") or name.get("mid") or name.get("id") or name.get("text")
                    label = name.get("entity_name") or name.get("name") or name.get("text") or entity_id
                    catalog.ensure_entity(entity_id, label)
                else:
                    catalog.ensure_entity(name, name)
            for reasoning_path in item.get("reasoning_paths") or []:
                if not isinstance(reasoning_path, list):
                    continue
                for triple in reasoning_path:
                    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                        catalog.ensure_entity(triple[0], triple[0])
                        catalog.ensure_entity(triple[2], triple[2])


class KGQALinkerDataset(Dataset):
    def __init__(
        self,
        path: Path,
        catalog: EntityCatalog,
        relation_vocab: Optional[Dict[str, int]] = None,
    ):
        self.path = path
        self.catalog = catalog
        self.relation_vocab = relation_vocab or {}
        self.examples = [self._convert(item) for item in load_json_or_jsonl(path)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LinkerExample:
        return self.examples[index]

    def _convert(self, item: dict) -> LinkerExample:
        question = str(item.get("question") or item.get("utterance") or "")
        qid = str(item.get("id") or item.get("question_id") or len(question))
        entity_pairs = _extract_entity_pairs(item, self.catalog)
        positive_starts = _extract_positive_starts(item, entity_pairs)

        explicit_mentions = item.get("mentions") if isinstance(item.get("mentions"), list) else []
        starts = set(str(value) for value in positive_starts)
        mentions: List[GoldMention] = []

        if explicit_mentions:
            for raw_mention in explicit_mentions:
                if not isinstance(raw_mention, dict):
                    continue
                entity_id = (
                    raw_mention.get("entity_id")
                    or raw_mention.get("mid")
                    or raw_mention.get("id")
                    or raw_mention.get("entity")
                )
                name = raw_mention.get("entity_name") or raw_mention.get("name") or raw_mention.get("text") or entity_id
                ensured = self.catalog.ensure_entity(entity_id, name)
                if not ensured:
                    continue
                span = raw_mention.get("char_span") or raw_mention.get("span_char")
                char_span = None
                if isinstance(span, list) and len(span) == 2:
                    char_span = (int(span[0]), int(span[1]))
                if char_span is None:
                    record = self.catalog.get(ensured)
                    char_span = find_mention_span(question, record.all_names())
                mentions.append(
                    GoldMention(
                        text=str(raw_mention.get("text") or name),
                        entity_id=ensured,
                        entity_name=str(name),
                        char_span=char_span,
                        is_start=bool(raw_mention.get("is_start") or ensured in starts),
                    )
                )
        else:
            for entity_id, name in entity_pairs:
                record = self.catalog.get(entity_id)
                char_span = find_mention_span(question, record.all_names())
                mentions.append(
                    GoldMention(
                        text=name,
                        entity_id=entity_id,
                        entity_name=name,
                        char_span=char_span,
                        is_start=entity_id in starts,
                    )
                )

        return LinkerExample(
            qid=qid,
            question=question,
            mentions=mentions,
            positive_start_entity_ids=positive_starts,
            relation_labels=extract_relation_labels(item),
            raw=item,
        )


class LinkerCollator:
    def __init__(
        self,
        tokenizer,
        catalog: EntityCatalog,
        max_question_len: int = 96,
        max_entity_len: int = 160,
        max_span_len: int = 10,
        negatives_per_positive: int = 15,
        alias_candidate_limit: int = 50,
        start_negative_count: int = 20,
        relation_vocab: Optional[Dict[str, int]] = None,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.max_question_len = max_question_len
        self.max_entity_len = max_entity_len
        self.max_span_len = max_span_len
        self.negatives_per_positive = negatives_per_positive
        self.alias_candidate_limit = alias_candidate_limit
        self.start_negative_count = start_negative_count
        self.relation_vocab = relation_vocab or {}
        self.random = random.Random(seed)
        self.all_entity_ids = list(catalog.iter_ids())

    def __call__(self, examples: Sequence[LinkerExample]) -> dict:
        questions = [example.question for example in examples]
        q_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = q_inputs.pop("offset_mapping")

        batch_spans: List[List[Tuple[int, int]]] = []
        batch_span_labels: List[torch.Tensor] = []
        link_groups: List[dict] = []
        start_groups: List[dict] = []
        entity_ids_to_encode: List[str] = []
        entity_seen = set()

        for batch_idx, example in enumerate(examples):
            spans = self._enumerate_spans(offsets[batch_idx])
            span_to_index = {span: idx for idx, span in enumerate(spans)}
            positive_span_to_mentions = self._gold_token_spans(example, offsets[batch_idx])
            span_labels = torch.zeros(len(spans), dtype=torch.float32)
            for span in positive_span_to_mentions:
                if span in span_to_index:
                    span_labels[span_to_index[span]] = 1.0

            batch_spans.append(spans)
            batch_span_labels.append(span_labels)

            for span, mentions in positive_span_to_mentions.items():
                if span not in span_to_index:
                    continue
                positive_ids = unique_keep_order(mention.entity_id for mention in mentions)
                candidate_ids = self._candidate_entities(mentions, positive_ids)
                self._collect_entity_ids(candidate_ids, entity_ids_to_encode, entity_seen)
                link_groups.append(
                    {
                        "batch_index": batch_idx,
                        "span": span,
                        "span_index": span_to_index[span],
                        "candidate_ids": candidate_ids,
                        "positive_ids": positive_ids,
                    }
                )

            start_positive_ids = [entity_id for entity_id in example.positive_start_entity_ids if entity_id in self.catalog]
            if start_positive_ids:
                start_candidate_ids = self._start_candidates(example, start_positive_ids)
                self._collect_entity_ids(start_candidate_ids, entity_ids_to_encode, entity_seen)
                start_groups.append(
                    {
                        "batch_index": batch_idx,
                        "candidate_ids": start_candidate_ids,
                        "positive_ids": start_positive_ids,
                    }
                )

        if not entity_ids_to_encode:
            entity_ids_to_encode = self.all_entity_ids[:1]
        entity_texts = [self.catalog.get_text(entity_id) for entity_id in entity_ids_to_encode]
        entity_inputs = self.tokenizer(
            entity_texts,
            padding=True,
            truncation=True,
            max_length=self.max_entity_len,
            return_tensors="pt",
        )
        entity_features = torch.tensor(
            [self.catalog.get_features(entity_id) for entity_id in entity_ids_to_encode],
            dtype=torch.float32,
        )

        path_labels = None
        if self.relation_vocab:
            path_labels = torch.zeros((len(examples), len(self.relation_vocab)), dtype=torch.float32)
            for batch_idx, example in enumerate(examples):
                for rel in example.relation_labels:
                    rel_idx = self.relation_vocab.get(rel)
                    if rel_idx is not None:
                        path_labels[batch_idx, rel_idx] = 1.0

        return {
            "question_inputs": q_inputs,
            "spans": batch_spans,
            "span_labels": batch_span_labels,
            "link_groups": link_groups,
            "start_groups": start_groups,
            "entity_inputs": entity_inputs,
            "entity_ids": entity_ids_to_encode,
            "entity_features": entity_features,
            "path_labels": path_labels,
            "examples": examples,
        }

    def _collect_entity_ids(self, ids: Iterable[str], output: List[str], seen: set) -> None:
        for entity_id in ids:
            if entity_id not in self.catalog or entity_id in seen:
                continue
            seen.add(entity_id)
            output.append(entity_id)

    def _enumerate_spans(self, offsets: torch.Tensor) -> List[Tuple[int, int]]:
        valid_positions = [
            idx
            for idx, (start, end) in enumerate(offsets.tolist())
            if end > start
        ]
        spans: List[Tuple[int, int]] = []
        for start_pos_index, start in enumerate(valid_positions):
            for end in valid_positions[start_pos_index : start_pos_index + self.max_span_len]:
                spans.append((start, end))
        return spans

    def _gold_token_spans(
        self,
        example: LinkerExample,
        offsets: torch.Tensor,
    ) -> Dict[Tuple[int, int], List[GoldMention]]:
        output: Dict[Tuple[int, int], List[GoldMention]] = {}
        for mention in example.mentions:
            if mention.char_span is None:
                continue
            token_span = self._char_to_token_span(mention.char_span, offsets)
            if token_span is None:
                continue
            output.setdefault(token_span, []).append(mention)
        return output

    @staticmethod
    def _char_to_token_span(
        char_span: Tuple[int, int],
        offsets: torch.Tensor,
    ) -> Optional[Tuple[int, int]]:
        char_start, char_end = char_span
        token_start = None
        token_end = None
        for idx, (start, end) in enumerate(offsets.tolist()):
            if end <= start:
                continue
            if token_start is None and start <= char_start < end:
                token_start = idx
            if start < char_end <= end:
                token_end = idx
                break
            if start < char_end and end >= char_end:
                token_end = idx
                break
        if token_start is None:
            for idx, (start, end) in enumerate(offsets.tolist()):
                if end <= start:
                    continue
                if start >= char_start:
                    token_start = idx
                    break
        if token_end is None:
            for idx, (start, end) in reversed(list(enumerate(offsets.tolist()))):
                if end <= start:
                    continue
                if end <= char_end:
                    token_end = idx
                    break
        if token_start is None or token_end is None or token_end < token_start:
            return None
        return token_start, token_end

    def _candidate_entities(
        self,
        mentions: Sequence[GoldMention],
        positive_ids: Sequence[str],
    ) -> List[str]:
        candidates = list(positive_ids)
        for mention in mentions:
            for alias in (mention.text, mention.entity_name):
                candidates.extend(self.catalog.alias_candidates(alias, limit=self.alias_candidate_limit))
            if mention.entity_id in self.catalog:
                record = self.catalog.get(mention.entity_id)
                for alias in record.all_names():
                    candidates.extend(self.catalog.alias_candidates(alias, limit=self.alias_candidate_limit))
        candidates = unique_keep_order(entity_id for entity_id in candidates if entity_id in self.catalog)
        while len(candidates) < len(positive_ids) + self.negatives_per_positive and self.all_entity_ids:
            candidate = self.random.choice(self.all_entity_ids)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _start_candidates(
        self,
        example: LinkerExample,
        positive_ids: Sequence[str],
    ) -> List[str]:
        candidates = list(positive_ids)
        for mention in example.mentions:
            if mention.entity_id in self.catalog:
                candidates.append(mention.entity_id)
                record = self.catalog.get(mention.entity_id)
                for alias in record.all_names():
                    candidates.extend(self.catalog.alias_candidates(alias, limit=self.alias_candidate_limit))
        candidates = unique_keep_order(entity_id for entity_id in candidates if entity_id in self.catalog)
        while len(candidates) < len(positive_ids) + self.start_negative_count and self.all_entity_ids:
            candidate = self.random.choice(self.all_entity_ids)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates
