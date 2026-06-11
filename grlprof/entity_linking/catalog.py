from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence


def normalize_text(text: object) -> str:
    """Normalize an alias or mention for lookup, keeping behavior ASCII-safe."""
    if text is None:
        return ""
    value = str(text).lower()
    value = value.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def unique_keep_order(values: Iterable[object]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    description: str = ""
    relations: List[str] = field(default_factory=list)
    degree: float = 0.0
    popularity: float = 0.0

    def all_names(self) -> List[str]:
        return unique_keep_order([self.name, *self.aliases])

    def to_text(self, max_aliases: int = 12, max_relations: int = 24) -> str:
        pieces = [f"[NAME] {self.name}"]
        aliases = self.aliases[:max_aliases]
        if aliases:
            pieces.append("[ALIAS] " + "; ".join(aliases))
        if self.types:
            pieces.append("[TYPE] " + "; ".join(self.types[:12]))
        if self.description:
            pieces.append("[DESC] " + self.description)
        if self.relations:
            pieces.append("[REL] " + "; ".join(self.relations[:max_relations]))
        return " ".join(pieces)

    def numeric_features(self) -> List[float]:
        return [
            math.log1p(max(float(self.degree), 0.0)),
            math.log1p(max(float(self.popularity), 0.0)),
            math.log1p(len(self.aliases)),
            math.log1p(len(self.relations)),
        ]

    def to_json(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "aliases": self.aliases,
            "types": self.types,
            "description": self.description,
            "relations": self.relations,
            "degree": self.degree,
            "popularity": self.popularity,
        }


class EntityCatalog:
    """Entity text store plus alias index used by the linker training code.

    The loader accepts a few practical formats:
    - JSONL records with entity_id/name/aliases fields.
    - JSON list of such records.
    - JSON dict mapping entity id to record dict.
    - JSON dict mapping entity name to alias list, such as data/aliases/*.json.
    """

    feature_size = 4

    def __init__(self, records: Optional[Iterable[EntityRecord]] = None):
        self.records: Dict[str, EntityRecord] = {}
        self.alias_to_entity_ids: Dict[str, List[str]] = {}
        if records:
            for record in records:
                self.add(record)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.records

    def add(self, record: EntityRecord) -> None:
        existing = self.records.get(record.entity_id)
        if existing is None:
            record.aliases = unique_keep_order(record.aliases)
            record.types = unique_keep_order(record.types)
            record.relations = unique_keep_order(record.relations)
            self.records[record.entity_id] = record
        else:
            if record.name and not existing.name:
                existing.name = record.name
            existing.aliases = unique_keep_order([*existing.aliases, *record.aliases])
            existing.types = unique_keep_order([*existing.types, *record.types])
            existing.relations = unique_keep_order([*existing.relations, *record.relations])
            if record.description and not existing.description:
                existing.description = record.description
            existing.degree = max(existing.degree, record.degree)
            existing.popularity = max(existing.popularity, record.popularity)
            record = existing

        self._index_record(record)

    def ensure_entity(self, entity_id: object, name: Optional[object] = None) -> Optional[str]:
        if entity_id is None and name is None:
            return None
        entity_id_text = str(entity_id if entity_id is not None else name).strip()
        name_text = str(name if name is not None else entity_id_text).strip()
        if not entity_id_text:
            return None
        if entity_id_text not in self.records:
            self.add(EntityRecord(entity_id=entity_id_text, name=name_text, aliases=[]))
        elif name_text and name_text != self.records[entity_id_text].name:
            self.records[entity_id_text].aliases = unique_keep_order(
                [*self.records[entity_id_text].aliases, name_text]
            )
            self._index_record(self.records[entity_id_text])
        return entity_id_text

    def get(self, entity_id: str) -> EntityRecord:
        return self.records[entity_id]

    def get_text(self, entity_id: str) -> str:
        return self.records[entity_id].to_text()

    def get_features(self, entity_id: str) -> List[float]:
        record = self.records.get(entity_id)
        if record is None:
            return [0.0] * self.feature_size
        return record.numeric_features()

    def alias_candidates(self, mention: object, limit: int = 50) -> List[str]:
        key = normalize_text(mention)
        if not key:
            return []
        return list(self.alias_to_entity_ids.get(key, []))[:limit]

    def iter_ids(self) -> Iterable[str]:
        return self.records.keys()

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entity_id in sorted(self.records):
                f.write(json.dumps(self.records[entity_id].to_json(), ensure_ascii=False) + "\n")

    def _index_record(self, record: EntityRecord) -> None:
        for alias in record.all_names():
            key = normalize_text(alias)
            if not key:
                continue
            bucket = self.alias_to_entity_ids.setdefault(key, [])
            if record.entity_id not in bucket:
                bucket.append(record.entity_id)

    @classmethod
    def from_files(
        cls,
        catalog_paths: Optional[Sequence[Path]] = None,
        alias_paths: Optional[Sequence[Path]] = None,
    ) -> "EntityCatalog":
        catalog = cls()
        for path in catalog_paths or []:
            if path and path.exists():
                _load_catalog_path(catalog, path)
        for path in alias_paths or []:
            if path and path.exists():
                _load_alias_path(catalog, path)
        return catalog


def _record_from_mapping(raw: MutableMapping[str, object], fallback_id: Optional[str] = None) -> EntityRecord:
    entity_id = (
        raw.get("entity_id")
        or raw.get("id")
        or raw.get("mid")
        or raw.get("freebase_id")
        or fallback_id
        or raw.get("name")
    )
    name = raw.get("name") or raw.get("label") or raw.get("title") or entity_id
    aliases = raw.get("aliases") or raw.get("alias") or []
    types = raw.get("types") or raw.get("type") or raw.get("notable_types") or []
    relations = (
        raw.get("relations")
        or raw.get("relation_names")
        or raw.get("outgoing_relations")
        or raw.get("neighbor_relations")
        or []
    )
    if isinstance(aliases, str):
        aliases = [aliases]
    if isinstance(types, str):
        types = [types]
    if isinstance(relations, str):
        relations = [relations]
    return EntityRecord(
        entity_id=str(entity_id),
        name=str(name),
        aliases=unique_keep_order(aliases),
        types=unique_keep_order(types),
        description=str(raw.get("description") or raw.get("desc") or ""),
        relations=unique_keep_order(relations),
        degree=float(raw.get("degree") or raw.get("entity_degree") or 0.0),
        popularity=float(raw.get("popularity") or raw.get("pop") or 0.0),
    )


def _load_catalog_path(catalog: EntityCatalog, path: Path) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if isinstance(raw, dict):
                    catalog.add(_record_from_mapping(raw))
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for raw in data:
            if isinstance(raw, dict):
                catalog.add(_record_from_mapping(raw))
    elif isinstance(data, dict):
        for key, raw in data.items():
            if isinstance(raw, dict):
                catalog.add(_record_from_mapping(raw, fallback_id=str(key)))
            elif isinstance(raw, list):
                aliases = unique_keep_order(raw)
                catalog.add(EntityRecord(entity_id=str(key), name=str(key), aliases=aliases))


def _load_alias_path(catalog: EntityCatalog, path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Alias file must be a JSON object: {path}")
    for name, aliases in data.items():
        if not isinstance(aliases, list):
            aliases = [aliases]
        catalog.add(EntityRecord(entity_id=str(name), name=str(name), aliases=unique_keep_order(aliases)))
