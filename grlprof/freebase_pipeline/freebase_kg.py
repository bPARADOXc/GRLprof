from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True, order=True)
class Triple:
    h: str
    r: str
    t: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.h, self.r, self.t

    def as_list(self) -> List[str]:
        return [self.h, self.r, self.t]


class InMemoryFreebaseKG:
    """Simple Freebase triple store for rough subgraph construction.

    This backend loads triples into adjacency dictionaries. It is suitable for
    experiments and filtered Freebase slices. Large full-dump deployments can
    replace this class with a disk-backed entity adjacency index while keeping
    the sampler interface unchanged.
    """

    def __init__(
        self,
        triples: Iterable[Triple],
        cvt_entities: Optional[Iterable[str]] = None,
        entity_names: Optional[Dict[str, str]] = None,
    ):
        self.out_adj: DefaultDict[str, List[Triple]] = defaultdict(list)
        self.in_adj: DefaultDict[str, List[Triple]] = defaultdict(list)
        self.entities = set()
        self.relations = set()
        for triple in triples:
            self.out_adj[triple.h].append(triple)
            self.in_adj[triple.t].append(triple)
            self.entities.add(triple.h)
            self.entities.add(triple.t)
            self.relations.add(triple.r)
        self.cvt_entities = set(cvt_entities or [])
        self.entity_names = entity_names or {}

    @classmethod
    def from_path(
        cls,
        path: Path,
        delimiter: str = "\t",
        cvt_path: Optional[Path] = None,
        entity_names_path: Optional[Path] = None,
        max_triples: int = -1,
    ) -> "InMemoryFreebaseKG":
        triples = list(load_triples(path, delimiter=delimiter, max_triples=max_triples))
        cvt_entities = load_cvt_entities(cvt_path) if cvt_path else set()
        entity_names = load_entity_names(entity_names_path) if entity_names_path else {}
        return cls(triples, cvt_entities=cvt_entities, entity_names=entity_names)

    def get_in_out_triples(self, entity: str) -> List[Triple]:
        return self.out_adj.get(entity, []) + self.in_adj.get(entity, [])

    def get_out_triples(self, entity: str) -> List[Triple]:
        return self.out_adj.get(entity, [])

    def get_in_triples(self, entity: str) -> List[Triple]:
        return self.in_adj.get(entity, [])

    def degree(self, entity: str) -> int:
        return len(self.out_adj.get(entity, [])) + len(self.in_adj.get(entity, []))

    def has_entity(self, entity: str) -> bool:
        return entity in self.entities

    def is_cvt(self, entity: str) -> bool:
        if entity in self.cvt_entities:
            return True
        name = self.entity_names.get(entity, "")
        if name:
            return False
        degree = self.degree(entity)
        if degree == 0:
            return False
        out_degree = len(self.out_adj.get(entity, []))
        in_degree = len(self.in_adj.get(entity, []))
        # CVTs often have no human-readable name and sit between semantic edges.
        return in_degree > 0 and out_degree > 0 and degree <= 200


def load_triples(path: Path, delimiter: str = "\t", max_triples: int = -1) -> Iterator[Triple]:
    suffix = path.suffix.lower()
    count = 0
    if suffix in (".jsonl", ".json"):
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    triple = _triple_from_json(raw)
                    if triple is not None:
                        yield triple
                        count += 1
                        if 0 < max_triples <= count:
                            return
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("triples") or data.get("graph") or []
            for raw in data:
                triple = _triple_from_json(raw)
                if triple is not None:
                    yield triple
                    count += 1
                    if 0 < max_triples <= count:
                        return
        return

    if suffix in (".pkl", ".pickle"):
        with path.open("rb") as f:
            data = pickle.load(f)
        for raw in data:
            triple = _triple_from_json(raw)
            if triple is not None:
                yield triple
                count += 1
                if 0 < max_triples <= count:
                    return
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) < 3:
                continue
            yield Triple(parts[0], parts[1], parts[2])
            count += 1
            if 0 < max_triples <= count:
                return


def _triple_from_json(raw: object) -> Optional[Triple]:
    if isinstance(raw, dict):
        h = raw.get("h") or raw.get("head") or raw.get("subject")
        r = raw.get("r") or raw.get("relation") or raw.get("predicate")
        t = raw.get("t") or raw.get("tail") or raw.get("object")
        if h is not None and r is not None and t is not None:
            return Triple(str(h), str(r), str(t))
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return Triple(str(raw[0]), str(raw[1]), str(raw[2]))
    return None


def load_cvt_entities(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                data = data.keys()
            return {str(item).strip() for item in data if str(item).strip()}
        return {line.strip() for line in f if line.strip()}


def load_entity_names(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping
