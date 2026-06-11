from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from grlprof.entity_linking.catalog import EntityCatalog, normalize_text
from grlprof.entity_linking.model import KGELQStartModel, mean_pool


@dataclass
class LinkedEntity:
    entity_id: str
    entity_name: str
    mention: str
    link_score: float
    start_score: float = 0.0
    is_start: bool = False

    def to_json(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "mention": self.mention,
            "link_score": self.link_score,
            "start_score": self.start_score,
            "is_start": self.is_start,
        }


class AliasEntityPredictor:
    def __init__(self, catalog: EntityCatalog, max_ngram: int = 8):
        self.catalog = catalog
        self.max_ngram = max_ngram

    def predict(
        self,
        question: str,
        top_entities: int = 5,
        top_starts: int = 3,
    ) -> List[LinkedEntity]:
        tokens = normalize_text(question).split()
        candidates: Dict[str, LinkedEntity] = {}
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + self.max_ngram) + 1):
                mention = " ".join(tokens[start:end])
                entity_ids = self.catalog.alias_candidates(mention, limit=20)
                if not entity_ids:
                    continue
                base_score = len(mention) / max(len(normalize_text(question)), 1)
                for rank, entity_id in enumerate(entity_ids):
                    record = self.catalog.get(entity_id)
                    score = base_score + 1.0 / (rank + 1)
                    current = candidates.get(entity_id)
                    if current is None or score > current.link_score:
                        candidates[entity_id] = LinkedEntity(
                            entity_id=entity_id,
                            entity_name=record.name,
                            mention=mention,
                            link_score=score,
                            start_score=score,
                        )
        ranked = sorted(candidates.values(), key=lambda item: (-item.link_score, item.entity_id))[:top_entities]
        for item in sorted(ranked, key=lambda item: (-item.start_score, item.entity_id))[:top_starts]:
            item.is_start = True
        return ranked


class LinkerCheckpointPredictor:
    def __init__(
        self,
        checkpoint_dir: Path,
        device: str = "cuda",
        mention_threshold: float = 0.25,
        max_candidate_mentions: int = 10,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        with (checkpoint_dir / "training_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.catalog = EntityCatalog.from_files([checkpoint_dir / "entity_catalog.jsonl"], [])
        model_name = config["model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", use_fast=True)
        self.model = KGELQStartModel(
            model_name_or_path=model_name,
            entity_feature_size=int(config.get("entity_feature_size", self.catalog.feature_size)),
            relation_vocab_size=int(config.get("relation_vocab_size", 0)),
            share_encoders=bool(config.get("share_encoders", False)),
        )
        self.model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu"), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.max_question_len = int(config.get("max_question_len", 96))
        self.max_entity_len = int(config.get("max_entity_len", 160))
        self.max_span_len = int(config.get("max_span_len", 10))
        self.alias_candidate_limit = int(config.get("alias_candidate_limit", 50))
        self.mention_threshold = mention_threshold
        self.max_candidate_mentions = max_candidate_mentions

    @torch.no_grad()
    def predict(
        self,
        question: str,
        top_entities: int = 5,
        top_starts: int = 3,
    ) -> List[LinkedEntity]:
        q_inputs = self.tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=self.max_question_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = q_inputs.pop("offset_mapping")[0]
        q_inputs = {key: value.to(self.device) for key, value in q_inputs.items()}
        outputs = self.model.question_encoder(**q_inputs)
        q_hidden = outputs.last_hidden_state
        q_vec = F.normalize(mean_pool(q_hidden, q_inputs["attention_mask"]), p=2, dim=-1)[0]

        spans = self._enumerate_spans(offsets)
        if not spans:
            return []
        span_reprs, span_logits = self.model._encode_spans(q_hidden, [spans])
        span_probs = torch.sigmoid(span_logits[0]).detach().cpu()
        ranked_span_indices = [
            idx
            for idx in torch.argsort(span_probs, descending=True).tolist()
            if span_probs[idx].item() >= self.mention_threshold
        ][: self.max_candidate_mentions]
        if not ranked_span_indices:
            ranked_span_indices = torch.argsort(span_probs, descending=True).tolist()[: min(3, len(spans))]

        link_candidates: Dict[str, float] = {}
        mention_by_entity: Dict[str, str] = {}
        span_repr_by_entity: Dict[str, torch.Tensor] = {}
        for span_idx in ranked_span_indices:
            start_tok, end_tok = spans[span_idx]
            char_start = int(offsets[start_tok][0])
            char_end = int(offsets[end_tok][1])
            mention = question[char_start:char_end]
            candidate_ids = self.catalog.alias_candidates(mention, limit=self.alias_candidate_limit)
            if not candidate_ids:
                continue
            candidate_vecs, kept_ids = self._encode_entities(candidate_ids)
            if candidate_vecs.numel() == 0:
                continue
            mention_vec = span_reprs[0][span_idx].to(self.device)
            scores = torch.matmul(candidate_vecs, mention_vec) / math.sqrt(self.model.hidden_size)
            for entity_id, score in zip(kept_ids, scores.detach().cpu().tolist()):
                if score > link_candidates.get(entity_id, float("-inf")):
                    link_candidates[entity_id] = float(score)
                    mention_by_entity[entity_id] = mention
                    span_repr_by_entity[entity_id] = mention_vec

        if not link_candidates:
            return []
        candidate_ids = sorted(link_candidates, key=lambda eid: (-link_candidates[eid], eid))[: max(top_entities * 4, top_entities)]
        entity_vecs, kept_ids = self._encode_entities(candidate_ids)
        features = torch.tensor([self.catalog.get_features(eid) for eid in kept_ids], dtype=torch.float32, device=self.device)
        q_expand = q_vec.unsqueeze(0).expand_as(entity_vecs)
        link_priors = torch.tensor([link_candidates[eid] for eid in kept_ids], dtype=torch.float32, device=self.device).unsqueeze(-1)
        start_features = torch.cat(
            [q_expand, entity_vecs, q_expand * entity_vecs, torch.abs(q_expand - entity_vecs), link_priors, features],
            dim=-1,
        )
        start_scores = self.model.start_scorer(start_features).squeeze(-1).detach().cpu().tolist()

        linked = []
        for entity_id, start_score in zip(kept_ids, start_scores):
            record = self.catalog.get(entity_id)
            linked.append(
                LinkedEntity(
                    entity_id=entity_id,
                    entity_name=record.name,
                    mention=mention_by_entity.get(entity_id, record.name),
                    link_score=link_candidates[entity_id],
                    start_score=float(start_score),
                )
            )
        linked.sort(key=lambda item: (-item.link_score, -item.start_score, item.entity_id))
        linked = linked[:top_entities]
        for item in sorted(linked, key=lambda item: (-item.start_score, -item.link_score, item.entity_id))[:top_starts]:
            item.is_start = True
        return linked

    def _encode_entities(self, candidate_ids: Sequence[str]) -> Tuple[torch.Tensor, List[str]]:
        kept_ids = [entity_id for entity_id in candidate_ids if entity_id in self.catalog]
        if not kept_ids:
            return torch.zeros((0, self.model.hidden_size), device=self.device), []
        entity_inputs = self.tokenizer(
            [self.catalog.get_text(entity_id) for entity_id in kept_ids],
            padding=True,
            truncation=True,
            max_length=self.max_entity_len,
            return_tensors="pt",
        )
        entity_inputs = {key: value.to(self.device) for key, value in entity_inputs.items()}
        outputs = self.model.entity_encoder(**entity_inputs)
        vecs = F.normalize(mean_pool(outputs.last_hidden_state, entity_inputs["attention_mask"]), p=2, dim=-1)
        return vecs, kept_ids

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


def load_linked_entities(path: Path) -> List[LinkedEntity]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("linked_entities") or data.get("entities") or []
    output = []
    for item in data:
        output.append(
            LinkedEntity(
                entity_id=str(item.get("entity_id") or item.get("id") or item.get("mid")),
                entity_name=str(item.get("entity_name") or item.get("name") or item.get("entity_id")),
                mention=str(item.get("mention") or item.get("text") or item.get("entity_name") or ""),
                link_score=float(item.get("link_score") or item.get("score") or 0.0),
                start_score=float(item.get("start_score") or 0.0),
                is_start=bool(item.get("is_start") or item.get("start")),
            )
        )
    return output
