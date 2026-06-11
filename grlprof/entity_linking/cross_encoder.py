from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel

from .catalog import EntityCatalog, unique_keep_order
from .data import KGQALinkerDataset, LinkerExample


@dataclass
class CrossEncoderPair:
    qid: str
    question: str
    mention: str
    entity_id: str
    entity_text: str
    label: float


class CrossEncoderPairDataset(Dataset):
    def __init__(
        self,
        base_dataset: KGQALinkerDataset,
        catalog: EntityCatalog,
        negatives_per_positive: int = 15,
        alias_candidate_limit: int = 50,
        seed: int = 0,
    ):
        self.catalog = catalog
        self.random = random.Random(seed)
        self.all_entity_ids = list(catalog.iter_ids())
        self.pairs = self._build_pairs(
            base_dataset.examples,
            negatives_per_positive=negatives_per_positive,
            alias_candidate_limit=alias_candidate_limit,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> CrossEncoderPair:
        return self.pairs[index]

    def _build_pairs(
        self,
        examples: Sequence[LinkerExample],
        negatives_per_positive: int,
        alias_candidate_limit: int,
    ) -> List[CrossEncoderPair]:
        output: List[CrossEncoderPair] = []
        for example in examples:
            for mention in example.mentions:
                if mention.entity_id not in self.catalog:
                    continue
                candidate_ids = [mention.entity_id]
                for alias in (mention.text, mention.entity_name):
                    candidate_ids.extend(self.catalog.alias_candidates(alias, limit=alias_candidate_limit))
                candidate_ids = unique_keep_order(
                    entity_id for entity_id in candidate_ids if entity_id in self.catalog
                )
                while len(candidate_ids) < 1 + negatives_per_positive and self.all_entity_ids:
                    sampled = self.random.choice(self.all_entity_ids)
                    if sampled not in candidate_ids:
                        candidate_ids.append(sampled)
                for entity_id in candidate_ids:
                    output.append(
                        CrossEncoderPair(
                            qid=example.qid,
                            question=example.question,
                            mention=mention.text,
                            entity_id=entity_id,
                            entity_text=self.catalog.get_text(entity_id),
                            label=1.0 if entity_id == mention.entity_id else 0.0,
                        )
                    )
        return output


class CrossEncoderCollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, pairs: Sequence[CrossEncoderPair]) -> dict:
        left = [pair.question for pair in pairs]
        right = [
            f"[MENTION] {pair.mention} [ENTITY] {pair.entity_text}"
            for pair in pairs
        ]
        inputs = self.tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.float32)
        return {"inputs": inputs, "labels": labels, "pairs": pairs}


class CrossEncoderReranker(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
        labels = batch.get("labels")
        outputs = self.encoder(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls).squeeze(-1)
        result = {"logits": logits}
        if labels is not None:
            labels = labels.to(device)
            positives = labels.sum()
            negatives = labels.numel() - positives
            pos_weight = (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=100.0)
            result["loss"] = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
                result["accuracy"] = (predictions == labels).to(torch.float32).mean()
        return result
