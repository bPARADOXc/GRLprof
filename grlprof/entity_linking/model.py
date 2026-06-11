from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return pooled / denom


class KGELQStartModel(nn.Module):
    """ELQ-style entity linker with a KG-aware start-entity ranking head.

    The model deliberately keeps retrieval exact and local during training:
    span vectors are scored against candidate entity vectors with a dot product.
    At large Freebase scale the same entity encoder can be used to precompute
    vectors for FAISS or another ANN backend.
    """

    def __init__(
        self,
        model_name_or_path: str,
        entity_feature_size: int = 4,
        relation_vocab_size: int = 0,
        share_encoders: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.question_encoder = AutoModel.from_pretrained(model_name_or_path)
        if share_encoders:
            self.entity_encoder = self.question_encoder
        else:
            self.entity_encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = int(self.question_encoder.config.hidden_size)
        self.hidden_size = hidden_size
        self.entity_feature_size = entity_feature_size
        self.relation_vocab_size = relation_vocab_size
        self.share_encoders = share_encoders

        self.span_scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        start_input_size = hidden_size * 4 + 1 + entity_feature_size
        self.start_scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(start_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.path_head = (
            nn.Linear(hidden_size, relation_vocab_size)
            if relation_vocab_size > 0
            else None
        )

    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        question_inputs = {
            key: value.to(device)
            for key, value in batch["question_inputs"].items()
        }
        entity_inputs = {
            key: value.to(device)
            for key, value in batch["entity_inputs"].items()
        }
        entity_features = batch["entity_features"].to(device)

        q_outputs = self.question_encoder(**question_inputs)
        q_hidden = q_outputs.last_hidden_state
        q_cls = mean_pool(q_hidden, question_inputs["attention_mask"])

        e_outputs = self.entity_encoder(**entity_inputs)
        entity_vecs = mean_pool(e_outputs.last_hidden_state, entity_inputs["attention_mask"])
        entity_vecs = F.normalize(entity_vecs, p=2, dim=-1)
        q_norm = F.normalize(q_cls, p=2, dim=-1)

        entity_index = {entity_id: idx for idx, entity_id in enumerate(batch["entity_ids"])}
        span_reprs, span_logits = self._encode_spans(q_hidden, batch["spans"])

        losses = {}
        metrics = {}

        mention_loss = self._mention_loss(span_logits, batch["span_labels"], device)
        if mention_loss is not None:
            losses["mention"] = mention_loss
            metrics.update(self._mention_metrics(span_logits, batch["span_labels"], device))

        link_loss, link_metrics, link_score_cache = self._link_loss_and_metrics(
            span_reprs=span_reprs,
            entity_vecs=entity_vecs,
            entity_index=entity_index,
            link_groups=batch["link_groups"],
            device=device,
        )
        if link_loss is not None:
            losses["link"] = link_loss
            metrics.update(link_metrics)

        start_loss, start_metrics = self._start_loss_and_metrics(
            q_vecs=q_norm,
            entity_vecs=entity_vecs,
            entity_features=entity_features,
            entity_index=entity_index,
            start_groups=batch["start_groups"],
            link_score_cache=link_score_cache,
            device=device,
        )
        if start_loss is not None:
            losses["start"] = start_loss
            metrics.update(start_metrics)

        path_labels = batch.get("path_labels")
        if self.path_head is not None and path_labels is not None:
            path_labels = path_labels.to(device)
            path_logits = self.path_head(q_cls)
            losses["path"] = F.binary_cross_entropy_with_logits(path_logits, path_labels)

        if losses:
            total_loss = sum(losses.values())
        else:
            total_loss = q_hidden.new_tensor(0.0)

        return {
            "loss": total_loss,
            "losses": losses,
            "metrics": metrics,
        }

    def _encode_spans(
        self,
        q_hidden: torch.Tensor,
        batch_spans: List[List[Tuple[int, int]]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        span_reprs: List[torch.Tensor] = []
        span_logits: List[torch.Tensor] = []
        for batch_idx, spans in enumerate(batch_spans):
            sample_reprs = []
            for start, end in spans:
                start_vec = q_hidden[batch_idx, start]
                end_vec = q_hidden[batch_idx, end]
                mean_vec = q_hidden[batch_idx, start : end + 1].mean(dim=0)
                sample_reprs.append(torch.cat([start_vec, end_vec, mean_vec], dim=-1))
            if sample_reprs:
                reprs = torch.stack(sample_reprs, dim=0)
                logits = self.span_scorer(reprs).squeeze(-1)
                mention_reprs = reprs[:, -self.hidden_size :]
            else:
                reprs = q_hidden.new_zeros((0, self.hidden_size * 3))
                logits = q_hidden.new_zeros((0,))
                mention_reprs = q_hidden.new_zeros((0, self.hidden_size))
            span_reprs.append(F.normalize(mention_reprs, p=2, dim=-1))
            span_logits.append(logits)
        return span_reprs, span_logits

    @staticmethod
    def _mention_loss(
        span_logits: List[torch.Tensor],
        span_labels: List[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not span_logits:
            return None
        logits = torch.cat([item for item in span_logits if item.numel() > 0], dim=0)
        if logits.numel() == 0:
            return None
        labels = torch.cat([item.to(device) for item in span_labels if item.numel() > 0], dim=0)
        positives = labels.sum()
        negatives = labels.numel() - positives
        pos_weight = (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=100.0)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    @staticmethod
    def _mention_metrics(
        span_logits: List[torch.Tensor],
        span_labels: List[torch.Tensor],
        device: torch.device,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        tp = fp = fn = 0.0
        for logits, labels in zip(span_logits, span_labels):
            if logits.numel() == 0:
                continue
            pred = (torch.sigmoid(logits.detach()) >= threshold).to(torch.bool)
            gold = labels.to(device).to(torch.bool)
            tp += float((pred & gold).sum().item())
            fp += float((pred & ~gold).sum().item())
            fn += float((~pred & gold).sum().item())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "mention_precision": precision,
            "mention_recall": recall,
            "mention_f1": f1,
        }

    def _link_loss_and_metrics(
        self,
        span_reprs: List[torch.Tensor],
        entity_vecs: torch.Tensor,
        entity_index: Dict[str, int],
        link_groups: List[dict],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float], Dict[Tuple[int, str], torch.Tensor]]:
        losses = []
        top1 = 0.0
        top5 = 0.0
        count = 0.0
        link_score_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        for group in link_groups:
            batch_idx = int(group["batch_index"])
            span_index = int(group["span_index"])
            if span_index >= span_reprs[batch_idx].shape[0]:
                continue
            candidate_indices = [
                entity_index[entity_id]
                for entity_id in group["candidate_ids"]
                if entity_id in entity_index
            ]
            if not candidate_indices:
                continue
            candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
            candidate_vecs = entity_vecs.index_select(0, candidate_tensor)
            mention_vec = span_reprs[batch_idx][span_index]
            scores = torch.matmul(candidate_vecs, mention_vec) / math.sqrt(self.hidden_size)

            positive_positions = [
                pos
                for pos, entity_id in enumerate(group["candidate_ids"])
                if entity_id in set(group["positive_ids"]) and entity_id in entity_index
            ]
            if not positive_positions:
                continue
            pos_tensor = torch.tensor(positive_positions, dtype=torch.long, device=device)
            losses.append(-(torch.logsumexp(scores.index_select(0, pos_tensor), dim=0) - torch.logsumexp(scores, dim=0)))

            for pos, entity_id in enumerate(group["candidate_ids"]):
                if entity_id not in entity_index:
                    continue
                key = (batch_idx, entity_id)
                current = scores[pos]
                cached = link_score_cache.get(key)
                if cached is None:
                    link_score_cache[key] = current
                else:
                    link_score_cache[key] = torch.maximum(cached, current)

            with torch.no_grad():
                ranked = torch.argsort(scores, descending=True)
                pos_set = set(positive_positions)
                top1 += float(int(ranked[0].item() in pos_set))
                topk = ranked[: min(5, ranked.numel())].tolist()
                top5 += float(any(idx in pos_set for idx in topk))
                count += 1.0

        if not losses:
            return None, {}, link_score_cache
        metrics = {
            "link_top1": top1 / max(count, 1.0),
            "link_top5": top5 / max(count, 1.0),
        }
        return torch.stack(losses).mean(), metrics, link_score_cache

    def _start_loss_and_metrics(
        self,
        q_vecs: torch.Tensor,
        entity_vecs: torch.Tensor,
        entity_features: torch.Tensor,
        entity_index: Dict[str, int],
        start_groups: List[dict],
        link_score_cache: Dict[Tuple[int, str], torch.Tensor],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        losses = []
        top1 = 0.0
        top3 = 0.0
        count = 0.0

        for group in start_groups:
            batch_idx = int(group["batch_index"])
            candidate_ids = [entity_id for entity_id in group["candidate_ids"] if entity_id in entity_index]
            if not candidate_ids:
                continue
            candidate_indices = torch.tensor(
                [entity_index[entity_id] for entity_id in candidate_ids],
                dtype=torch.long,
                device=device,
            )
            candidates = entity_vecs.index_select(0, candidate_indices)
            features = entity_features.index_select(0, candidate_indices)
            q_vec = q_vecs[batch_idx].unsqueeze(0).expand_as(candidates)
            link_priors = torch.stack(
                [
                    link_score_cache.get((batch_idx, entity_id), q_vecs.new_tensor(0.0))
                    for entity_id in candidate_ids
                ],
                dim=0,
            ).unsqueeze(-1)
            start_features = torch.cat(
                [
                    q_vec,
                    candidates,
                    q_vec * candidates,
                    torch.abs(q_vec - candidates),
                    link_priors,
                    features,
                ],
                dim=-1,
            )
            scores = self.start_scorer(start_features).squeeze(-1)
            positive_positions = [
                pos
                for pos, entity_id in enumerate(candidate_ids)
                if entity_id in set(group["positive_ids"])
            ]
            if not positive_positions:
                continue
            pos_tensor = torch.tensor(positive_positions, dtype=torch.long, device=device)
            losses.append(-(torch.logsumexp(scores.index_select(0, pos_tensor), dim=0) - torch.logsumexp(scores, dim=0)))

            with torch.no_grad():
                ranked = torch.argsort(scores, descending=True)
                pos_set = set(positive_positions)
                top1 += float(int(ranked[0].item() in pos_set))
                topk = ranked[: min(3, ranked.numel())].tolist()
                top3 += float(any(idx in pos_set for idx in topk))
                count += 1.0

        if not losses:
            return None, {}
        metrics = {
            "start_top1": top1 / max(count, 1.0),
            "start_top3": top3 / max(count, 1.0),
        }
        return torch.stack(losses).mean(), metrics
