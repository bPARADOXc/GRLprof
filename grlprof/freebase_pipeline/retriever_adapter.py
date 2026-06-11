from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .freebase_kg import Triple


class TransformerTextEncoder:
    """Text encoder compatible with GRLprof retriever embeddings."""

    def __init__(
        self,
        model_name_or_path: str = "Alibaba-NLP/gte-large-en-v1.5",
        device: str = "cuda",
        normalize: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.normalize = normalize
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            hidden_size = int(self.model.config.hidden_size)
            return torch.zeros((0, hidden_size), dtype=torch.float32)
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded).last_hidden_state
            pooled = outputs[:, 0]
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu())
        return torch.cat(chunks, dim=0)

    def __call__(self, question: str, text_entity_list: Sequence[str], relation_list: Sequence[str]):
        return self.embed([question]), self.embed(text_entity_list), self.embed(relation_list)


def load_entity_identifiers(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_retriever_sample(
    sample_id: str,
    question: str,
    graph: Sequence[Triple],
    q_entities: Sequence[str],
    text_encoder: TransformerTextEncoder,
    entity_identifiers: Optional[Iterable[str]] = None,
    answers: Optional[Sequence[str]] = None,
) -> dict:
    entity_identifier_set = set(entity_identifiers or [])
    all_entities = sorted({triple.h for triple in graph} | {triple.t for triple in graph})
    text_entity_list = [entity for entity in all_entities if entity not in entity_identifier_set]
    non_text_entity_list = [entity for entity in all_entities if entity in entity_identifier_set]
    entity2id = {entity: idx for idx, entity in enumerate(text_entity_list)}
    offset = len(entity2id)
    for idx, entity in enumerate(non_text_entity_list):
        entity2id[entity] = offset + idx

    relation_list = sorted({triple.r for triple in graph})
    rel2id = {relation: idx for idx, relation in enumerate(relation_list)}

    h_id_list = [entity2id[triple.h] for triple in graph]
    r_id_list = [rel2id[triple.r] for triple in graph]
    t_id_list = [entity2id[triple.t] for triple in graph]
    q_entity_id_list = [entity2id[entity] for entity in q_entities if entity in entity2id]
    a_entity_id_list = [entity2id[entity] for entity in (answers or []) if entity in entity2id]

    q_emb, entity_embs, relation_embs = text_encoder(question, text_entity_list, relation_list)
    num_entities = len(text_entity_list) + len(non_text_entity_list)
    topic_entity_mask = torch.zeros(num_entities)
    if q_entity_id_list:
        topic_entity_mask[q_entity_id_list] = 1.0
    topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2).float()

    return {
        "id": sample_id,
        "question": question,
        "graph": [triple.as_list() for triple in graph],
        "q_entity": list(q_entities),
        "q_entity_id_list": q_entity_id_list,
        "text_entity_list": text_entity_list,
        "non_text_entity_list": non_text_entity_list,
        "relation_list": relation_list,
        "h_id_list": h_id_list,
        "r_id_list": r_id_list,
        "t_id_list": t_id_list,
        "a_entity": list(answers or []),
        "a_entity_id_list": a_entity_id_list,
        "q_emb": q_emb,
        "entity_embs": entity_embs,
        "relation_embs": relation_embs,
        "topic_entity_one_hot": topic_entity_one_hot,
        "target_triple_probs": torch.zeros(len(graph)),
    }


def load_subgraphrag_retriever(checkpoint_path: Path, repo_root: Path, device: str):
    retrieve_root = repo_root / "retrieval"
    if str(retrieve_root) not in sys.path:
        sys.path.insert(0, str(retrieve_root))
    from src.model.retriever import Retriever

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    emb_size = _infer_emb_size_from_state_dict(checkpoint["model_state_dict"])
    model = Retriever(emb_size, **config["retriever"])
    model.load_state_dict(checkpoint["model_state_dict"])
    runtime_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(runtime_device)
    model.eval()
    return model, config, runtime_device


def _infer_emb_size_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("pred.0.weight")
    if weight is None:
        for key, value in state_dict.items():
            if key.endswith("pred.0.weight"):
                weight = value
                break
    if weight is None:
        raise ValueError("Could not infer retriever embedding size from checkpoint.")
    # pred_in_size = 4*emb + topic/DDE features. The hidden layer output equals emb_size.
    return int(weight.shape[0])


@torch.no_grad()
def score_sample_triples(
    model,
    sample: dict,
    device: torch.device,
    top_k: int = 500,
) -> List[Tuple[str, str, str, float]]:
    h_id_tensor = torch.tensor(sample["h_id_list"], dtype=torch.long, device=device)
    r_id_tensor = torch.tensor(sample["r_id_list"], dtype=torch.long, device=device)
    t_id_tensor = torch.tensor(sample["t_id_list"], dtype=torch.long, device=device)
    q_emb = sample["q_emb"].to(device)
    entity_embs = sample["entity_embs"].to(device)
    relation_embs = sample["relation_embs"].to(device)
    topic_entity_one_hot = sample["topic_entity_one_hot"].to(device)

    if len(h_id_tensor) == 0:
        return []
    logits = model(
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        len(sample["non_text_entity_list"]),
        relation_embs,
        topic_entity_one_hot,
    )
    scores = torch.sigmoid(logits).reshape(-1)
    top_count = min(max(top_k, 1), len(scores))
    values, indices = torch.topk(scores, top_count)
    entity_list = sample["text_entity_list"] + sample["non_text_entity_list"]
    relation_list = sample["relation_list"]
    scored = []
    for score, triple_idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        scored.append(
            (
                entity_list[sample["h_id_list"][triple_idx]],
                relation_list[sample["r_id_list"][triple_idx]],
                entity_list[sample["t_id_list"][triple_idx]],
                float(score),
            )
        )
    return scored
