from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from grlprof.entity_linking.catalog import EntityCatalog
from .entity_predictor import AliasEntityPredictor, LinkedEntity, LinkerCheckpointPredictor, load_linked_entities
from .freebase_kg import InMemoryFreebaseKG
from .path_recovery import recover_reasoning_paths
from .retriever_adapter import (
    TransformerTextEncoder,
    build_retriever_sample,
    load_entity_identifiers,
    load_subgraphrag_retriever,
    score_sample_triples,
)
from .rough_subgraph import RoughSubgraphSampler, estimate_difficulty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run question -> entity linking -> Freebase rough subgraph -> GRLprof scoring -> path recovery."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--sample-id", default="interactive-0")
    parser.add_argument("--freebase-triples", type=Path, required=True)
    parser.add_argument("--freebase-delimiter", default="\t")
    parser.add_argument("--freebase-cvt-entities", type=Path, default=None)
    parser.add_argument("--freebase-entity-names", type=Path, default=None)
    parser.add_argument("--max-freebase-triples", type=int, default=-1)

    parser.add_argument("--linker-checkpoint", type=Path, default=None)
    parser.add_argument("--entity-catalog", type=Path, nargs="*", default=[])
    parser.add_argument("--alias-file", type=Path, nargs="*", default=[])
    parser.add_argument("--linked-entities-json", type=Path, default=None)
    parser.add_argument("--start-entities", default="", help="Comma-separated entity ids/names. Overrides predicted starts.")
    parser.add_argument("--question-entities", default="", help="Comma-separated entity ids/names. Added to linked entities.")
    parser.add_argument("--top-question-entities", type=int, default=5)
    parser.add_argument("--top-start-entities", type=int, default=3)

    parser.add_argument("--retriever-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--text-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    parser.add_argument("--entity-identifiers", type=Path, default=None)
    parser.add_argument("--retriever-top-k", type=int, default=500)

    parser.add_argument("--max-physical-hop", type=int, default=0)
    parser.add_argument("--relation-top-m", type=int, default=0)
    parser.add_argument("--rough-max-triples", type=int, default=0)
    parser.add_argument("--score-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-prefix-expansions", type=int, default=3)
    parser.add_argument("--max-reasoning-paths", type=int, default=256)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-pth", type=Path, default=None)
    return parser.parse_args()


def split_entities(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def predict_entities(args: argparse.Namespace) -> List[LinkedEntity]:
    manual_question_entities = split_entities(args.question_entities)
    manual_start_entities = split_entities(args.start_entities)

    if args.linked_entities_json is not None:
        linked = load_linked_entities(args.linked_entities_json)
    elif args.linker_checkpoint is not None:
        predictor = LinkerCheckpointPredictor(args.linker_checkpoint, device=args.device)
        linked = predictor.predict(
            args.question,
            top_entities=args.top_question_entities,
            top_starts=args.top_start_entities,
        )
    else:
        catalog = EntityCatalog.from_files(args.entity_catalog, args.alias_file)
        if len(catalog) == 0:
            for entity in manual_question_entities + manual_start_entities:
                catalog.ensure_entity(entity, entity)
        predictor = AliasEntityPredictor(catalog)
        linked = predictor.predict(
            args.question,
            top_entities=args.top_question_entities,
            top_starts=args.top_start_entities,
        )

    by_id = {item.entity_id: item for item in linked}
    for entity in manual_question_entities:
        by_id.setdefault(
            entity,
            LinkedEntity(entity_id=entity, entity_name=entity, mention=entity, link_score=1.0),
        )
    for entity in manual_start_entities:
        item = by_id.setdefault(
            entity,
            LinkedEntity(entity_id=entity, entity_name=entity, mention=entity, link_score=1.0),
        )
        item.is_start = True
        item.start_score = max(item.start_score, 1.0)
    protected_ids = set(manual_question_entities) | set(manual_start_entities)
    linked = list(by_id.values())
    if linked and not any(item.is_start for item in linked):
        for item in sorted(linked, key=lambda ent: (-ent.start_score, -ent.link_score, ent.entity_id))[
            : args.top_start_entities
        ]:
            item.is_start = True
    ranked = sorted(linked, key=lambda item: (-item.link_score, -item.start_score, item.entity_id))
    selected = ranked[: args.top_question_entities]
    selected_ids = {item.entity_id for item in selected}
    for item in ranked:
        if item.entity_id in protected_ids and item.entity_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.entity_id)
    return selected


def run_pipeline(args: argparse.Namespace) -> dict:
    linked_entities = predict_entities(args)
    question_entities = [item.entity_id for item in linked_entities]
    start_entities = [item.entity_id for item in linked_entities if item.is_start]
    if not question_entities:
        raise RuntimeError("No question entities were predicted or provided.")
    if not start_entities:
        start_entities = question_entities[:1]

    kg = InMemoryFreebaseKG.from_path(
        args.freebase_triples,
        delimiter=args.freebase_delimiter,
        cvt_path=args.freebase_cvt_entities,
        entity_names_path=args.freebase_entity_names,
        max_triples=args.max_freebase_triples,
    )
    difficulty = estimate_difficulty(args.question, len(question_entities))
    sampler = RoughSubgraphSampler(kg)
    rough = sampler.build(
        question=args.question,
        start_entities=start_entities,
        question_entities=question_entities,
        difficulty=difficulty,
        max_triples=args.rough_max_triples or None,
        max_physical_hop=args.max_physical_hop or None,
        relation_top_m=args.relation_top_m or None,
    )

    scored_triples = []
    if not args.skip_scoring:
        if args.retriever_checkpoint is None:
            raise RuntimeError("--retriever-checkpoint is required unless --skip-scoring is set.")
        repo_root = Path(__file__).resolve().parents[2]
        text_encoder = TransformerTextEncoder(args.text_encoder, device=args.device)
        entity_identifiers = load_entity_identifiers(args.entity_identifiers)
        sample = build_retriever_sample(
            sample_id=args.sample_id,
            question=args.question,
            graph=rough.triples,
            q_entities=start_entities,
            text_encoder=text_encoder,
            entity_identifiers=entity_identifiers,
            answers=[],
        )
        model, _, runtime_device = load_subgraphrag_retriever(args.retriever_checkpoint, repo_root, args.device)
        scored_triples = score_sample_triples(model, sample, runtime_device, top_k=args.retriever_top_k)
    else:
        scored_triples = [(triple.h, triple.r, triple.t, 1.0) for triple in rough.triples[: args.retriever_top_k]]

    reasoning_paths = recover_reasoning_paths(
        graph_triples=rough.triples,
        scored_triples=scored_triples,
        q_entities=start_entities,
        score_ratio=args.score_ratio,
        max_prefix_expansions=args.max_prefix_expansions,
        max_paths=args.max_reasoning_paths,
    )

    retrieval_record = {
        "question": args.question,
        "scored_triples": scored_triples,
        "q_entity": start_entities,
        "q_entity_in_graph": [entity for entity in start_entities if kg.has_entity(entity)],
        "a_entity": [],
        "a_entity_in_graph": [],
        "max_path_length": max((len(path) for path in reasoning_paths), default=None),
        "target_relevant_triples": [],
    }
    output = {
        "id": args.sample_id,
        "question": args.question,
        "linked_entities": [item.to_json() for item in linked_entities],
        "difficulty": {
            "name": rough.difficulty.name,
            "hop_level": rough.difficulty.hop_level,
            "constraint_level": rough.difficulty.constraint_level,
            "aggregation": rough.difficulty.aggregation,
            "comparison": rough.difficulty.comparison,
            "temporal": rough.difficulty.temporal,
            "multi_topic": rough.difficulty.multi_topic,
        },
        "rough_subgraph": {
            "num_triples": len(rough.triples),
            "start_entities": rough.start_entities,
            "question_entities": rough.question_entities,
            "triples": rough.graph_as_lists(),
        },
        "retrieval_record": retrieval_record,
        "num_reasoning_paths": len(reasoning_paths),
        "reasoning_paths": reasoning_paths,
    }
    return output


def main() -> None:
    args = parse_args()
    output = run_pipeline(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if args.output_pth is not None:
        args.output_pth.parent.mkdir(parents=True, exist_ok=True)
        torch.save({args.sample_id: output["retrieval_record"]}, args.output_pth)
    print(f"Wrote {args.output_json}")
    if args.output_pth is not None:
        print(f"Wrote {args.output_pth}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
