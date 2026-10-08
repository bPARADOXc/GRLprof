#!/usr/bin/env python3
"""Validate merged KGQA data and path embeddings before adapter runs.

This preflight intentionally starts at the SubgraphRAG output boundary.  It
does not invoke entity linking or candidate-subgraph acquisition.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embedding_config import (  # noqa: E402
    embedding_filename,
    embedding_hidden_size,
    known_embedding_aliases,
)


REQUIRED_FIELDS = {
    "id",
    "question",
    "answer",
    "q_entity",
    "a_entity",
    "reasoning_paths",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check merged SubgraphRAG-to-GRLprof JSON files, reasoning paths, "
            "and cached path embeddings."
        )
    )
    parser.add_argument("--dataset", required=True, choices=("webqsp", "cwq"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--embedding-type",
        default="Alibaba-NLP/gte-modernbert-base",
        choices=known_embedding_aliases(),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation", "test"),
        choices=("train", "validation", "test"),
    )
    parser.add_argument(
        "--skip-embedding-contents",
        action="store_true",
        help="Check embedding filenames only; do not load and inspect .pt contents.",
    )
    parser.add_argument(
        "--require-all-embeddings",
        action="store_true",
        help="Fail even when the only missing embedding IDs are samples with no paths.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list, got {type(value).__name__}")
    return value


def valid_triple(triple: Any) -> bool:
    return (
        isinstance(triple, (list, tuple))
        and len(triple) >= 3
        and all(isinstance(part, str) for part in triple[:3])
    )


def inspect_json(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_json_list(path)
    missing_fields = []
    invalid_paths = []
    count_mismatches = []
    empty_path_ids = []
    ids = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            missing_fields.append({"index": index, "fields": ["<record is not an object>"]})
            continue
        sample_id = record.get("id", f"<index:{index}>")
        ids.append(sample_id)
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            missing_fields.append({"id": sample_id, "fields": missing})

        paths = record.get("reasoning_paths")
        if not isinstance(paths, list):
            invalid_paths.append(sample_id)
            continue
        if not paths:
            empty_path_ids.append(sample_id)
        elif any(
            not isinstance(reasoning_path, list)
            or not reasoning_path
            or any(not valid_triple(triple) for triple in reasoning_path)
            for reasoning_path in paths
        ):
            invalid_paths.append(sample_id)

        declared = record.get("num_reasoning_paths")
        if declared is not None and declared != len(paths):
            count_mismatches.append(sample_id)

    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    summary = {
        "split": split,
        "json_file": str(path),
        "records": len(records),
        "unique_ids": len(set(ids)),
        "missing_required_fields": len(missing_fields),
        "invalid_reasoning_paths": len(invalid_paths),
        "empty_reasoning_paths": len(empty_path_ids),
        "num_reasoning_paths_mismatches": len(count_mismatches),
        "duplicate_ids": len(duplicate_ids),
        "empty_path_ids_preview": empty_path_ids[:10],
        "_empty_path_ids": empty_path_ids,
        "errors": [],
        "warnings": [],
    }

    if missing_fields:
        summary["errors"].append(f"{len(missing_fields)} records miss required fields")
    if invalid_paths:
        summary["errors"].append(f"{len(invalid_paths)} records contain invalid reasoning paths")
    if duplicate_ids:
        summary["errors"].append(f"{len(duplicate_ids)} duplicate sample IDs")
    if split in {"train", "validation"} and empty_path_ids:
        summary["errors"].append(
            f"{len(empty_path_ids)} {split} records have no reasoning paths"
        )
    elif empty_path_ids:
        summary["warnings"].append(
            f"{len(empty_path_ids)} {split} records have no reasoning paths"
        )
    if count_mismatches:
        summary["warnings"].append(
            f"{len(count_mismatches)} num_reasoning_paths values do not match the list length"
        )
    return summary, records


def torch_load(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect embedding contents") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_embeddings(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    embedding_path: Path,
    embedding_type: str,
    skip_contents: bool,
    require_all: bool,
) -> None:
    empty_path_ids = set(summary.pop("_empty_path_ids", []))
    summary["embedding_file"] = str(embedding_path)
    if not embedding_path.is_file():
        summary["errors"].append(f"missing embedding file: {embedding_path}")
        return
    summary["embedding_bytes"] = embedding_path.stat().st_size
    if skip_contents:
        return

    embeddings = torch_load(embedding_path)
    if not isinstance(embeddings, dict):
        summary["errors"].append(
            f"embedding file must contain a dict, got {type(embeddings).__name__}"
        )
        return

    json_ids = {record.get("id") for record in records if isinstance(record, dict)}
    embedding_ids = set(embeddings)
    missing = sorted(json_ids - embedding_ids)
    extra = sorted(embedding_ids - json_ids)
    unexplained_missing = sorted(set(missing) - empty_path_ids)

    summary["embedding_ids"] = len(embedding_ids)
    summary["missing_embedding_ids"] = len(missing)
    summary["extra_embedding_ids"] = len(extra)
    summary["unexplained_missing_embedding_ids"] = len(unexplained_missing)

    if unexplained_missing:
        summary["errors"].append(
            f"{len(unexplained_missing)} samples with reasoning paths have no embedding"
        )
    elif missing:
        message = f"{len(missing)} samples without paths have no embedding"
        if require_all:
            summary["errors"].append(message)
        else:
            summary["warnings"].append(message)
    if extra:
        summary["errors"].append(f"{len(extra)} embedding IDs are absent from JSON")

    expected_hidden = embedding_hidden_size(embedding_type)
    checked_paths = 0
    malformed_embeddings = 0
    for sample_paths in embeddings.values():
        if not isinstance(sample_paths, list):
            malformed_embeddings += 1
            continue
        for reasoning_path in sample_paths:
            if not isinstance(reasoning_path, list) or not reasoning_path:
                malformed_embeddings += 1
                continue
            checked_paths += 1
            for triple in reasoning_path:
                shape = tuple(getattr(triple, "shape", ()))
                if shape != (3, expected_hidden):
                    malformed_embeddings += 1
                    break
            if checked_paths >= 100:
                break
        if checked_paths >= 100:
            break
    summary["sampled_embedding_paths"] = checked_paths
    summary["malformed_sampled_embeddings"] = malformed_embeddings
    if malformed_embeddings:
        summary["errors"].append(
            f"{malformed_embeddings} sampled embedding paths have an invalid structure"
        )


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    result = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "embedding_type": args.embedding_type,
        "splits": [],
    }

    for split in args.splits:
        json_path = data_dir / f"{split}.json"
        if not json_path.is_file():
            result["splits"].append(
                {
                    "split": split,
                    "json_file": str(json_path),
                    "errors": [f"missing JSON file: {json_path}"],
                    "warnings": [],
                }
            )
            continue
        try:
            summary, records = inspect_json(json_path, split)
            emb_path = data_dir / embedding_filename(
                args.embedding_type, split, dataset=args.dataset
            )
            inspect_embeddings(
                summary,
                records,
                emb_path,
                args.embedding_type,
                args.skip_embedding_contents,
                args.require_all_embeddings,
            )
        except Exception as exc:  # keep all split diagnostics in one report
            summary = {
                "split": split,
                "json_file": str(json_path),
                "errors": [f"{type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        result["splits"].append(summary)

    error_count = sum(len(item.get("errors", [])) for item in result["splits"])
    warning_count = sum(len(item.get("warnings", [])) for item in result["splits"])
    result["ok"] = error_count == 0
    result["error_count"] = error_count
    result["warning_count"] = warning_count

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
