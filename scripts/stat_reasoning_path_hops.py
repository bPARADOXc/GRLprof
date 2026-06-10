#!/usr/bin/env python3
"""Report reasoning-path hop distributions for merged WebQSP/CWQ splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_ROOTS = {
    "webqsp": Path("data/webqsp_may_merged"),
    "cwq": Path("data/cwq_may_merged"),
}

SPLIT_FILES = {
    "train": "train.json",
    "val": "validation.json",
    "validation": "validation.json",
    "test": "test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count what percentage of reasoning paths have 1 hop, 2 hops, etc. "
            "A path's hop count is len(path), where path is a list of triples."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("webqsp", "cwq", "all"),
        default="all",
        help="Dataset to summarize. Default reports both existing default roots.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Override merged JSON directory. Use with --dataset webqsp or --dataset cwq.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=("train", "validation", "test"),
        help="Splits to summarize.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write row-level statistics as CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full statistics as JSON.",
    )
    return parser.parse_args()


def canonical_split(split: str) -> str:
    return "validation" if split == "val" else split


def dataset_roots(dataset: str, input_root: Optional[Path]) -> Dict[str, Path]:
    if input_root is not None:
        if dataset == "all":
            raise ValueError("--input-root must be used with --dataset webqsp or --dataset cwq.")
        return {dataset: input_root}

    if dataset == "all":
        return {
            name: root
            for name, root in DEFAULT_ROOTS.items()
            if root.is_dir()
        }
    return {dataset: DEFAULT_ROOTS[dataset]}


def load_split(root: Path, split: str) -> List[dict]:
    split_file = root / SPLIT_FILES[split]
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {split_file}, got {type(payload).__name__}.")
    return payload


def summarize_samples(samples: Iterable[dict]) -> dict:
    hop_counts: Counter[int] = Counter()
    total_samples = 0
    samples_with_paths = 0
    empty_path_samples = 0
    invalid_paths = 0

    for sample in samples:
        total_samples += 1
        paths = sample.get("reasoning_paths") or []
        if paths:
            samples_with_paths += 1
        else:
            empty_path_samples += 1

        for path in paths:
            if not isinstance(path, list):
                invalid_paths += 1
                continue
            hop_counts[len(path)] += 1

    total_paths = sum(hop_counts.values())
    hop_distribution = [
        {
            "hop": hop,
            "count": count,
            "percent": (count / total_paths * 100.0) if total_paths else 0.0,
        }
        for hop, count in sorted(hop_counts.items())
    ]

    return {
        "total_samples": total_samples,
        "samples_with_paths": samples_with_paths,
        "empty_path_samples": empty_path_samples,
        "total_paths": total_paths,
        "invalid_paths": invalid_paths,
        "hop_distribution": hop_distribution,
    }


def collect_stats(args: argparse.Namespace) -> List[dict]:
    roots = dataset_roots(args.dataset, args.input_root)
    if not roots:
        raise FileNotFoundError("No default dataset roots found under data/webqsp_may_merged or data/cwq_may_merged.")

    stats = []
    for dataset_name, root in roots.items():
        for raw_split in args.splits:
            split = canonical_split(raw_split)
            samples = load_split(root, split)
            summary = summarize_samples(samples)
            stats.append(
                {
                    "dataset": dataset_name,
                    "root": str(root),
                    "split": split,
                    **summary,
                }
            )
    return stats


def print_stats(stats: List[dict]) -> None:
    for item in stats:
        print(
            f"\n[{item['dataset']}/{item['split']}] "
            f"samples={item['total_samples']} "
            f"samples_with_paths={item['samples_with_paths']} "
            f"empty_path_samples={item['empty_path_samples']} "
            f"total_paths={item['total_paths']} "
            f"invalid_paths={item['invalid_paths']}"
        )
        print("hop\tcount\tpercent")
        for row in item["hop_distribution"]:
            print(f"{row['hop']}\t{row['count']}\t{row['percent']:.2f}%")


def write_csv(path: Path, stats: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "root",
                "total_samples",
                "samples_with_paths",
                "empty_path_samples",
                "total_paths",
                "invalid_paths",
                "hop",
                "count",
                "percent",
            ],
        )
        writer.writeheader()
        for item in stats:
            for row in item["hop_distribution"]:
                writer.writerow(
                    {
                        "dataset": item["dataset"],
                        "split": item["split"],
                        "root": item["root"],
                        "total_samples": item["total_samples"],
                        "samples_with_paths": item["samples_with_paths"],
                        "empty_path_samples": item["empty_path_samples"],
                        "total_paths": item["total_paths"],
                        "invalid_paths": item["invalid_paths"],
                        "hop": row["hop"],
                        "count": row["count"],
                        "percent": f"{row['percent']:.6f}",
                    }
                )


def write_json(path: Path, stats: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    stats = collect_stats(args)
    print_stats(stats)
    if args.output_csv is not None:
        write_csv(args.output_csv, stats)
        print(f"\nwrote CSV to {args.output_csv}")
    if args.output_json is not None:
        write_json(args.output_json, stats)
        print(f"wrote JSON to {args.output_json}")


if __name__ == "__main__":
    main()
