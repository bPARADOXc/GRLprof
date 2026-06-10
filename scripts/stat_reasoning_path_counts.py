#!/usr/bin/env python3
"""Report per-sample reasoning-path count distributions for merged datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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

COUNT_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("<=8", 0, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    (">256", 257, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count what percentage of samples have <=8, 9-16, 17-32, "
            "33-64, 65-128, 129-256, or >256 reasoning paths."
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


def bin_for_count(count: int) -> str:
    for label, min_value, max_value in COUNT_BINS:
        if count < min_value:
            continue
        if max_value is None or count <= max_value:
            return label
    raise ValueError(f"Could not bin count: {count}")


def summarize_samples(samples: Iterable[dict]) -> dict:
    bin_counts = {label: 0 for label, _, _ in COUNT_BINS}
    total_samples = 0
    total_paths = 0
    min_paths = None
    max_paths = None
    mismatched_num_reasoning_paths = 0

    for sample in samples:
        total_samples += 1
        paths = sample.get("reasoning_paths") or []
        path_count = len(paths) if isinstance(paths, list) else 0
        declared_count = sample.get("num_reasoning_paths")
        if isinstance(declared_count, int) and declared_count != path_count:
            mismatched_num_reasoning_paths += 1

        bin_counts[bin_for_count(path_count)] += 1
        total_paths += path_count
        min_paths = path_count if min_paths is None else min(min_paths, path_count)
        max_paths = path_count if max_paths is None else max(max_paths, path_count)

    average_paths = (total_paths / total_samples) if total_samples else 0.0
    distribution = [
        {
            "bucket": label,
            "count": count,
            "percent": (count / total_samples * 100.0) if total_samples else 0.0,
        }
        for label, count in bin_counts.items()
    ]

    return {
        "total_samples": total_samples,
        "total_paths": total_paths,
        "average_paths_per_sample": average_paths,
        "min_paths_per_sample": min_paths if min_paths is not None else 0,
        "max_paths_per_sample": max_paths if max_paths is not None else 0,
        "mismatched_num_reasoning_paths": mismatched_num_reasoning_paths,
        "count_distribution": distribution,
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
            f"total_paths={item['total_paths']} "
            f"avg_paths={item['average_paths_per_sample']:.2f} "
            f"min_paths={item['min_paths_per_sample']} "
            f"max_paths={item['max_paths_per_sample']} "
            f"mismatched_num_reasoning_paths={item['mismatched_num_reasoning_paths']}"
        )
        print("bucket\tcount\tpercent")
        for row in item["count_distribution"]:
            print(f"{row['bucket']}\t{row['count']}\t{row['percent']:.2f}%")


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
                "total_paths",
                "average_paths_per_sample",
                "min_paths_per_sample",
                "max_paths_per_sample",
                "mismatched_num_reasoning_paths",
                "bucket",
                "count",
                "percent",
            ],
        )
        writer.writeheader()
        for item in stats:
            for row in item["count_distribution"]:
                writer.writerow(
                    {
                        "dataset": item["dataset"],
                        "split": item["split"],
                        "root": item["root"],
                        "total_samples": item["total_samples"],
                        "total_paths": item["total_paths"],
                        "average_paths_per_sample": f"{item['average_paths_per_sample']:.6f}",
                        "min_paths_per_sample": item["min_paths_per_sample"],
                        "max_paths_per_sample": item["max_paths_per_sample"],
                        "mismatched_num_reasoning_paths": item["mismatched_num_reasoning_paths"],
                        "bucket": row["bucket"],
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
