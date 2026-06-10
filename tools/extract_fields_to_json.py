#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List
# Example:
# python tools/extract_fields_to_json.py checkpoints/webqsp/experiment/eval_results/webqsp_checkpoint_eval/final_test_details.jsonl


FIELDS = ("id", "question", "prediction_raw", "gold_raw")


def extract_records(input_path: Path) -> List[Dict]:
    records = []
    with input_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            missing = [field for field in FIELDS if field not in item]
            if missing:
                raise KeyError(
                    f"Line {line_no} in {input_path} is missing fields: {', '.join(missing)}"
                )

            records.append({field: item[field] for field in FIELDS})

    return records


def candidate_eval_dirs(path: Path) -> List[Path]:
    return [
        path,
        path / "webqsp_checkpoint_eval",
        path / "webqsp-checkpoint",
        path / "eval_results" / "webqsp_checkpoint_eval",
        path / "eval_results" / "webqsp-checkpoint",
        path / "eval-results" / "webqsp_checkpoint_eval",
        path / "eval-results" / "webqsp-checkpoint",
    ]


def iter_jsonl_files(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Input file must be .jsonl: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = []
    for candidate in candidate_eval_dirs(input_path):
        if not candidate.is_dir():
            continue
        files = sorted(candidate.glob(pattern))
        if files:
            break

    if not files and recursive:
        files = sorted(input_path.rglob(pattern))

    return [path for path in files if path.is_file() and path.suffix == ".jsonl"]


def output_path_for(input_path: Path, output_dir: Path = None, suffix: str = "_extracted") -> Path:
    filename = f"{input_path.stem}{suffix}.json"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    return input_path.with_name(filename)


def convert_file(input_path: Path, output_path: Path, overwrite: bool) -> int:
    if output_path.exists() and not overwrite:
        print(f"Skip existing: {output_path}")
        return -1

    records = extract_records(input_path)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(records, outfile, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {output_path}")
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract selected fields from one JSONL file or all JSONL files in an experiment dir."
    )
    parser.add_argument(
        "input",
        help="Input JSONL file, eval dir, or experiment dir.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path for single-file mode. Defaults to <input_stem>_extracted.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch mode. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="JSONL filename glob for batch mode. Defaults to *.jsonl.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input dir if no JSONL files are found in known eval dirs.",
    )
    parser.add_argument(
        "--suffix",
        default="_extracted",
        help="Output filename suffix before .json. Defaults to _extracted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted JSON files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    jsonl_files = iter_jsonl_files(input_path, args.pattern, args.recursive)
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files matched {args.pattern} under {input_path}")

    if args.output and len(jsonl_files) > 1:
        raise ValueError("--output can only be used when converting one JSONL file. Use --output-dir for batch mode.")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    converted = 0
    skipped = 0
    for jsonl_file in jsonl_files:
        output_path = (
            Path(args.output).resolve()
            if args.output
            else output_path_for(jsonl_file, output_dir=output_dir, suffix=args.suffix)
        )
        count = convert_file(jsonl_file, output_path, overwrite=args.overwrite)
        if count < 0:
            skipped += 1
        else:
            converted += 1

    if len(jsonl_files) > 1:
        print(f"Done. converted={converted}, skipped={skipped}, total={len(jsonl_files)}")


if __name__ == "__main__":
    main()
