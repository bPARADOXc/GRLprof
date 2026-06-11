from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .catalog import EntityCatalog
from .cross_encoder import CrossEncoderCollator, CrossEncoderPairDataset, CrossEncoderReranker
from .data import KGQALinkerDataset, build_catalog_from_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cross-encoder entity reranker for KGQA entity linking."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default="roberta-base")
    parser.add_argument("--entity-catalog", type=Path, nargs="*", default=[])
    parser.add_argument("--alias-file", type=Path, nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--negatives-per-positive", type=int, default=15)
    parser.add_argument("--alias-candidate-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-pairs", type=int, default=-1)
    parser.add_argument("--limit-valid-pairs", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_metrics(total: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def average_metrics(total: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps <= 0:
        return {}
    return {key: value / steps for key, value in sorted(total.items())}


def make_loader(
    sample_file: Optional[Path],
    tokenizer,
    catalog: EntityCatalog,
    args: argparse.Namespace,
    train: bool,
) -> Optional[DataLoader]:
    if sample_file is None or not sample_file.exists():
        return None
    base_dataset = KGQALinkerDataset(sample_file, catalog, {})
    pair_dataset = CrossEncoderPairDataset(
        base_dataset=base_dataset,
        catalog=catalog,
        negatives_per_positive=args.negatives_per_positive,
        alias_candidate_limit=args.alias_candidate_limit,
        seed=args.seed if train else args.seed + 17,
    )
    limit = args.limit_train_pairs if train else args.limit_valid_pairs
    if limit > 0:
        pair_dataset.pairs = pair_dataset.pairs[:limit]
    collator = CrossEncoderCollator(tokenizer, max_length=args.max_length)
    return DataLoader(
        pair_dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        collate_fn=collator,
    )


def optimizer_and_scheduler(model, train_loader: DataLoader, args: argparse.Namespace):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def run_epoch(model, loader, optimizer, scheduler, args, epoch: int) -> Dict[str, float]:
    model.train()
    total: Dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc=f"train reranker epoch {epoch}", dynamic_ncols=True):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        add_metrics(total, {"loss": float(loss.detach().cpu()), "accuracy": float(output["accuracy"].detach().cpu())})
        steps += 1
    return average_metrics(total, steps)


@torch.no_grad()
def evaluate(model, loader, label: str) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    total: Dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc=label, dynamic_ncols=True):
        output = model(batch)
        add_metrics(total, {"loss": float(output["loss"].detach().cpu()), "accuracy": float(output["accuracy"].detach().cpu())})
        steps += 1
    return average_metrics(total, steps)


def save_checkpoint(model, tokenizer, catalog: EntityCatalog, args, output_dir: Path, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    catalog.to_jsonl(output_dir / "entity_catalog.jsonl")
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": args.model_name_or_path,
                "max_length": args.max_length,
                "negatives_per_positive": args.negatives_per_positive,
                "alias_candidate_limit": args.alias_candidate_limit,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = [args.train_file]
    if args.valid_file is not None:
        sample_paths.append(args.valid_file)

    catalog = EntityCatalog.from_files(args.entity_catalog, args.alias_file)
    build_catalog_from_samples(catalog, sample_paths)
    if len(catalog) == 0:
        raise RuntimeError("Entity catalog is empty.")
    print(f"Catalog entities: {len(catalog)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/sep token.")

    train_loader = make_loader(args.train_file, tokenizer, catalog, args, train=True)
    valid_loader = make_loader(args.valid_file, tokenizer, catalog, args, train=False)
    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("No reranker training pairs were built.")
    print(f"Train batches: {len(train_loader)}")
    if valid_loader is not None:
        print(f"Valid batches: {len(valid_loader)}")

    model = CrossEncoderReranker(args.model_name_or_path)
    model.to(torch.device(args.device))
    optimizer, scheduler = optimizer_and_scheduler(model, train_loader, args)

    best_score = float("-inf")
    history = []
    best_metrics: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, args, epoch)
        valid_metrics = evaluate(model, valid_loader, f"valid reranker epoch {epoch}")
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        score = (valid_metrics or train_metrics).get("accuracy", 0.0) - (valid_metrics or train_metrics).get("loss", 0.0)
        if score > best_score:
            best_score = score
            best_metrics = valid_metrics or train_metrics
            save_checkpoint(model, tokenizer, catalog, args, args.output_dir / "best", best_metrics)

    save_checkpoint(model, tokenizer, catalog, args, args.output_dir / "final", best_metrics)
    with (args.output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Saved reranker checkpoints under: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
