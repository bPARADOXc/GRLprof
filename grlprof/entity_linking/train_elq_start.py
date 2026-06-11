from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .catalog import EntityCatalog
from .data import (
    KGQALinkerDataset,
    LinkerCollator,
    build_catalog_from_samples,
    build_relation_vocab,
    load_relation_vocab,
    save_relation_vocab,
)
from .model import KGELQStartModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ELQ-style question entity linker with a start-entity selector "
            "for WebQSP/CWQ KGQA."
        )
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name-or-path",
        default="roberta-base",
        help="PLM checkpoint for question/entity encoders. Use a local path when network is unavailable.",
    )
    parser.add_argument(
        "--entity-catalog",
        type=Path,
        nargs="*",
        default=[],
        help="Optional JSONL/JSON entity catalog files with entity_id/name/aliases/types/relations.",
    )
    parser.add_argument(
        "--alias-file",
        type=Path,
        nargs="*",
        default=[],
        help="Optional alias JSON files such as data/aliases/WQSP_aliase_data.json.",
    )
    parser.add_argument("--relation-vocab", type=Path, default=None)
    parser.add_argument("--relation-vocab-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-question-len", type=int, default=96)
    parser.add_argument("--max-entity-len", type=int, default=160)
    parser.add_argument("--max-span-len", type=int, default=10)
    parser.add_argument("--negatives-per-positive", type=int, default=15)
    parser.add_argument("--alias-candidate-limit", type=int, default=50)
    parser.add_argument("--start-negative-count", type=int, default=20)
    parser.add_argument("--mention-loss-weight", type=float, default=1.0)
    parser.add_argument("--link-loss-weight", type=float, default=1.0)
    parser.add_argument("--start-loss-weight", type=float, default=1.0)
    parser.add_argument("--path-loss-weight", type=float, default=0.2)
    parser.add_argument("--disable-path-loss", action="store_true")
    parser.add_argument("--share-encoders", action="store_true")
    parser.add_argument("--freeze-entity-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-samples", type=int, default=-1)
    parser.add_argument("--limit-valid-samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_loss(outputs: dict, args: argparse.Namespace) -> torch.Tensor:
    losses = outputs["losses"]
    if not losses:
        return outputs["loss"]
    total = None
    weights = {
        "mention": args.mention_loss_weight,
        "link": args.link_loss_weight,
        "start": args.start_loss_weight,
        "path": 0.0 if args.disable_path_loss else args.path_loss_weight,
    }
    for name, loss_value in losses.items():
        weight = weights.get(name, 1.0)
        if weight == 0:
            continue
        term = loss_value * weight
        total = term if total is None else total + term
    if total is None:
        return outputs["loss"] * 0.0
    return total


def average_metrics(total: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps <= 0:
        return {}
    return {key: value / steps for key, value in sorted(total.items())}


def add_metrics(total: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def train_one_epoch(
    model: KGELQStartModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    metrics_total: Dict[str, float] = {}
    steps = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = weighted_loss(outputs, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        metrics = {"loss": float(loss.detach().cpu())}
        for name, value in outputs["losses"].items():
            metrics[f"loss_{name}"] = float(value.detach().cpu())
        metrics.update(outputs["metrics"])
        add_metrics(metrics_total, metrics)
        steps += 1
        progress.set_postfix({key: f"{value:.4f}" for key, value in metrics.items() if key in ("loss", "link_top5", "start_top3")})
    return average_metrics(metrics_total, steps)


@torch.no_grad()
def evaluate(
    model: KGELQStartModel,
    loader: Optional[DataLoader],
    args: argparse.Namespace,
    label: str,
) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    metrics_total: Dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc=label, dynamic_ncols=True):
        outputs = model(batch)
        loss = weighted_loss(outputs, args)
        metrics = {"loss": float(loss.detach().cpu())}
        for name, value in outputs["losses"].items():
            metrics[f"loss_{name}"] = float(value.detach().cpu())
        metrics.update(outputs["metrics"])
        add_metrics(metrics_total, metrics)
        steps += 1
    return average_metrics(metrics_total, steps)


def save_checkpoint(
    model: KGELQStartModel,
    tokenizer,
    catalog: EntityCatalog,
    relation_vocab: Dict[str, int],
    args: argparse.Namespace,
    output_dir: Path,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    catalog.to_jsonl(output_dir / "entity_catalog.jsonl")
    save_relation_vocab(relation_vocab, output_dir / "relation_vocab.json")
    config = {
        "model_name_or_path": args.model_name_or_path,
        "share_encoders": args.share_encoders,
        "entity_feature_size": catalog.feature_size,
        "relation_vocab_size": len(relation_vocab),
        "max_question_len": args.max_question_len,
        "max_entity_len": args.max_entity_len,
        "max_span_len": args.max_span_len,
        "negatives_per_positive": args.negatives_per_positive,
        "alias_candidate_limit": args.alias_candidate_limit,
        "start_negative_count": args.start_negative_count,
        "metrics": metrics or {},
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def build_loaders(
    args: argparse.Namespace,
    tokenizer,
    catalog: EntityCatalog,
    relation_vocab: Dict[str, int],
) -> tuple[DataLoader, Optional[DataLoader]]:
    train_dataset = KGQALinkerDataset(args.train_file, catalog, relation_vocab)
    if args.limit_train_samples > 0:
        train_dataset.examples = train_dataset.examples[: args.limit_train_samples]

    valid_dataset = None
    if args.valid_file is not None and args.valid_file.exists():
        valid_dataset = KGQALinkerDataset(args.valid_file, catalog, relation_vocab)
        if args.limit_valid_samples > 0:
            valid_dataset.examples = valid_dataset.examples[: args.limit_valid_samples]

    train_collator = LinkerCollator(
        tokenizer=tokenizer,
        catalog=catalog,
        max_question_len=args.max_question_len,
        max_entity_len=args.max_entity_len,
        max_span_len=args.max_span_len,
        negatives_per_positive=args.negatives_per_positive,
        alias_candidate_limit=args.alias_candidate_limit,
        start_negative_count=args.start_negative_count,
        relation_vocab=relation_vocab,
        seed=args.seed,
    )
    eval_collator = LinkerCollator(
        tokenizer=tokenizer,
        catalog=catalog,
        max_question_len=args.max_question_len,
        max_entity_len=args.max_entity_len,
        max_span_len=args.max_span_len,
        negatives_per_positive=args.negatives_per_positive,
        alias_candidate_limit=args.alias_candidate_limit,
        start_negative_count=args.start_negative_count,
        relation_vocab=relation_vocab,
        seed=args.seed + 13,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collator,
        )
    return train_loader, valid_loader


def optimizer_and_scheduler(
    model: KGELQStartModel,
    train_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, object]:
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
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def select_score(metrics: Dict[str, float]) -> float:
    if not metrics:
        return float("-inf")
    return (
        metrics.get("start_top3", 0.0)
        + metrics.get("link_top5", 0.0)
        + metrics.get("mention_f1", 0.0)
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = [args.train_file]
    if args.valid_file is not None:
        sample_paths.append(args.valid_file)

    print("Loading entity catalog...")
    catalog = EntityCatalog.from_files(args.entity_catalog, args.alias_file)
    build_catalog_from_samples(catalog, sample_paths)
    if len(catalog) == 0:
        raise RuntimeError("Entity catalog is empty. Provide --entity-catalog/--alias-file or samples with q_entity.")
    print(f"Catalog entities: {len(catalog)}")

    if args.disable_path_loss:
        relation_vocab: Dict[str, int] = {}
    elif args.relation_vocab is not None and args.relation_vocab.exists():
        relation_vocab = load_relation_vocab(args.relation_vocab)
    else:
        relation_vocab = build_relation_vocab([args.train_file], max_size=args.relation_vocab_size)
        save_relation_vocab(relation_vocab, args.output_dir / "relation_vocab.json")
    print(f"Relation vocab size: {len(relation_vocab)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    if tokenizer.pad_token is None:
        raise ValueError(
            "Tokenizer has no pad/eos/sep token. Use a BERT/RoBERTa/DeBERTa-style encoder "
            "or add a pad token before training."
        )
    model = KGELQStartModel(
        model_name_or_path=args.model_name_or_path,
        entity_feature_size=catalog.feature_size,
        relation_vocab_size=0 if args.disable_path_loss else len(relation_vocab),
        share_encoders=args.share_encoders,
    )
    if args.freeze_entity_encoder:
        for param in model.entity_encoder.parameters():
            param.requires_grad = False
    model.to(torch.device(args.device))

    train_loader, valid_loader = build_loaders(args, tokenizer, catalog, relation_vocab)
    optimizer, scheduler = optimizer_and_scheduler(model, train_loader, args)

    best_score = float("-inf")
    best_metrics: Dict[str, float] = {}
    metrics_history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, args, epoch)
        valid_metrics = evaluate(model, valid_loader, args, label=f"valid epoch {epoch}")
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        metrics_history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))

        score = select_score(valid_metrics or train_metrics)
        if score > best_score:
            best_score = score
            best_metrics = valid_metrics or train_metrics
            save_checkpoint(model, tokenizer, catalog, relation_vocab, args, args.output_dir / "best", best_metrics)
        if args.save_every_epoch:
            save_checkpoint(
                model,
                tokenizer,
                catalog,
                relation_vocab,
                args,
                args.output_dir / f"epoch-{epoch}",
                valid_metrics or train_metrics,
            )

    save_checkpoint(model, tokenizer, catalog, relation_vocab, args, args.output_dir / "final", best_metrics)
    with (args.output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)
    print(f"Best score: {best_score:.6f}")
    print(f"Saved checkpoints under: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
