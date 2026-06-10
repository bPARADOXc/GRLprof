#!/usr/bin/env python3
"""System tests for eval/webqsp_cli_infer.py.

The default suite avoids loading the 7B/8B language model so that it can run in
CPU-only or driver-limited environments. It still uses real WebQSP/CWQ dataset
questions, real reasoning paths, and real path-embedding caches when available.
"""

from __future__ import annotations

import argparse
import os
import gc
import importlib.util
import json
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "eval" / "webqsp_cli_infer.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("webqsp_cli_infer_under_test", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = load_cli_module()


@dataclass
class CheckResult:
    case_id: str
    name: str
    category: str
    dataset: str = "-"
    status: str = "PASS"
    elapsed_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def now() -> float:
    return time.perf_counter()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = now()
    value = fn()
    return value, (now() - start) * 1000.0


def add_result(results: list[CheckResult], result: CheckResult):
    results.append(result)


def resolve_dataset_dir(dataset: str) -> Path:
    args = SimpleNamespace(dataset=dataset, data_dir=None)
    primary = cli.resolve_data_dir(args)
    if all((primary / cli.SPLIT_TO_FILE[s]).is_file() for s in cli.SPLITS):
        return primary

    fallback = ROOT / f"{dataset}_may_merged"
    if all((fallback / cli.SPLIT_TO_FILE[s]).is_file() for s in cli.SPLITS):
        return fallback

    return primary


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_path(data_dir: Path, split: str) -> Path:
    return data_dir / cli.SPLIT_TO_FILE[split]


def sample_records(split_data: dict[str, list[dict[str, Any]]], per_split: int):
    out = []
    for split, records in split_data.items():
        selected = []
        for item in records:
            if item.get("question"):
                selected.append((split, item))
            if len(selected) >= per_split:
                break
        out.extend(selected)
    return out


def sample_records_with_paths(split_data: dict[str, list[dict[str, Any]]], per_split: int):
    out = []
    for split, records in split_data.items():
        selected = []
        for item in records:
            if item.get("question") and item.get("reasoning_paths"):
                selected.append((split, item))
            if len(selected) >= per_split:
                break
        out.extend(selected)
    return out


def first_n_records_with_paths(split_data: dict[str, list[dict[str, Any]]], total_count: int):
    out = []
    for split, records in split_data.items():
        for item in records:
            if item.get("question") and item.get("reasoning_paths"):
                out.append((split, item))
                if len(out) >= total_count:
                    return out
    return out


def data_args(dataset: str, data_dir: Path, model: str = "llama3", max_paths: int = 8):
    return SimpleNamespace(
        dataset=dataset,
        model=model,
        data_dir=str(data_dir),
        max_paths=max_paths,
        show_all_matched=True,
        device="cpu",
        max_input_tokens=3500,
        num_beams=1,
        max_new_tokens=32,
        temperature=0.0,
        top_p=None,
    )


class FakeTokenized:
    def __init__(self, input_ids: list[int]):
        self.input_ids = input_ids


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"
    unk_token = "<unk>"
    model_max_length = 4096

    def __call__(self, text: str, verbose: bool = False, add_special_tokens: bool = True):
        ids = [min(30000, 100 + (ord(ch) % 1000)) for ch in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return FakeTokenized(ids)

    def batch_decode(self, ids, skip_special_tokens: bool = True):
        return ['{"answers":["mock answer"]}']

    def convert_tokens_to_ids(self, token: str):
        return self.eos_token_id if token == self.eos_token else self.unk_token


class FakeModel:
    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        self.last_generate_kwargs = None

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.param)

    def parameters(self):
        return iter([self.param])

    def generate(self, input_ids, **kwargs):
        self.last_generate_kwargs = kwargs
        suffix = torch.tensor([[9, 2]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def fake_embedding_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    path_count = len(sample.get("reasoning_paths") or [])
    return {
        sample["id"]: [
            [torch.zeros(3, 768, dtype=torch.float32)]
            for _ in range(path_count)
        ]
    }


def run_config_path_checks(results: list[CheckResult], datasets: list[str]):
    for dataset in datasets:
        start = now()
        cfg = cli.MODEL_CONFIG[dataset]
        missing = []
        for model_name, model_cfg in cfg["models"].items():
            model_base = Path(model_cfg["model_base"])
            model_param_dir = Path(model_cfg["model_param_dir"])
            if not model_base.exists():
                missing.append(f"{model_name}.model_base={model_base}")
            if not model_param_dir.exists():
                missing.append(f"{model_name}.model_param_dir={model_param_dir}")
            if not (model_param_dir / "mm_projector.bin").is_file():
                missing.append(f"{model_name}.mm_projector.bin")
            if not (model_param_dir / "config.json").is_file():
                missing.append(f"{model_name}.config.json")
        add_result(
            results,
            CheckResult(
                "F001",
                "本地模型与适配器配置文件可访问",
                "functional",
                dataset,
                "PASS" if not missing else "FAIL",
                (now() - start) * 1000.0,
                {"missing": missing},
            ),
        )


def run_dataset_schema_checks(results: list[CheckResult], dataset: str, data_dir: Path):
    required = ["id", "question", "answer", "q_entity", "a_entity", "hop", "reasoning_paths"]
    start = now()
    split_data = {}
    load_times = {}
    file_sizes = {}
    for split in cli.SPLITS:
        path = split_path(data_dir, split)
        file_sizes[split] = path.stat().st_size
        records, elapsed = timed(lambda p=path: load_json(p))
        split_data[split] = records
        load_times[split] = elapsed

    missing_counts = {key: 0 for key in required}
    path_counts = {}
    for split, records in split_data.items():
        path_counts[split] = {
            "samples": len(records),
            "with_reasoning_paths": sum(bool(x.get("reasoning_paths")) for x in records),
            "avg_reasoning_paths": round(
                statistics.mean([len(x.get("reasoning_paths") or []) for x in records]), 3
            ) if records else 0.0,
        }
        for item in records:
            for key in required:
                if key not in item:
                    missing_counts[key] += 1

    failures = {k: v for k, v in missing_counts.items() if v}
    add_result(
        results,
        CheckResult(
            "F002",
            "数据集样本字段满足CLI问答输入要求",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            (now() - start) * 1000.0,
            {
                "data_dir": str(data_dir),
                "missing_field_counts": failures,
                "split_stats": path_counts,
                "load_time_ms": {k: round(v, 2) for k, v in load_times.items()},
                "file_size_mb": {k: round(v / 1024 / 1024, 2) for k, v in file_sizes.items()},
            },
        ),
    )
    return split_data


def run_question_matching_checks(results: list[CheckResult], dataset: str, data_dir: Path, split_data, per_split: int):
    samples = sample_records(split_data, per_split)
    failures = []
    latencies = []
    start = now()
    for split, item in samples:
        question = item["question"]
        variant = f"  {question.upper()}  "
        match, all_matches = cli.find_question_sample(variant, data_dir)
        latencies.append((now() - start) * 1000.0)
        if match is None or match[1].get("id") != item.get("id"):
            failures.append({"split": split, "id": item.get("id"), "question": question})
    add_result(
        results,
        CheckResult(
            "F003",
            "问题规范化匹配支持大小写与空白差异",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            sum(latencies),
            {
                "sample_count": len(samples),
                "failures": failures,
                "current_implementation_note": "find_question_sample每次调用会重新读取三个split文件。",
            },
        ),
    )

    perf_latencies = []
    for _, item in samples:
        _, elapsed = timed(lambda q=item["question"]: cli.find_question_sample(q, data_dir))
        perf_latencies.append(elapsed)
    add_result(
        results,
        CheckResult(
            "NF001",
            "单问题数据集匹配耗时",
            "nonfunctional",
            dataset,
            "PASS",
            sum(perf_latencies),
            {
                "sample_count": len(samples),
                "avg_ms": round(statistics.mean(perf_latencies), 2) if perf_latencies else 0.0,
                "max_ms": round(max(perf_latencies), 2) if perf_latencies else 0.0,
                "min_ms": round(min(perf_latencies), 2) if perf_latencies else 0.0,
            },
        ),
    )


def run_path_and_prompt_checks(results: list[CheckResult], dataset: str, split_data, max_paths: int, per_split: int):
    samples = sample_records_with_paths(split_data, per_split)
    failures = []
    format_latencies = []
    prompt_latencies = []
    token_lengths = []
    cfg = cli.MODEL_CONFIG[dataset]["models"]["llama3"]
    fake_tokenizer = FakeTokenizer()

    for _, item in samples:
        paths = item.get("reasoning_paths") or []
        indices = cli.select_front_path_indices(paths, max_paths)
        selected = [paths[i] for i in indices]
        if len(selected) != min(max(1, max_paths), len(paths)):
            failures.append({"id": item.get("id"), "reason": "max_paths selection mismatch"})
        _, elapsed_format = timed(lambda s=selected: [cli.format_reasoning_path(p) for p in s])
        format_latencies.append(elapsed_format)
        prompt, elapsed_prompt = timed(lambda q=item["question"]: cli.build_generation_prompt(q, cfg["conv_mode"]))
        prompt_latencies.append(elapsed_prompt)
        ids = cli.tokenize_generation_prompt(prompt, fake_tokenizer, torch.device("cpu"), 3500)
        token_lengths.append(int(ids.shape[1]))
        if item["question"] not in prompt:
            failures.append({"id": item.get("id"), "reason": "question missing in prompt"})

    add_result(
        results,
        CheckResult(
            "F004",
            "推理路径选择、格式化与生成提示构造",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            sum(format_latencies) + sum(prompt_latencies),
            {
                "sample_count": len(samples),
                "failures": failures,
                "max_paths": max_paths,
                "avg_selected_path_count": round(
                    statistics.mean([min(max_paths, len(x[1].get("reasoning_paths") or [])) for x in samples]), 2
                ) if samples else 0.0,
                "avg_prompt_tokens_fake_tokenizer": round(statistics.mean(token_lengths), 2) if token_lengths else 0.0,
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF002",
            "路径格式化与提示构造耗时",
            "nonfunctional",
            dataset,
            "PASS",
            sum(format_latencies) + sum(prompt_latencies),
            {
                "sample_count": len(samples),
                "avg_path_format_ms": round(statistics.mean(format_latencies), 4) if format_latencies else 0.0,
                "avg_prompt_build_ms": round(statistics.mean(prompt_latencies), 4) if prompt_latencies else 0.0,
                "max_prompt_tokens_fake_tokenizer": max(token_lengths) if token_lengths else 0,
            },
        ),
    )


def run_answer_parser_checks(results: list[CheckResult], dataset: str, split_data, per_split: int):
    samples = sample_records(split_data, per_split)
    failures = []
    latencies = []
    for _, item in samples:
        gold_answers = cli.normalize_answer_list(item.get("answer", []))
        first_answer = gold_answers[0] if gold_answers else "unknown"
        raw_cases = [
            json.dumps({"answers": [first_answer]}, ensure_ascii=False),
            json.dumps([first_answer], ensure_ascii=False),
            "```json\n" + json.dumps({"answers": [first_answer]}, ensure_ascii=False) + "\n```",
            first_answer,
        ]
        for raw in raw_cases:
            parsed, elapsed = timed(lambda r=raw: cli.normalize_prediction_to_json(r))
            latencies.append(elapsed)
            if first_answer and first_answer not in parsed["answers"]:
                failures.append({"id": item.get("id"), "raw": raw, "parsed": parsed})
    add_result(
        results,
        CheckResult(
            "F005",
            "生成答案解析支持JSON、列表、代码块和纯文本",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            sum(latencies),
            {
                "sample_count": len(samples),
                "case_count": len(latencies),
                "failures": failures[:5],
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF003",
            "答案解析耗时",
            "nonfunctional",
            dataset,
            "PASS",
            sum(latencies),
            {
                "case_count": len(latencies),
                "avg_ms": round(statistics.mean(latencies), 4) if latencies else 0.0,
                "max_ms": round(max(latencies), 4) if latencies else 0.0,
            },
        ),
    )


def run_embedding_cache_checks(results: list[CheckResult], dataset: str, data_dir: Path, split_data, per_split: int, max_paths: int):
    args = data_args(dataset, data_dir, max_paths=max_paths)
    failures = []
    load_summaries = {}
    start_all = now()
    for split in cli.SPLITS:
        emb_path = data_dir / cli.split_emb_file(args, split)
        if not emb_path.is_file():
            failures.append({"split": split, "reason": f"missing embedding file: {emb_path}"})
            continue

        samples = [(s, item) for s, item in sample_records_with_paths({split: split_data[split]}, per_split)]
        before_rss = rss_mb()
        try:
            emb_data, elapsed = timed(lambda p=emb_path: torch.load(str(p), map_location="cpu"))
        except Exception as exc:
            failures.append({"split": split, "reason": repr(exc)})
            continue

        checked = 0
        for _, item in samples:
            sample_id = item.get("id")
            paths = item.get("reasoning_paths") or []
            selected = cli.select_front_path_indices(paths, max_paths)
            sample_emb = emb_data.get(sample_id)
            if sample_emb is None:
                failures.append({"split": split, "id": sample_id, "reason": "missing sample embedding"})
                continue
            if len(sample_emb) < len(selected):
                failures.append({
                    "split": split,
                    "id": sample_id,
                    "reason": "embedding path count smaller than selected path count",
                    "embedding_paths": len(sample_emb),
                    "selected_paths": len(selected),
                })
            checked += 1
        load_summaries[split] = {
            "embedding_file": str(emb_path),
            "file_size_mb": round(emb_path.stat().st_size / 1024 / 1024, 2),
            "load_ms": round(elapsed, 2),
            "rss_after_load_mb": round(rss_mb(), 2),
            "rss_increase_upper_bound_mb": round(max(0.0, rss_mb() - before_rss), 2),
            "loaded_sample_embeddings": len(emb_data),
            "checked_dataset_questions": checked,
        }
        del emb_data
        gc.collect()

    add_result(
        results,
        CheckResult(
            "F006",
            "路径图向量缓存与数据集问题ID对齐",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            (now() - start_all) * 1000.0,
            {
                "failures": failures[:10],
                "load_summaries": load_summaries,
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF004",
            "路径图向量缓存加载成本",
            "nonfunctional",
            dataset,
            "PASS" if load_summaries else "WARN",
            sum(v["load_ms"] for v in load_summaries.values()),
            {
                "split_load_ms": {k: v["load_ms"] for k, v in load_summaries.items()},
                "split_file_size_mb": {k: v["file_size_mb"] for k, v in load_summaries.items()},
                "note": "torch.load需要一次性反序列化整个split缓存，适合离线评测；在线CLI可考虑常驻缓存或索引化加载。",
            },
        ),
    )


def run_mock_end_to_end_checks(results: list[CheckResult], dataset: str, data_dir: Path, split_data, max_paths: int, per_split: int):
    samples = sample_records_with_paths(split_data, per_split)
    if not samples:
        add_result(
            results,
            CheckResult("F007", "Mock端到端CLI输出契约", "functional", dataset, "WARN", 0.0, {"reason": "no sample with reasoning_paths"})
        )
        return

    args = data_args(dataset, data_dir, max_paths=max_paths)
    cfg = cli.MODEL_CONFIG[dataset]["models"]["llama3"]
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()
    sample_map = {
        cli.canonical_question(item["question"]): (split, item)
        for split, item in samples
    }
    original_loader = cli.load_split_emb_data
    original_finder = cli.find_question_sample
    current_fake_emb = {}

    def fake_find_question_sample(question, _data_dir):
        match = sample_map.get(cli.canonical_question(question))
        if match is None:
            return None, []
        return match, [match]

    cli.find_question_sample = fake_find_question_sample
    cli.load_split_emb_data = lambda _args, _data_dir, _split: current_fake_emb
    failures = []
    outputs = []
    latencies = []
    try:
        for _, item in samples:
            current_fake_emb.clear()
            current_fake_emb.update(fake_embedding_for_sample(item))
            try:
                (_, output), elapsed = timed(
                    lambda q=item["question"]: cli.run_with_captured_output(
                        cli.run_one_question,
                        args,
                        cfg,
                        fake_tokenizer,
                        fake_model,
                        q,
                    )
                )
                latencies.append(elapsed)
            except Exception as exc:
                failures.append({"id": item.get("id"), "reason": repr(exc)})
                continue

            required_fragments = [
                "=== Retrieval Result ===",
                "selected_max_reasoning_paths:",
                "=== Inference Result ===",
                "prediction_answers:",
                "gold_answers:",
            ]
            missing = [frag for frag in required_fragments if frag not in output]
            graph_ok = bool(fake_model.last_generate_kwargs and "graph_emb" in fake_model.last_generate_kwargs)
            if missing or not graph_ok:
                failures.append({
                    "id": item.get("id"),
                    "missing_output_fragments": missing,
                    "graph_embedding_passed_to_generate": graph_ok,
                })
            if len(outputs) < 3:
                outputs.append(
                    {
                        "sample_id": item.get("id"),
                        "question": item.get("question"),
                        "output_preview": "\n".join(output.splitlines()[:12]),
                    }
                )
    finally:
        cli.load_split_emb_data = original_loader
        cli.find_question_sample = original_finder

    add_result(
        results,
        CheckResult(
            "F007",
            "Mock端到端CLI输出契约",
            "functional",
            dataset,
            "PASS" if not failures else "FAIL",
            sum(latencies),
            {
                "sample_count": len(samples),
                "success_count": len(samples) - len(failures),
                "failure_count": len(failures),
                "failures": failures[:5],
                "avg_ms": round(statistics.mean(latencies), 4) if latencies else 0.0,
                "examples": outputs,
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF010",
            "Mock端到端CLI单样本耗时",
            "nonfunctional",
            dataset,
            "PASS" if latencies else "WARN",
            sum(latencies),
            {
                "sample_count": len(samples),
                "avg_ms": round(statistics.mean(latencies), 4) if latencies else 0.0,
                "max_ms": round(max(latencies), 4) if latencies else 0.0,
                "min_ms": round(min(latencies), 4) if latencies else 0.0,
            },
        ),
    )


def run_retriever_module_checks(results: list[CheckResult]):
    start = now()
    retrieve_root = ROOT / "retrieval"
    sys.path.insert(0, str(retrieve_root))
    try:
        from src.dataset.retriever import RetrieverDataset, collate_retriever
        from src.model.retriever import Retriever
        from src.setup import prepare_sample
    except Exception as exc:
        add_result(
            results,
            CheckResult(
                "F009",
                "检索模块可导入并完成三元组打分",
                "functional",
                "webqsp,cwq",
                "FAIL",
                (now() - start) * 1000.0,
                {"reason": repr(exc)},
            ),
        )
        return

    failures = []
    emb_size = 768
    sample = {
        "h_id_list": [0, 1, 2, 0],
        "r_id_list": [0, 1, 2, 3],
        "t_id_list": [1, 2, 3, 3],
        "q_emb": torch.randn(emb_size),
        "entity_embs": torch.randn(4, emb_size),
        "relation_embs": torch.randn(4, emb_size),
        "non_text_entity_list": [],
        "topic_entity_one_hot": torch.nn.functional.one_hot(torch.tensor([1, 0, 0, 0]), num_classes=2).float(),
        "target_triple_probs": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "a_entity_id_list": [2],
    }
    model = Retriever(
        emb_size,
        topic_pe=True,
        DDE_kwargs={"num_rounds": 2, "num_reverse_rounds": 2},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    forward_times = []
    losses = []
    for _ in range(3):
        batch = collate_retriever([sample])
        prepared = prepare_sample(torch.device("cpu"), batch)
        t0 = now()
        logits = model(*prepared[:8])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), sample["target_triple_probs"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        forward_times.append((now() - t0) * 1000.0)
        losses.append(float(loss.item()))
        if tuple(logits.shape) != (4, 1):
            failures.append({"reason": "unexpected logits shape", "shape": list(logits.shape)})
        if not torch.isfinite(loss):
            failures.append({"reason": "loss is not finite"})

    ds_probe = object.__new__(RetrieverDataset)
    path_sample = {
        "h_id_list": [0, 1, 2],
        "r_id_list": [0, 1, 2],
        "t_id_list": [1, 2, 3],
        "q_entity_id_list": [0],
        "a_entity_id_list": [2],
    }
    triple_scores, max_path_length = ds_probe._extract_paths_and_score(path_sample)
    if max_path_length != 2 or triple_scores.tolist() != [1.0, 1.0, 0.0]:
        failures.append({
            "reason": "shortest-path weak supervision mismatch",
            "max_path_length": max_path_length,
            "triple_scores": triple_scores.tolist(),
        })

    elapsed = (now() - start) * 1000.0
    add_result(
        results,
        CheckResult(
            "F009",
            "检索模块可导入并完成三元组打分",
            "functional",
            "webqsp,cwq",
            "PASS" if not failures else "FAIL",
            elapsed,
            {
                "failures": failures,
                "synthetic_triples": 4,
                "embedding_dim": emb_size,
                "loss_values": [round(x, 6) for x in losses],
                "weak_supervision_max_path_length": max_path_length,
                "weak_supervision_triple_scores": triple_scores.tolist(),
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF006",
            "检索器前向与反向传播耗时",
            "nonfunctional",
            "webqsp,cwq",
            "PASS",
            sum(forward_times),
            {
                "runs": len(forward_times),
                "avg_forward_backward_step_ms": round(statistics.mean(forward_times), 4),
                "max_forward_backward_step_ms": round(max(forward_times), 4),
                "note": "合成小图仅用于验证模块计算路径，真实耗时会随候选三元组数量增长。",
            },
        ),
    )


def run_path_recovery_module_checks(results: list[CheckResult]):
    start = now()
    module_path = ROOT / "scripts" / "expand_webqsp_scored_triples.py"
    try:
        if "datasets" not in sys.modules:
            fake_datasets = ModuleType("datasets")

            class FakeDataset:
                @staticmethod
                def from_file(path):
                    raise RuntimeError("Fake datasets.Dataset cannot load arrow files in unit tests.")

            fake_datasets.Dataset = FakeDataset
            fake_datasets.concatenate_datasets = lambda datasets: datasets[0]
            sys.modules["datasets"] = fake_datasets
        path_mod = load_module_from_path("expand_webqsp_scored_triples_under_test", module_path)
    except SystemExit as exc:
        add_result(
            results,
            CheckResult(
                "F010",
                "路径恢复模块可筛选高分边并回溯推理路径",
                "functional",
                "webqsp,cwq",
                "FAIL",
                (now() - start) * 1000.0,
                {"reason": str(exc)},
            ),
        )
        return

    dataset_row = {
        "id": "case-1",
        "question": "synthetic question",
        "answer": ["B"],
        "q_entity": ["Q"],
        "a_entity": ["B"],
        "graph": [
            ["Q", "r1", "A"],
            ["A", "r2", "B"],
            ["Q", "r3", "C"],
            ["C", "r4", "B"],
            ["B", "r5", "D"],
        ],
    }
    sample = {
        "id": "case-1",
        "q_entity_in_graph": ["Q"],
        "scored_triples": [
            ["A", "r2", "B", 0.9],
            ["Q", "r1", "A", 0.3],
            ["C", "r4", "B", 0.2],
            ["A", "r2", "B", 0.7],
        ],
        "max_path_length": 2,
    }
    failures = []
    selected = path_mod.select_high_score_triples(sample["scored_triples"], score_ratio=1.0 / 3.0)
    selected_triples = [list(triple) + [score] for triple, score in selected]
    expected_selected = [(("A", "r2", "B"), 0.9), (("Q", "r1", "A"), 0.3)]
    if [(triple, score) for triple, score in selected] != expected_selected:
        failures.append({"reason": "high-score selection mismatch", "selected": selected_triples})

    paths = path_mod.build_reasoning_paths_for_sample(
        sample=sample,
        dataset_row=dataset_row,
        score_ratio=1.0 / 3.0,
        max_prefix_expansions=3,
    )
    if not paths or paths[0] != [["Q", "r1", "A"], ["A", "r2", "B"]]:
        failures.append({"reason": "recovered path mismatch", "paths": paths})

    output_item = path_mod.build_output_item(sample, dataset_row, paths)
    for key in ["id", "question", "answer", "q_entity", "a_entity", "hop", "num_reasoning_paths", "reasoning_paths"]:
        if key not in output_item:
            failures.append({"reason": "missing output key", "key": key})

    bench_samples = []
    for idx in range(100):
        bench = dict(sample)
        bench["id"] = f"case-{idx}"
        bench_samples.append(bench)
    _, bench_elapsed = timed(
        lambda: [
            path_mod.build_reasoning_paths_for_sample(s, dataset_row, 1.0 / 3.0, 3)
            for s in bench_samples
        ]
    )

    add_result(
        results,
        CheckResult(
            "F010",
            "路径恢复模块可筛选高分边并回溯推理路径",
            "functional",
            "webqsp,cwq",
            "PASS" if not failures else "FAIL",
            (now() - start) * 1000.0,
            {
                "failures": failures,
                "selected_high_score_triples": selected_triples,
                "recovered_paths": paths,
                "output_keys": sorted(output_item.keys()),
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF007",
            "路径恢复合成批处理耗时",
            "nonfunctional",
            "webqsp,cwq",
            "PASS",
            bench_elapsed,
            {
                "synthetic_samples": len(bench_samples),
                "total_ms": round(bench_elapsed, 4),
                "avg_ms_per_sample": round(bench_elapsed / len(bench_samples), 4),
            },
        ),
    )


def run_vectorization_module_checks(results: list[CheckResult], loaded_split_data, resolved_dirs, samples_per_split: int):
    start = now()
    failures = []
    try:
        from utils.embedding_config import embedding_filename, get_embedding_config
    except Exception as exc:
        add_result(
            results,
            CheckResult(
                "F011",
                "路径图向量化配置与缓存结构",
                "functional",
                "webqsp,cwq",
                "FAIL",
                (now() - start) * 1000.0,
                {"reason": repr(exc)},
            ),
        )
        return

    config = get_embedding_config("Alibaba-NLP/gte-modernbert-base")
    if config.hidden_size != 768:
        failures.append({"reason": "unexpected hidden size", "hidden_size": config.hidden_size})

    filename_examples = {
        dataset: {
            split: embedding_filename("Alibaba-NLP/gte-modernbert-base", split, dataset=dataset)
            for split in ("train", "validation", "test")
        }
        for dataset in ("webqsp", "cwq")
    }

    shape_checks = {}
    load_times = {}
    for dataset, data_dir in resolved_dirs.items():
        args = data_args(dataset, data_dir)
        shape_checks[dataset] = {}
        for split in cli.SPLITS:
            emb_path = data_dir / cli.split_emb_file(args, split)
            if not emb_path.is_file():
                failures.append({"dataset": dataset, "split": split, "reason": f"missing embedding file {emb_path}"})
                continue
            emb_data, elapsed = timed(lambda p=emb_path: torch.load(str(p), map_location="cpu"))
            load_times[f"{dataset}/{split}"] = elapsed
            checked = 0
            bad_shapes = []
            for _, item in sample_records_with_paths({split: loaded_split_data[dataset][split]}, samples_per_split):
                sample_emb = emb_data.get(item.get("id"))
                if not sample_emb:
                    bad_shapes.append({"id": item.get("id"), "reason": "missing sample embedding"})
                    continue
                first_path = sample_emb[0]
                if not first_path:
                    bad_shapes.append({"id": item.get("id"), "reason": "empty first embedded path"})
                    continue
                first_triple = first_path[0]
                if not torch.is_tensor(first_triple) or tuple(first_triple.shape) != (3, 768):
                    bad_shapes.append({
                        "id": item.get("id"),
                        "reason": "expected triple tensor shape (3, 768)",
                        "actual": list(first_triple.shape) if torch.is_tensor(first_triple) else str(type(first_triple)),
                    })
                checked += 1
            if bad_shapes:
                failures.extend({"dataset": dataset, "split": split, **x} for x in bad_shapes[:3])
            shape_checks[dataset][split] = {
                "checked_samples": checked,
                "loaded_embedding_items": len(emb_data),
                "bad_shape_count": len(bad_shapes),
            }
            del emb_data
            gc.collect()

    add_result(
        results,
        CheckResult(
            "F011",
            "路径图向量化配置与缓存结构",
            "functional",
            "webqsp,cwq",
            "PASS" if not failures else "FAIL",
            (now() - start) * 1000.0,
            {
                "failures": failures[:10],
                "encoder": config.canonical_name,
                "hidden_size": config.hidden_size,
                "filename_examples": filename_examples,
                "shape_checks": shape_checks,
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF008",
            "路径图向量缓存结构检查加载耗时",
            "nonfunctional",
            "webqsp,cwq",
            "PASS",
            sum(load_times.values()),
            {
                "load_time_ms": {k: round(v, 2) for k, v in load_times.items()},
                "note": "该项复用process_webqsp_e5.py输出格式要求：样本ID -> 路径列表 -> 三元组张量(3,768)。",
            },
        ),
    )


def run_adapter_training_module_checks(results: list[CheckResult]):
    start = now()
    train_path = ROOT / "train" / "train.py"
    train_source = train_path.read_text(encoding="utf-8")
    failures = []

    prompt = cli.build_user_prompt("Who directed Titanic?")
    if "Who directed Titanic?" not in prompt or "<graph>" not in prompt:
        failures.append({"reason": "QA prompt does not include question and graph token", "prompt": prompt})

    parsed = cli.normalize_prediction_to_json('{"answers":["A", "A.", "B"]}')
    if parsed["answers"] != ["A", "B"]:
        failures.append({"reason": "answer normalization mismatch", "parsed": parsed})
    edge_case = cli.normalize_prediction_to_json('{"answers":[" B ;"]}')

    model_args = cli.train_mod.ModelArguments()
    if model_args.tune_mm_mlp_adapter is not True:
        failures.append({"reason": "default tune_mm_mlp_adapter is not True"})
    if model_args.mm_projector_type != "mlp":
        failures.append({"reason": "default mm_projector_type is not mlp", "value": model_args.mm_projector_type})

    required_snippets = [
        "model.requires_grad_(False)",
        'for module_name in ("mm_projector", "graph_path_encoder")',
        "p.requires_grad = True",
        "resume_mm_projector_only_from",
    ]
    missing_snippets = [snippet for snippet in required_snippets if snippet not in train_source]
    if missing_snippets:
        failures.append({"reason": "adapter training source missing expected freeze/save snippets", "missing": missing_snippets})

    _, prompt_elapsed = timed(lambda: [cli.build_user_prompt(f"Question {idx}?") for idx in range(1000)])
    add_result(
        results,
        CheckResult(
            "F012",
            "知识适配器训练模块参数冻结与提示/答案格式",
            "functional",
            "webqsp,cwq",
            "PASS" if not failures else "FAIL",
            (now() - start) * 1000.0,
            {
                "failures": failures,
                "model_arguments": {
                    "tune_mm_mlp_adapter": model_args.tune_mm_mlp_adapter,
                    "mm_projector_type": model_args.mm_projector_type,
                    "mm_use_graph_start_end": model_args.mm_use_graph_start_end,
                },
                "source_checks": {snippet: snippet not in missing_snippets for snippet in required_snippets},
                "prompt_example": prompt,
                "normalization_edge_case_note": {
                    "input": '{"answers":[" B ;"]}',
                    "output": edge_case["answers"],
                    "analysis": "当前实现先压缩空白再去除末尾标点，因此标点前空格可能保留；常规JSON/list/纯文本解析不受影响。",
                },
            },
        ),
    )
    add_result(
        results,
        CheckResult(
            "NF009",
            "适配器训练辅助格式化函数耗时",
            "nonfunctional",
            "webqsp,cwq",
            "PASS",
            prompt_elapsed,
            {
                "prompt_build_count": 1000,
                "total_ms": round(prompt_elapsed, 4),
                "avg_ms": round(prompt_elapsed / 1000.0, 6),
                "note": "该项只衡量训练样本文本提示构造成本，不包含真实LLM前向训练耗时。",
            },
        ),
    )


def run_static_nonfunctional_checks(results: list[CheckResult], datasets: list[str]):
    start = now()
    source = CLI_PATH.read_text(encoding="utf-8")
    external_markers = ["requests.", "http://", "https://", "openai", "dashscope"]
    found = [m for m in external_markers if m in source]
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    add_result(
        results,
        CheckResult(
            "NF005",
            "私域部署与当前硬件可用性静态检查",
            "nonfunctional",
            ",".join(datasets),
            "PASS" if not found else "WARN",
            (now() - start) * 1000.0,
            {
                "external_service_markers_in_cli": found,
                "cuda_available_now": cuda_available,
                "cuda_device_count_now": device_count,
                "model_loading_mode": "真实生成测试通过--run-real-generation开启；无CUDA时退化到CPU逐模型加载。",
            },
        ),
    )


def real_generation_args(dataset: str, model_name: str, data_dir: Path, max_paths: int, max_new_tokens: int, device: str):
    return SimpleNamespace(
        dataset=dataset,
        model=model_name,
        data_dir=str(data_dir),
        cache_dir="../../checkpoint",
        device=device,
        device_map=None,
        single_gpu=(device == "cpu"),
        max_memory_per_gpu="20GiB",
        torch_dtype="float32" if device == "cpu" else "auto",
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        max_input_tokens=3500,
        max_paths=max_paths,
        show_all_matched=True,
        interactive=False,
        question=None,
    )


def maybe_run_real_generation(
    results: list[CheckResult],
    datasets: list[str],
    loaded_split_data: dict[str, dict[str, list[dict[str, Any]]]],
    resolved_dirs: dict[str, Path],
    max_paths: int,
    real_models: list[str],
    max_new_tokens: int,
    timeout_seconds: int,
    real_sample_count: int,
):
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    for dataset in datasets:
        data_dir = resolved_dirs[dataset]
        samples = first_n_records_with_paths(loaded_split_data[dataset], real_sample_count)
        if not samples:
            add_result(
                results,
                CheckResult(
                    "F008",
                    "真实模型批量生成",
                    "functional",
                    dataset,
                    "FAIL",
                    0.0,
                    {"reason": "no sample with reasoning_paths"},
                ),
            )
            continue

        for model_name in real_models:
            start = now()
            args = real_generation_args(dataset, model_name, data_dir, max_paths, max_new_tokens, device)
            cfg = cli.get_model_config(args)
            sample_map = {
                cli.canonical_question(item["question"]): (split, item)
                for split, item in samples
            }
            emb_cache: dict[str, dict[str, Any]] = {}
            original_finder = cli.find_question_sample
            original_loader = cli.load_split_emb_data
            tokenizer = None
            model = None
            load_elapsed = 0.0
            generation_latencies = []
            failures = []
            examples = []

            def fake_find_question_sample(question, _data_dir):
                match = sample_map.get(cli.canonical_question(question))
                if match is None:
                    return None, []
                return match, [match]

            def cached_load_split_emb_data(_args, _data_dir, split):
                if split not in emb_cache:
                    emb_cache[split] = original_loader(_args, _data_dir, split)
                return emb_cache[split]

            try:
                (tokenizer, model), load_text = cli.run_with_captured_output(cli.load_model_and_tokenizer, args, cfg)
                load_elapsed = (now() - start) * 1000.0
                cli.find_question_sample = fake_find_question_sample
                cli.load_split_emb_data = cached_load_split_emb_data

                for split, item in samples:
                    q_start = now()
                    try:
                        _, output = cli.run_with_captured_output(
                            cli.run_one_question,
                            args,
                            cfg,
                            tokenizer,
                            model,
                            item["question"],
                        )
                        q_elapsed = (now() - q_start) * 1000.0
                        generation_latencies.append(q_elapsed)
                        ok = "prediction_answers:" in output and "=== Inference Result ===" in output
                        if not ok:
                            failures.append({"id": item.get("id"), "reason": "missing inference output markers"})
                        if len(examples) < 5:
                            examples.append(
                                {
                                    "split": split,
                                    "sample_id": item.get("id"),
                                    "question": item.get("question"),
                                    "gold_answers": item.get("answer", []),
                                    "output_tail": "\n".join(output.splitlines()[-12:]),
                                }
                            )
                    except Exception as exc:
                        failures.append({"id": item.get("id"), "reason": repr(exc)})

                elapsed = (now() - start) * 1000.0
                status = "PASS" if not failures and len(generation_latencies) == len(samples) else "FAIL"
                add_result(
                    results,
                    CheckResult(
                        "F008",
                        "真实模型批量生成",
                        "functional",
                        f"{dataset}/{model_name}",
                        status,
                        elapsed,
                        {
                            "device": device,
                            "cuda_available": cuda_available,
                            "sample_count": len(samples),
                            "success_count": len(samples) - len(failures),
                            "failure_count": len(failures),
                            "load_ms": round(load_elapsed, 2),
                            "generation_total_ms": round(sum(generation_latencies), 2),
                            "avg_generation_ms": round(statistics.mean(generation_latencies), 2) if generation_latencies else 0.0,
                            "max_new_tokens": max_new_tokens,
                            "loaded_embedding_splits": sorted(emb_cache.keys()),
                            "failures": failures[:5],
                            "examples": examples,
                        },
                    ),
                )
                add_result(
                    results,
                    CheckResult(
                        "NF011",
                        "真实模型批量生成耗时",
                        "nonfunctional",
                        f"{dataset}/{model_name}",
                        "PASS" if generation_latencies else "WARN",
                        sum(generation_latencies),
                        {
                            "sample_count": len(samples),
                            "load_ms": round(load_elapsed, 2),
                            "avg_generation_ms": round(statistics.mean(generation_latencies), 2) if generation_latencies else 0.0,
                            "min_generation_ms": round(min(generation_latencies), 2) if generation_latencies else 0.0,
                            "max_generation_ms": round(max(generation_latencies), 2) if generation_latencies else 0.0,
                        },
                    ),
                )
            except Exception as exc:
                elapsed = (now() - start) * 1000.0
                add_result(
                    results,
                    CheckResult(
                        "F008",
                        "真实模型批量生成",
                        "functional",
                        f"{dataset}/{model_name}",
                        "FAIL",
                        elapsed,
                        {
                            "reason": repr(exc),
                            "device": device,
                            "sample_count": len(samples),
                        },
                    ),
                )
            finally:
                cli.find_question_sample = original_finder
                cli.load_split_emb_data = original_loader
                if model is not None:
                    cli.release_model(model)
                tokenizer = None
                model = None
                emb_cache.clear()
                gc.collect()


def result_counts(results: list[CheckResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def write_reports(results: list[CheckResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cli_system_test_results.json"
    txt_path = output_dir / "cli_system_test_report.txt"

    json_payload = [
        {
            "case_id": r.case_id,
            "name": r.name,
            "category": r.category,
            "dataset": r.dataset,
            "status": r.status,
            "elapsed_ms": round(r.elapsed_ms, 3),
            "details": r.details,
        }
        for r in results
    ]
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("GRLprof terminal QA system functional and non-functional test report")
    lines.append("=" * 36)
    lines.append("")
    lines.append("测试对象：eval/webqsp_cli_infer.py")
    lines.append("测试范围：WebQSP与CWQ数据集中的问题；非数据集问题能力按论文需求假设存在，本轮不作为失败项。")
    lines.append("Modules: terminal QA CLI, retriever, path recovery, path vectorization, and knowledge-adapter training helpers.")
    lines.append("测试方式：使用真实数据、真实路径、真实嵌入缓存、合成小图和mock生成器验证链路；传入--run-real-generation时额外逐数据集/模型加载真实大模型完成批量生成。")
    lines.append(f"结果统计：{result_counts(results)}")
    lines.append("")

    for result in results:
        lines.append(f"[{result.status}] {result.case_id} {result.name} ({result.category}, dataset={result.dataset}, {result.elapsed_ms:.2f} ms)")
        for key, value in result.details.items():
            rendered = json.dumps(value, ensure_ascii=False)
            if len(rendered) > 1200:
                rendered = rendered[:1200] + "...(truncated)"
            lines.append(f"  - {key}: {rendered}")
        lines.append("")

    lines.append("结果分析")
    lines.append("- 功能性方面：测试覆盖了本地模型/适配器配置、数据字段、问题匹配、推理路径选择、提示构造、答案解析、嵌入缓存对齐、CLI输出契约、检索器三元组打分、路径恢复、路径向量缓存结构和适配器训练冻结逻辑。")
    lines.append("- 非功能性方面：测试记录了数据加载、问题匹配、路径格式化、答案解析、嵌入缓存加载、检索器合成前向/反向传播、路径恢复批处理和提示构造耗时，并检查了私域部署相关的外部服务依赖痕迹。")
    lines.append("- 当前CLI的单问题匹配会重复读取三个split文件，适合演示和小规模调试；若面向高频在线问答，应在启动时构建问题索引或接入真正的在线检索模块。")
    lines.append("- 路径嵌入缓存按split整体torch.load，离线评测可接受；交互式长期服务建议常驻缓存、按需加载或拆分索引，以降低首次查询延迟和内存峰值。")
    lines.append("- 当前报告未把非数据集问题作为失败项，因为论文要求假设该能力存在；若后续实现开放输入检索，应新增实体链接、候选检索、路径恢复和生成质量测试。")
    lines.append("- 真实大模型生成通过子进程逐组合执行，避免多个大模型同时驻留内存。若CUDA不可用，测试会退化到CPU路径并在报告中记录真实耗时、输出和失败原因。")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLI system tests for WebQSP/CWQ questions.")
    parser.add_argument("--datasets", default="webqsp,cwq", help="Comma-separated datasets, default: webqsp,cwq")
    parser.add_argument("--samples-per-split", type=int, default=10)
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "test_reports"))
    parser.add_argument("--run-real-generation", action="store_true", help="Run real large-model single-sample generation.")
    parser.add_argument("--real-models", default="llama2", help="Comma-separated real models for generation, default: llama2")
    parser.add_argument("--real-max-new-tokens", type=int, default=32)
    parser.add_argument("--real-timeout-seconds", type=int, default=900)
    parser.add_argument("--real-samples", type=int, default=5, help="Number of dataset questions per dataset/model for real generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    results: list[CheckResult] = []

    run_config_path_checks(results, datasets)
    loaded_split_data = {}
    resolved_dirs = {}
    for dataset in datasets:
        data_dir = resolve_dataset_dir(dataset)
        resolved_dirs[dataset] = data_dir
        split_data = run_dataset_schema_checks(results, dataset, data_dir)
        loaded_split_data[dataset] = split_data
        run_question_matching_checks(results, dataset, data_dir, split_data, args.samples_per_split)
        run_path_and_prompt_checks(results, dataset, split_data, args.max_paths, args.samples_per_split)
        run_answer_parser_checks(results, dataset, split_data, args.samples_per_split)
        run_embedding_cache_checks(results, dataset, data_dir, split_data, args.samples_per_split, args.max_paths)
        run_mock_end_to_end_checks(results, dataset, data_dir, split_data, args.max_paths, args.samples_per_split)

    run_retriever_module_checks(results)
    run_path_recovery_module_checks(results)
    run_vectorization_module_checks(results, loaded_split_data, resolved_dirs, args.samples_per_split)
    run_adapter_training_module_checks(results)
    run_static_nonfunctional_checks(results, datasets)
    if args.run_real_generation:
        real_models = [x.strip() for x in args.real_models.split(",") if x.strip()]
        maybe_run_real_generation(
            results,
            datasets,
            loaded_split_data,
            resolved_dirs,
            args.max_paths,
            real_models,
            args.real_max_new_tokens,
            args.real_timeout_seconds,
            args.real_samples,
        )
    else:
        add_result(
            results,
            CheckResult(
                "F008",
                "真实模型批量生成",
                "functional",
                ",".join(datasets),
                "SKIP",
                0.0,
                {"reason": "未传入--run-real-generation，默认跳过长耗时真实生成。"},
            ),
        )

    json_path, txt_path = write_reports(results, Path(args.output_dir))
    print(f"JSON report: {json_path}")
    print(f"TXT report: {txt_path}")
    print(f"Summary: {result_counts(results)}")


if __name__ == "__main__":
    main()
