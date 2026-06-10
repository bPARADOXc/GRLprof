"""Evaluate WebQSP/CWQ with reasoning paths rendered as plain text.

This script follows eval_webqsp_checkpoints.py's generation, parsing, metrics,
and output format, but it never inserts graph tokens and never loads graph
embeddings. Instead, it serializes each sample's reasoning_paths triples into
the user prompt as text.

Example:
CUDA_VISIBLE_DEVICES=0 python eval/eval_text_paths_checkpoints.py \
  --dataset cwq \
  --data_dir data/cwq_may_merged \
  --output_dir checkpoints/cwq/text_paths_llama3 \
  --model_base meta-llama/Meta-Llama-3-8B-Instruct \
  --conv_mode llama_3 \
  --splits test \
  --device cuda \
  --torch_dtype bfloat16 \
  --max_graph_paths 32
"""

#!/usr/bin/env python3

import argparse
import ast
import gc
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from model.builder import load_pretrained_model
from utils import conversation as conversation_lib
from utils.utils import disable_torch_init, get_model_name_from_path


TRAIN_SPEC = importlib.util.spec_from_file_location("llaga_train_module", ROOT / "train" / "train.py")
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


def ensure_tokenizer_pad_token(tokenizer):
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token


def sync_model_pad_token_id(model, tokenizer):
    if tokenizer.pad_token_id is None:
        return
    model.config.pad_token_id = tokenizer.pad_token_id
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id


def resolve_eos_token_id(model, tokenizer):
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    if eos_token_id is not None:
        return eos_token_id
    return tokenizer.eos_token_id


def resolve_torch_dtype(args):
    dtype_name = args.torch_dtype.lower()
    if dtype_name == "auto":
        if args.device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {args.torch_dtype}")


def torch_dtype_label(torch_dtype):
    return str(torch_dtype).replace("torch.", "")


def resolve_eval_device_map(args):
    if args.device_map is not None:
        return args.device_map
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        return {"": args.device}
    if torch.cuda.device_count() <= 1:
        return {"": args.device}
    return "balanced_low_0"


def resolve_max_memory(args, device_map):
    if not isinstance(device_map, str):
        return None
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        return None
    per_gpu_budget = args.max_memory_per_gpu
    if per_gpu_budget is None and torch.cuda.device_count() > 1:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        usable_gib = max(10, int(total_gib) - 6)
        per_gpu_budget = f"{usable_gib}GiB"
    if per_gpu_budget is None:
        return None
    return {gpu_idx: per_gpu_budget for gpu_idx in range(torch.cuda.device_count())}


def is_model_sharded(model) -> bool:
    return hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict)


def get_model_input_device(model, fallback_device: str) -> torch.device:
    try:
        input_device = model.get_input_embeddings().weight.device
        if input_device.type != "meta":
            return input_device
    except Exception:
        pass

    if is_model_sharded(model):
        for _, mapped_device in model.hf_device_map.items():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)
    return torch.device(fallback_device)


def summarize_model_devices(model):
    param_counts = {}
    for param in model.parameters():
        device_name = str(param.device)
        param_counts[device_name] = param_counts.get(device_name, 0) + 1

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        hf_device_map = {str(key): str(value) for key, value in hf_device_map.items()}

    return {
        "model_device": str(getattr(model, "device", "<unknown>")),
        "input_embedding_device": str(model.get_input_embeddings().weight.device),
        "parameter_device_counts": param_counts,
        "hf_device_map": hf_device_map,
    }


def safe_decode_token_ids(tokenizer, token_ids, skip_special_tokens=False):
    safe_ids = []
    for token_id in token_ids:
        try:
            token_id = int(token_id)
        except Exception:
            continue
        if token_id >= 0:
            safe_ids.append(token_id)
    try:
        return tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)
    except Exception as exc:
        return f"<decode_error: {exc}>"


DATASET_ROOTS = {
    "webqsp": ROOT / "data" / "webqsp_may_merged",
    "cwq": ROOT / "data" / "cwq_may_merged",
}

SPLIT_TO_FILE = {
    "train": "train.json",
    "val": "validation.json",
    "validation": "validation.json",
    "test": "test.json",
}


def canonical_split_name(split: str) -> str:
    return "validation" if split == "val" else split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WebQSP/CWQ QA with reasoning path triples serialized as prompt text."
    )
    parser.add_argument("--output_dir", required=True, help="Directory used to store eval_results for this text-path run.")
    parser.add_argument("--model_base", required=True, help="Base LLM path, e.g. local llama-2-7b-chat-hf.")
    parser.add_argument("--conv_mode", required=True, choices=sorted(conversation_lib.conv_templates.keys()))
    parser.add_argument("--cache_dir", default="./checkpoints/hf_cache")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), default="webqsp", help="Merged QA dataset to evaluate.")
    parser.add_argument("--data_dir", default=None, help="Override merged dataset directory. Defaults to webqsp_may_merged or cwq_may_merged.")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=sorted(SPLIT_TO_FILE))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)#模型最多生成多少token
    parser.add_argument("--limit", type=int, default=-1, help="Optional per-split sample cap for quick debugging.")
    parser.add_argument("--max_graph_paths", type=int, default=32, help="Maximum reasoning paths to render as text. Set to 0 to omit paths.")
    parser.add_argument("--max_path_text_chars", type=int, default=12000, help="Character cap for rendered reasoning-path text.")
    parser.add_argument("--max_input_tokens", type=int, default=3500, help="Maximum prompt tokens before generation. Set to 0 to use tokenizer defaults.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None, help="Device map for from_pretrained. Defaults to balanced_low_0 on multi-GPU CUDA.")
    parser.add_argument("--max_memory_per_gpu", default="20GiB", help='Optional per-GPU max memory budget such as "16GiB".')
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "fp16", "half", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype for inference. auto uses bf16 on supported CUDA GPUs.",
    )
    parser.add_argument(
        "--debug_generation",
        action="store_true",
        help="Write token/device diagnostics for empty outputs or generation errors.",
    )
    parser.add_argument(
        "--debug_generation_limit",
        type=int,
        default=20,
        help="Maximum number of generation debug records to write per split.",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Where to write summaries/details. Defaults to <output_dir>/eval_results/<dataset>_text_paths_eval.",
    )
    return parser.parse_args()


def normalize_answer_text(answer: str) -> str:
    return train_mod._normalize_answer_text(answer)


def normalize_answer_list(answers):
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []

    normalized = []
    seen = set()
    for answer in answers:
        value = normalize_answer_text(answer)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def iter_json_like_candidates(text: str):
    candidates = []

    def add(candidate):
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    stripped = strip_code_fences(text)
    add(text)
    add(stripped)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        add(stripped[first_brace:last_brace + 1])

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        add(stripped[first_bracket:last_bracket + 1])

    return candidates


def try_parse_json_like(candidate: str):
    try:
        return json.loads(candidate), "json.loads"
    except Exception:
        pass

    sanitized = re.sub(r",(\s*[}\]])", r"\1", candidate)
    if sanitized != candidate:
        try:
            return json.loads(sanitized), "json.loads_sanitized"
        except Exception:
            pass

    try:
        return ast.literal_eval(candidate), "ast.literal_eval"
    except Exception:
        pass

    if sanitized != candidate:
        try:
            return ast.literal_eval(sanitized), "ast.literal_eval_sanitized"
        except Exception:
            pass

    return None, None


def extract_answers_from_payload(payload):
    if isinstance(payload, dict):
        if "answers" in payload:
            answers = payload["answers"]
        elif "answer" in payload:
            answers = payload["answer"]
        else:
            return None
    elif isinstance(payload, list):
        answers = payload
    elif isinstance(payload, str):
        answers = [payload]
    else:
        return None

    if isinstance(answers, list):
        return [str(item) for item in answers]
    if answers is None:
        return []
    return [str(answers)]


def extract_quoted_strings(text: str):
    values = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', text):
        token = match.group(1)
        try:
            values.append(json.loads(f'"{token}"'))
        except Exception:
            values.append(token)
    return values


def recover_answers_from_truncated_json(raw_text: str):
    stripped = strip_code_fences(raw_text)

    answers_key_match = re.search(r'"answers"\s*:\s*\[', stripped)
    if answers_key_match is not None:
        list_body = stripped[answers_key_match.end():]
        answers = normalize_answer_list(extract_quoted_strings(list_body))
        if answers:
            return answers, "truncated_answers_field"

    first_bracket = stripped.find("[")
    if first_bracket != -1:
        list_body = stripped[first_bracket + 1:]
        answers = normalize_answer_list(extract_quoted_strings(list_body))
        if answers:
            return answers, "truncated_list"

    return None, None


def normalize_prediction_to_json(raw_text: str):
    raw_text = "" if raw_text is None else str(raw_text).strip()
    if not raw_text:
        answers = []
        return {
            "normalized_json": train_mod._serialize_answers_payload(answers),
            "answers": answers,
            "parse_mode": "empty",
        }

    for candidate in iter_json_like_candidates(raw_text):
        payload, parse_mode = try_parse_json_like(candidate)
        if payload is None:
            continue
        answers = extract_answers_from_payload(payload)
        if answers is None:
            continue
        answers = normalize_answer_list(answers)
        return {
            "normalized_json": train_mod._serialize_answers_payload(answers),
            "answers": answers,
            "parse_mode": parse_mode,
        }

    recovered_answers, parse_mode = recover_answers_from_truncated_json(raw_text)
    if recovered_answers is not None:
        return {
            "normalized_json": train_mod._serialize_answers_payload(recovered_answers),
            "answers": recovered_answers,
            "parse_mode": parse_mode,
        }

    # Fall back to a plain-text answer and normalize it through the same training helper.
    answers = normalize_answer_list([strip_code_fences(raw_text)])
    return {
        "normalized_json": train_mod._serialize_answers_payload(answers),
        "answers": answers,
        "parse_mode": "plain_text_fallback",
    }


def compute_metrics(pred_answers, gold_answers):
    pred_set = set(pred_answers)
    gold_set = set(gold_answers)
    true_positive = len(pred_set & gold_set)
    hit_at_1 = 1.0 if true_positive > 0 else 0.0

    if not pred_set and not gold_set:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        hit_at_1 = 1.0
    else:
        precision = true_positive / len(pred_set) if pred_set else 0.0
        recall = true_positive / len(gold_set) if gold_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_at_1": hit_at_1,
        "true_positive": true_positive,
        "pred_count": len(pred_set),
        "gold_count": len(gold_set),
    }


def stringify_path_value(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def format_reasoning_triple(triple) -> str:
    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
        head, relation, tail = triple[:3]
        return (
            f"{stringify_path_value(head)} --{stringify_path_value(relation)}--> "
            f"{stringify_path_value(tail)}"
        )
    return stringify_path_value(triple)


def format_reasoning_path(path) -> str:
    if isinstance(path, (list, tuple)):
        triples = [format_reasoning_triple(triple) for triple in path]
        triples = [triple for triple in triples if triple]
        return " ; ".join(triples)
    return stringify_path_value(path)


def build_reasoning_paths_text(reasoning_paths, max_paths: int, max_chars: int):
    reasoning_paths = reasoning_paths or []
    if max_paths == 0 or not reasoning_paths:
        return "No reasoning paths were retrieved.", 0, len(reasoning_paths), False

    selected_paths = reasoning_paths if max_paths < 0 else reasoning_paths[:max_paths]
    lines = []
    truncated_by_chars = False
    current_chars = 0
    for path_index, path in enumerate(selected_paths, start=1):
        line = f"{path_index}. {format_reasoning_path(path)}"
        if max_chars > 0 and current_chars + len(line) + 1 > max_chars:
            truncated_by_chars = True
            break
        lines.append(line)
        current_chars += len(line) + 1

    if not lines:
        return "Reasoning paths were omitted because the text budget was too small.", 0, len(reasoning_paths), True
    return "\n".join(lines), len(lines), len(reasoning_paths), truncated_by_chars


def build_text_paths_user_prompt(question: str, reasoning_paths, args):
    paths_text, rendered_count, total_count, truncated_by_chars = build_reasoning_paths_text(
        reasoning_paths,
        max_paths=args.max_graph_paths,
        max_chars=args.max_path_text_chars,
    )
    truncation_note = ""
    if total_count > rendered_count or truncated_by_chars:
        truncation_note = (
            f"\nOnly the first {rendered_count} of {total_count} reasoning paths are shown."
        )

    prompt = (
        "Based on the reasoning paths below, answer the question. "
        'Return only a JSON object in this format: {"answers": ["answer1", "answer2"]}.\n\n'
        f"Question: {question}\n\n"
        f"Reasoning paths:{truncation_note}\n{paths_text}"
    )
    path_info = {
        "reasoning_path_count": total_count,
        "rendered_reasoning_path_count": rendered_count,
        "path_text_truncated": bool(total_count > rendered_count or truncated_by_chars),
    }
    return prompt, path_info


def build_generation_prompt(sample, args):
    conv = conversation_lib.conv_templates[args.conv_mode].copy()
    user_prompt, path_info = build_text_paths_user_prompt(
        sample["question"],
        sample.get("reasoning_paths"),
        args,
    )
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv, path_info


def tokenize_generation_prompt(prompt: str, tokenizer, runtime_device: torch.device, max_input_tokens: int = 3500):
    add_special_tokens = not prompt.startswith("<|begin_of_text|>")
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "add_special_tokens": add_special_tokens,
    }
    max_length = max_input_tokens if max_input_tokens and max_input_tokens > 0 else getattr(tokenizer, "model_max_length", None)
    # Some tokenizers use a huge sentinel value for "no explicit limit";
    # fast tokenizer truncation overflows if that sentinel is passed through.
    if isinstance(max_length, int) and 0 < max_length < 1_000_000:
        tokenizer_kwargs["max_length"] = max_length
    input_ids = tokenizer(prompt, **tokenizer_kwargs).input_ids.squeeze(0)
    return input_ids.unsqueeze(0).to(runtime_device)


def resolve_data_dir(args) -> Path:
    return Path(args.data_dir).resolve() if args.data_dir else DATASET_ROOTS[args.dataset]


def load_split_data(split: str, args):
    split = canonical_split_name(split)
    data_dir = resolve_data_dir(args)
    json_path = data_dir / SPLIT_TO_FILE[split]
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing {args.dataset} split file: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def discover_checkpoints(output_dir: Path):
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if (checkpoint_dir / "config.json").exists():
            checkpoints.append((checkpoint_dir.name, checkpoint_dir))

    final_weight_files = ["mm_projector.bin", "pytorch_model.bin", "adapter_model.bin"]
    include_final = True
    if checkpoints:
        trainer_state_path = output_dir / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                trainer_state = json.loads(trainer_state_path.read_text())
            except Exception:
                trainer_state = {}
            global_step = trainer_state.get("global_step")
            latest_checkpoint_step = int(checkpoints[0][0].split("-")[-1])
            if isinstance(global_step, int) and latest_checkpoint_step == global_step:
                include_final = False

    if include_final and (output_dir / "config.json").exists() and any((output_dir / name).exists() for name in final_weight_files):
        checkpoints.insert(0, ("final", output_dir))
    return checkpoints


def normalize_checkpoint_name(checkpoint_name: str) -> str:
    checkpoint_name = str(checkpoint_name).strip()
    if checkpoint_name == "final":
        return checkpoint_name
    if checkpoint_name.isdigit():
        return f"checkpoint-{checkpoint_name}"
    return checkpoint_name


def filter_checkpoints(checkpoints, requested_checkpoints):
    if not requested_checkpoints:
        return checkpoints

    checkpoint_map = {checkpoint_name: checkpoint_path for checkpoint_name, checkpoint_path in checkpoints}
    filtered = []
    missing = []
    for requested_name in requested_checkpoints:
        checkpoint_name = normalize_checkpoint_name(requested_name)
        checkpoint_path = checkpoint_map.get(checkpoint_name)
        if checkpoint_path is None:
            missing.append(checkpoint_name)
            continue
        filtered.append((checkpoint_name, checkpoint_path))

    if missing:
        available = ", ".join(checkpoint_name for checkpoint_name, _ in checkpoints) or "none"
        raise ValueError(
            "Requested checkpoints not found under output_dir: %s. Available: %s"
            % (", ".join(missing), available)
        )

    return filtered


def resolve_eval_targets(output_dir: Path, args):
    return [("llama_text_paths", Path(args.model_base).resolve())]


def evaluate_split(model, tokenizer, split_name: str, split_data, args, details_path: Path, hit_at_1_zero_path: Path = None):
    samples = split_data[: args.limit] if args.limit > 0 else split_data
    stop_conv = conversation_lib.conv_templates[args.conv_mode]
    stop_str = stop_conv.sep if stop_conv.sep_style != conversation_lib.SeparatorStyle.TWO else stop_conv.sep2
    runtime_device = get_model_input_device(model, args.device)
    model_device_debug = summarize_model_devices(model) if args.debug_generation else None
    debug_path = Path(str(details_path) + ".debug.jsonl") if args.debug_generation else None
    debug_records_written = 0

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_hit_at_1 = 0.0
    micro_true_positive = 0
    micro_pred_count = 0
    micro_gold_count = 0
    parse_mode_counts = {}
    generation_error_count = 0
    count = 0
    path_text_truncated_count = 0
    input_truncated_count = 0

    hit_at_1_zero_records = []

    with open(details_path, "w") as detail_file:
        debug_file = open(debug_path, "w") if debug_path is not None else None
        for sample in tqdm(samples, desc=f"{split_name}", leave=False):
            question = sample["question"]
            gold_answers = normalize_answer_list(sample["answer"])

            prompt, _, path_info = build_generation_prompt(sample, args)
            input_ids = tokenize_generation_prompt(prompt, tokenizer, runtime_device, max_input_tokens=args.max_input_tokens)
            input_token_len = input_ids.shape[1]
            input_truncated = bool(args.max_input_tokens and args.max_input_tokens > 0 and input_token_len >= args.max_input_tokens)
            truncation_info = {
                **path_info,
                "input_token_len": input_token_len,
                "max_input_tokens": args.max_input_tokens,
                "input_truncated": input_truncated,
            }
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=resolve_eos_token_id(model, tokenizer),
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            if args.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p
            else:
                gen_kwargs["do_sample"] = False

            output_ids = None
            generated_token_ids = []
            generated_text_no_skip = ""
            full_text_no_skip = ""
            try:
                with torch.inference_mode():
                    output_ids = model.generate(**gen_kwargs)
                generated_ids_tensor = output_ids[:, input_token_len:]
                generated_token_ids = generated_ids_tensor[0].detach().cpu().tolist()
                generated_text_no_skip = safe_decode_token_ids(
                    tokenizer,
                    generated_token_ids,
                    skip_special_tokens=False,
                )
                full_text_no_skip = safe_decode_token_ids(
                    tokenizer,
                    output_ids[0, -min(output_ids.shape[1], 128):].detach().cpu().tolist(),
                    skip_special_tokens=False,
                )
                generated_text = tokenizer.batch_decode(generated_ids_tensor, skip_special_tokens=True)[0].strip()
                if stop_str and generated_text.endswith(stop_str):
                    generated_text = generated_text[:-len(stop_str)].strip()
                generation_error = None
            except Exception as exc:
                generated_text = ""
                generation_error = str(exc)

            pred_info = normalize_prediction_to_json(generated_text)
            metric_info = compute_metrics(pred_info["answers"], gold_answers)
            parse_mode = pred_info["parse_mode"]
            parse_mode_counts[parse_mode] = parse_mode_counts.get(parse_mode, 0) + 1
            if generation_error is not None:
                generation_error_count += 1

            if (
                debug_file is not None
                and debug_records_written < args.debug_generation_limit
                and (generation_error is not None or not generated_text)
            ):
                debug_record = {
                    "id": sample["id"],
                    "question": question,
                    "generation_error": generation_error,
                    "input_token_len": input_token_len,
                    "output_shape": list(output_ids.shape) if output_ids is not None else None,
                    "generated_token_ids": generated_token_ids[:128],
                    "generated_text_no_skip": generated_text_no_skip[:1000],
                    "full_text_no_skip_tail": full_text_no_skip[-1000:],
                    "runtime_device": str(runtime_device),
                    "input_ids_device": str(input_ids.device),
                    "model_devices": model_device_debug,
                    "eos_token_id": resolve_eos_token_id(model, tokenizer),
                    "pad_token_id": tokenizer.pad_token_id,
                    "stop_str": stop_str,
                    **truncation_info,
                }
                debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
                debug_file.flush()
                debug_records_written += 1

            record = {
                "id": sample["id"],
                "question": question,
                "prediction_raw": generated_text,
                "prediction_json": pred_info["normalized_json"],
                "prediction_answers": pred_info["answers"],
                "gold_raw": sample["answer"],
                "gold_answers": gold_answers,
                "parse_mode": pred_info["parse_mode"],
                "precision": metric_info["precision"],
                "recall": metric_info["recall"],
                "f1": metric_info["f1"],
                "hit_at_1": metric_info["hit_at_1"],
                "true_positive": metric_info["true_positive"],
                "prediction_count": metric_info["pred_count"],
                "gold_count": metric_info["gold_count"],
                "generation_error": generation_error,
                **truncation_info,
            }
            detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if hit_at_1_zero_path is not None and metric_info["hit_at_1"] == 0.0:
                hit_at_1_zero_records.append(record)

            total_precision += metric_info["precision"]
            total_recall += metric_info["recall"]
            total_f1 += metric_info["f1"]
            total_hit_at_1 += metric_info["hit_at_1"]
            micro_true_positive += metric_info["true_positive"]
            micro_pred_count += metric_info["pred_count"]
            micro_gold_count += metric_info["gold_count"]
            if truncation_info["path_text_truncated"]:
                path_text_truncated_count += 1
            if truncation_info["input_truncated"]:
                input_truncated_count += 1
            count += 1

            del input_ids
            if output_ids is not None:
                del output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if debug_file is not None:
            debug_file.close()

    if micro_pred_count == 0 and micro_gold_count == 0:
        micro_precision = 1.0 if count > 0 else None
        micro_recall = 1.0 if count > 0 else None
        micro_f1 = 1.0 if count > 0 else None
    else:
        micro_precision = (micro_true_positive / micro_pred_count) if micro_pred_count > 0 else 0.0
        micro_recall = (micro_true_positive / micro_gold_count) if micro_gold_count > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

    normal_parse_modes = {
        "json.loads",
        "json.loads_sanitized",
        "ast.literal_eval",
        "ast.literal_eval_sanitized",
    }
    parse_mode_abnormal_count = sum(
        mode_count for mode, mode_count in parse_mode_counts.items()
        if mode not in normal_parse_modes
    )

    if hit_at_1_zero_path is not None:
        with open(hit_at_1_zero_path, "w") as f:
            for record in hit_at_1_zero_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "num_samples": count,
        "avg_precision": total_precision / count if count else None,
        "avg_recall": total_recall / count if count else None,
        "avg_f1": total_f1 / count if count else None,
        "avg_hit_at_1": total_hit_at_1 / count if count else None,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_hit_at_1": (total_hit_at_1 / count) if count else None,
        "micro_true_positive": micro_true_positive,
        "micro_prediction_count": micro_pred_count,
        "micro_gold_count": micro_gold_count,
        "parse_mode_counts": parse_mode_counts,
        "parse_mode_abnormal_count": parse_mode_abnormal_count,
        "generation_error_count": generation_error_count,
        "path_text_truncated_count": path_text_truncated_count,
        "input_truncated_count": input_truncated_count,
        "skipped_missing_graph_count": 0,
        "skipped_missing_graph_ids": [],
        "hit_at_1_zero_count": len(hit_at_1_zero_records),
        "hit_at_1_zero_file": str(hit_at_1_zero_path) if hit_at_1_zero_path is not None else None,
        "details_file": str(details_path),
        "debug_generation_file": str(debug_path) if debug_path is not None else None,
    }


def main():
    args = parse_args()
    args.splits = list(dict.fromkeys(canonical_split_name(split) for split in args.splits))
    disable_torch_init()

    device_map = resolve_eval_device_map(args)
    max_memory = resolve_max_memory(args, device_map)
    torch_dtype = resolve_torch_dtype(args)

    output_dir = Path(args.output_dir).resolve()
    data_dir = resolve_data_dir(args)
    default_results_name = f"{args.dataset}_text_paths_eval"
    results_dir = Path(args.results_dir).resolve() if args.results_dir else output_dir / "eval_results" / default_results_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.conv_mode not in conversation_lib.conv_templates:
        raise ValueError(f"Unknown conv_mode: {args.conv_mode}")
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]

    split_cache = {
        split: load_split_data(split, args)
        for split in args.splits
    }
    checkpoints = resolve_eval_targets(output_dir, args)

    summary = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_base": args.model_base,
        "conv_mode": args.conv_mode,
        "splits": args.splits,
        "max_graph_paths": args.max_graph_paths,
        "max_path_text_chars": args.max_path_text_chars,
        "max_input_tokens": args.max_input_tokens,
        "requested_checkpoints": None,
        "no_graph_token": True,
        "text_paths": True,
        "torch_dtype": torch_dtype_label(torch_dtype),
        "all_splits_for_all_checkpoints": True,
        "checkpoints": [],
    }

    final_checkpoint_name = checkpoints[0][0]

    for checkpoint_name, checkpoint_path in checkpoints:
        model_path = str(checkpoint_path)
        model_name = get_model_name_from_path(model_path)
        print(f"Evaluating checkpoint: {checkpoint_name} ({model_path})")

        tokenizer, model, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            cache_dir=args.cache_dir,
            device=args.device,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
        )
        ensure_tokenizer_pad_token(tokenizer)
        sync_model_pad_token_id(model, tokenizer)
        if not is_model_sharded(model):
            model = model.to(torch_dtype if args.device.startswith("cuda") else torch.float32)
            model = model.to(args.device)
        model.eval()
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]

        checkpoint_result = {
            "checkpoint": checkpoint_name,
            "path": model_path,
            "splits": {},
        }

        splits_to_run = list(args.splits)

        for split in splits_to_run:
            split_data = split_cache[split]
            details_path = results_dir / f"{checkpoint_name}_{split}_details.jsonl"
            hit_at_1_zero_path = None
            if checkpoint_name == final_checkpoint_name and split == "test":
                hit_at_1_zero_path = results_dir / f"{checkpoint_name}_{split}_hit_at_1_zero.jsonl"
            split_result = evaluate_split(
                model,
                tokenizer,
                split,
                split_data,
                args,
                details_path,
                hit_at_1_zero_path=hit_at_1_zero_path,
            )
            checkpoint_result["splits"][split] = split_result

        summary["checkpoints"].append(checkpoint_result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
