#!/usr/bin/env python3
"""Single-question CLI inference for WebQSP with max reasoning path only."""

import argparse
import ast
import gc
import importlib.util
import json
import re
import sys
import traceback
import os
import warnings
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import asyncio

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.key_binding import KeyBindings
import torch
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from model.builder import load_pretrained_model
from utils import conversation as conversation_lib
from utils.constants import GRAPH_TOKEN_INDEX
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token


TRAIN_SPEC = importlib.util.spec_from_file_location("llaga_train_module", ROOT / "train" / "train.py")
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


def default_model_base(local_subdir: str, hf_repo: str) -> str:
    local_path = ROOT / "llama" / local_subdir
    return str(local_path) if local_path.exists() else hf_repo


MODEL_CONFIG = {
    "webqsp": {
        "data_dir": os.environ.get("GRLPROF_WEBQSP_DATA_DIR", str(ROOT / "data" / "webqsp_may_merged")),
        "embedding_prefix": "webqsp_paths_gte_modernbert_base",
        "models": {
            "llama2": {
                "model_base": os.environ.get("GRLPROF_LLAMA2_MODEL_BASE", default_model_base("llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf")),
                "conv_mode": "llaga_llama_2",
                "model_param_dir": os.environ.get("GRLPROF_WEBQSP_LLAMA2_ADAPTER", ""),
            },
            "llama3": {
                "model_base": os.environ.get("GRLPROF_LLAMA3_MODEL_BASE", default_model_base("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")),
                "conv_mode": "llama_3",
                "model_param_dir": os.environ.get("GRLPROF_WEBQSP_LLAMA3_ADAPTER", ""),
            },
        },
    },
    "cwq": {
        "data_dir": os.environ.get("GRLPROF_CWQ_DATA_DIR", str(ROOT / "data" / "cwq_may_merged")),
        "embedding_prefix": "cwq_paths_gte_modernbert_base",
        "models": {
            "llama2": {
                "model_base": os.environ.get("GRLPROF_LLAMA2_MODEL_BASE", default_model_base("llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf")),
                "conv_mode": "llaga_llama_2",
                "model_param_dir": os.environ.get("GRLPROF_CWQ_LLAMA2_ADAPTER", ""),
            },
            "llama3": {
                "model_base": os.environ.get("GRLPROF_LLAMA3_MODEL_BASE", default_model_base("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")),
                "conv_mode": "llama_3",
                "model_param_dir": os.environ.get("GRLPROF_CWQ_LLAMA3_ADAPTER", ""),
            },
        },
    },
}

SPLITS = ("train", "validation", "test")
SPLIT_TO_FILE = {"train": "train.json", "validation": "validation.json", "test": "test.json"}


def parse_args():
    parser = argparse.ArgumentParser(description="Infer one KGQA question with the first retrieved reasoning paths.")
    parser.add_argument("--dataset", choices=sorted(MODEL_CONFIG), default="webqsp", help="Dataset to query.")
    parser.add_argument("--model", choices=["llama2", "llama3"], required=True, help="llama2 or llama3")
    parser.add_argument("--question", default=None, help="Single-run mode question. If omitted, run interactive CLI.")
    parser.add_argument("--data_dir", default=None, help="Override dataset directory.")
    parser.add_argument("--cache_dir", default="../../checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None, help="Default: balanced_low_0 on multi-GPU CUDA.")
    parser.add_argument("--single_gpu", action="store_true", help="Force model to stay on one device and disable sharding.")
    parser.add_argument("--max_memory_per_gpu", default="20GiB")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "fp16", "half", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_tokens", type=int, default=3500)
    parser.add_argument(
        "--max_paths",
        type=int,
        default=8,
        help="Maximum number of front-ranked reasoning paths to pass to the model and print.",
    )
    parser.add_argument("--show_all_matched", action="store_true", help="Print all matched split/id entries.")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode.")
    return parser.parse_args()


def canonical_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


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


def resolve_eval_device_map(args):
    if args.single_gpu:
        return {"": args.device}
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
    if per_gpu_budget is None:
        return None
    return {gpu_idx: per_gpu_budget for gpu_idx in range(torch.cuda.device_count())}


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
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and tokenizer.eos_token is not None:
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    return eos_token_id


def resolve_pad_token_id(model, tokenizer, eos_token_id):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        if isinstance(eos_token_id, int):
            pad_token_id = eos_token_id
        elif isinstance(eos_token_id, (list, tuple)) and eos_token_id:
            pad_token_id = eos_token_id[0]
    return pad_token_id

@contextmanager
def suppress_model_loading_output(enabled: bool = True):
    if not enabled:
        yield
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        logger_names = [
            "transformers",
            "accelerate",
            "huggingface_hub",
        ]

        old_levels = {}
        for name in logger_names:
            logger = logging.getLogger(name)
            old_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                yield
        finally:
            for name, level in old_levels.items():
                logging.getLogger(name).setLevel(level)

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


def load_split_data(data_dir: Path, split: str):
    with open(data_dir / SPLIT_TO_FILE[split], "r") as f:
        return json.load(f)


def get_dataset_config(args):
    return MODEL_CONFIG[args.dataset]


def get_model_config(args):
    return get_dataset_config(args)["models"][args.model]


def resolve_data_dir(args) -> Path:
    return Path(args.data_dir).resolve() if args.data_dir else Path(get_dataset_config(args)["data_dir"]).resolve()


def split_emb_file(args, split: str) -> str:
    prefix = get_dataset_config(args)["embedding_prefix"]
    return f"{prefix}_{split}.pt"


def load_split_emb_data(args, data_dir: Path, split: str):
    emb_path = data_dir / split_emb_file(args, split)
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing path embedding file: {emb_path}")
    return torch.load(str(emb_path), map_location="cpu")


def find_question_sample(question: str, data_dir: Path):
    target = canonical_question(question)
    matches = []
    for split in SPLITS:
        data = load_split_data(data_dir, split)
        for item in data:
            if canonical_question(item.get("question", "")) == target:
                matches.append((split, item))
    if not matches:
        return None, []
    return matches[0], matches


def path_len(path) -> int:
    if isinstance(path, (list, tuple)):
        return len(path)
    return 1 if path else 0


def stringify_path_value(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def format_reasoning_triple(triple) -> str:
    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
        h, r, t = triple[:3]
        return f"{stringify_path_value(h)} --{stringify_path_value(r)}--> {stringify_path_value(t)}"
    return stringify_path_value(triple)


def format_reasoning_path(path) -> str:
    if isinstance(path, (list, tuple)):
        triples = [format_reasoning_triple(t) for t in path]
        triples = [x for x in triples if x]
        return " ; ".join(triples)
    return stringify_path_value(path)


def build_user_prompt(question: str):
    return train_mod._build_qa_user_prompt(question)


def build_generation_prompt(question: str, conv_mode: str):
    conv = conversation_lib.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], build_user_prompt(question))
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def to_device_dtype_nested_path(path_emb, device, dtype):
    out = []
    for triple_tensor in path_emb:
        if not torch.is_tensor(triple_tensor):
            raise TypeError(f"Invalid path embedding element type: {type(triple_tensor)}")
        out.append(triple_tensor.to(device=device, dtype=dtype))
    return out


def select_front_path_indices(paths, max_paths: int):
    if not paths:
        return []
    limit = max(1, int(max_paths))
    return list(range(min(limit, len(paths))))


def print_selected_paths(paths):
    print(f"selected_max_reasoning_paths: {len(paths)}")
    if paths:
        for output_idx, path in enumerate(paths, start=1):
            print(f"{output_idx}. {format_reasoning_path(path)}")
    else:
        print("None")


def print_answers_only(answers):
    if answers:
        print("answers:")
        for answer in answers:
            print(answer)
    else:
        print("answers:")


def tokenize_generation_prompt(prompt: str, tokenizer, runtime_device: torch.device, max_input_tokens: int):
    add_special_tokens = not prompt.startswith("<|begin_of_text|>")
    input_ids = tokenizer_graph_token(
        prompt,
        tokenizer,
        GRAPH_TOKEN_INDEX,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    if max_input_tokens and max_input_tokens > 0 and input_ids.shape[0] > max_input_tokens:
        input_ids = input_ids[:max_input_tokens]
    return input_ids.unsqueeze(0).to(runtime_device)


def strip_code_fences(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def normalize_answer_text(answer: str) -> str:
    return train_mod._normalize_answer_text(answer)


def normalize_answer_list(answers):
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []
    out, seen = [], set()
    for ans in answers:
        v = normalize_answer_text(ans)
        k = v.lower()
        if not v or k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def normalize_prediction_to_json(raw_text: str):
    raw_text = strip_code_fences(raw_text)
    if not raw_text:
        return {"normalized_json": train_mod._serialize_answers_payload([]), "answers": []}

    candidates = [raw_text]
    first_brace, last_brace = raw_text.find("{"), raw_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace:last_brace + 1])

    for candidate in candidates:
        payload = None
        try:
            payload = json.loads(candidate)
        except Exception:
            try:
                payload = ast.literal_eval(candidate)
            except Exception:
                payload = None
        if payload is None:
            continue
        if isinstance(payload, dict):
            answers = payload.get("answers", payload.get("answer", []))
        elif isinstance(payload, list):
            answers = payload
        elif isinstance(payload, str):
            answers = [payload]
        else:
            answers = []
        if not isinstance(answers, list):
            answers = [answers]
        answers = normalize_answer_list([str(a) for a in answers])
        return {"normalized_json": train_mod._serialize_answers_payload(answers), "answers": answers}

    fallback_answers = normalize_answer_list([raw_text])
    return {
        "normalized_json": train_mod._serialize_answers_payload(fallback_answers),
        "answers": fallback_answers,
    }


def load_model_and_tokenizer(args, cfg):
    disable_torch_init()
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg["conv_mode"]]
    torch_dtype = resolve_torch_dtype(args)
    device_map = resolve_eval_device_map(args)
    max_memory = resolve_max_memory(args, device_map)
    if not cfg.get("model_param_dir"):
        env_name = f"GRLPROF_{args.dataset.upper()}_{args.model.upper()}_ADAPTER"
        raise ValueError(f"Adapter checkpoint is not configured. Set {env_name} or pass a model config override.")
    model_param_dir = Path(cfg["model_param_dir"]).resolve()
    model_path = str(model_param_dir)
    model_name = f"grlprof-{get_model_name_from_path(model_path)}"
    print(f"正在加载... 数据集={args.dataset}, 模型={args.model} ...")
    with suppress_model_loading_output(enabled=True):
        tokenizer, model, _ = load_pretrained_model(
            model_path=model_path,
            model_base=cfg["model_base"],
            model_name=model_name,
            cache_dir=args.cache_dir,
            device=args.device,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
        )

    print("模型加载完成")
    ensure_tokenizer_pad_token(tokenizer)
    sync_model_pad_token_id(model, tokenizer)

    # Keep graph encoder and graph start/end embedding on the same device as token embeddings.
    # This avoids cross-GPU tensor cat in graph soft-prompt injection under sharded loading.
    embed_device = model.get_input_embeddings().weight.device
    graph_encoder = getattr(model.get_model(), "graph_path_encoder", None)
    if graph_encoder is not None:
        graph_encoder.to(device=embed_device)
    if hasattr(model.get_model(), "graph_start_end_emb"):
        graph_start_end_emb = getattr(model.get_model(), "graph_start_end_emb")
        if graph_start_end_emb is not None:
            graph_start_end_emb.data = graph_start_end_emb.data.to(device=embed_device)

    if not is_model_sharded(model):
        model = model.to(torch_dtype if args.device.startswith("cuda") else torch.float32)
        model = model.to(args.device)
    model.eval()
    return tokenizer, model


def run_one_question(args, cfg, tokenizer, model, question: str):
    data_dir = resolve_data_dir(args)

    match, all_matches = find_question_sample(question, data_dir)
    if match is None:
        raise ValueError(f"Question not found in {data_dir}: {question}")

    split_name, sample = match
    reasoning_paths = sample.get("reasoning_paths") or []
    chosen_indices = select_front_path_indices(reasoning_paths, args.max_paths)
    chosen_paths = [reasoning_paths[idx] for idx in chosen_indices]

    if args.show_all_matched:
        print("=== Retrieval Result ===")
        print(f"dataset: {args.dataset}")
        print(f"model: {args.model}")
        print(f"question: {sample.get('question', question)}")
        print(f"sample_id: {sample.get('id')}")
        print(f"split: {split_name}")
        print(f"total_reasoning_paths: {len(reasoning_paths)}")
        if len(all_matches) > 1:
            print("matched_entries:")
            for s, it in all_matches:
                print(f"- split={s}, id={it.get('id')}")
        print_selected_paths(chosen_paths)

    runtime_device = get_model_input_device(model, args.device)
    if not is_model_sharded(model):
        runtime_device = next(model.parameters()).device
    else:
        runtime_device = model.get_input_embeddings().weight.device
    model_dtype = next(model.parameters()).dtype

    emb_data = load_split_emb_data(args, data_dir, split_name)
    sample_id = sample.get("id")
    sample_paths_emb = emb_data.get(sample_id)
    if sample_paths_emb is None:
        raise KeyError(f"Missing embedding for sample id={sample_id} in split={split_name}")
    if not isinstance(sample_paths_emb, list):
        raise TypeError(f"Invalid embedding layout for sample id={sample_id}: {type(sample_paths_emb)}")
    invalid_indices = [idx for idx in chosen_indices if idx < 0 or idx >= len(sample_paths_emb)]
    if invalid_indices:
        raise IndexError(
            f"Chosen path indices {invalid_indices} out of range for sample id={sample_id}, "
            f"emb_path_count={len(sample_paths_emb)}"
        )
    selected_paths_emb = [
        to_device_dtype_nested_path(sample_paths_emb[idx], runtime_device, model_dtype)
        for idx in chosen_indices
    ]
    graph_emb = [selected_paths_emb]

    prompt = build_generation_prompt(sample.get("question", question), cfg["conv_mode"])
    input_ids = tokenize_generation_prompt(prompt, tokenizer, runtime_device, args.max_input_tokens)
    input_token_len = input_ids.shape[1]
    eos_token_id = resolve_eos_token_id(model, tokenizer)
    pad_token_id = resolve_pad_token_id(model, tokenizer, eos_token_id)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(pad_token_id) if pad_token_id is not None else torch.ones_like(input_ids),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        graph=[sample_id],
        graph_emb=graph_emb,
    )
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    generated_ids_tensor = output_ids[:, input_token_len:]
    generated_text = tokenizer.batch_decode(generated_ids_tensor, skip_special_tokens=True)[0].strip()
    normalized = normalize_prediction_to_json(generated_text)

    if args.show_all_matched:
        print("\n=== Inference Result ===")
        print(f"prediction_raw: {generated_text}")
        print(f"prediction_json: {normalized['normalized_json']}")
        print(f"prediction_answers: {json.dumps(normalized['answers'], ensure_ascii=False)}")
        print(f"gold_answers: {json.dumps(normalize_answer_list(sample.get('answer', [])), ensure_ascii=False)}")
    else:
        print_selected_paths(chosen_paths)
        print_answers_only(normalized["answers"])
    del input_ids, output_ids


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
# def print_interactive_help():
#     print("命令:")
#     print("  /dataset webqsp|cwq    切换数据集或者重新加载")
#     print("  /model llama2|llama3   切换模型")
#     print("  /max_paths N           设置推理过程中最多使用前多少条推理路径")
#     print("  /showall on|off        是否显示详细信息")
#     print("  /help                  帮助")
#     print("  /exit                  退出CLI")
#     print("Any other input is treated as a WebQSP question.")

def get_interactive_help_text():
    return "\n".join([
        "命令:",
        "  /dataset webqsp|cwq    切换数据集或者重新加载",
        "  /model llama2|llama3   切换模型",
        "  /max_paths N           设置推理过程中最多使用前多少条推理路径",
        "  /showall on|off        是否显示详细信息",
        "  /help                  帮助",
        "  /exit                  退出CLI",
        "其他输入会被当作当前数据集的问题。",
    ])

def print_interactive_help():
    print(get_interactive_help_text())
def make_cli_prompt(args):
    return f"\n{args.dataset}:{args.model}> "
def run_with_captured_output(fn, *args, **kwargs):
    stdout_buf = StringIO()
    stderr_buf = StringIO()

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        result = fn(*args, **kwargs)

    stdout_text = stdout_buf.getvalue()
    stderr_text = stderr_buf.getvalue()

    text = stdout_text
    if stderr_text:
        text += ("\n" if text else "") + stderr_text

    return result, text

def run_interactive_tui(args):
    cfg = get_model_config(args)
    tokenizer = None
    model = None
    busy = False

    output_area = TextArea(
        text="知识图谱问答交互CLI\n",
        read_only=True,
        scrollbar=True,
        wrap_lines=True,
        focusable=False,
    )

    def append_output(text: str):
        if text is None:
            return
        text = str(text).rstrip()
        if not text:
            return

        if output_area.text and not output_area.text.endswith("\n"):
            output_area.text += "\n"

        output_area.text += text + "\n"
        output_area.buffer.cursor_position = len(output_area.text)

        try:
            app.invalidate()
        except Exception:
            pass

    def status_text():
        state = "运行中" if busy else "就绪"
        return (
            f" KGQA CLI | 数据集={args.dataset} | 模型={args.model} "
            f"| max_paths={args.max_paths} | showall={args.show_all_matched} "
            f"| 状态={state} | Ctrl-Q 退出 "
        )

    status_bar = Window(
        content=FormattedTextControl(status_text),
        height=1,
        style="reverse",
    )

    async def load_current_model(message: str = None):
        nonlocal tokenizer, model, cfg, busy

        busy = True
        input_area.read_only = True

        try:
            result, text = await asyncio.to_thread(
                run_with_captured_output,
                load_model_and_tokenizer,
                args,
                cfg,
            )
            tokenizer, model = result
            append_output("模型已就绪")

        except Exception as exc:
            append_output(f"模型加载失败: {exc}")
            append_output(traceback.format_exc())

        finally:
            busy = False
            input_area.read_only = False
            app.layout.focus(input_area)
            app.invalidate()

    async def handle_line(line: str):
        nonlocal tokenizer, model, cfg, busy

        line = line.strip()
        if not line:
            return

        if busy:
            append_output("当前任务还在运行，请等待完成。")
            return

        append_output(f"{args.dataset}:{args.model}> {line}")

        if line in {"/exit", "exit", "quit"}:
            append_output("再见！")
            app.exit()
            return

        if line == "/help":
            append_output(get_interactive_help_text())
            return

        busy = True
        input_area.read_only = True
        app.invalidate()

        try:
            if line.startswith("/showall "):
                val = line.split(None, 1)[1].strip().lower()
                if val in {"on", "true", "1"}:
                    args.show_all_matched = True
                    append_output("细节展示: 开")
                elif val in {"off", "false", "0"}:
                    args.show_all_matched = False
                    append_output("细节展示: 关")
                else:
                    append_output("命令: /showall on|off")
                return

            if line.startswith("/max_paths "):
                val = line.split(None, 1)[1].strip()
                try:
                    max_paths = int(val)
                except ValueError:
                    append_output("命令: /max_paths N")
                    return

                if max_paths < 1:
                    append_output("最大推理路径数量必须大于0")
                    return

                args.max_paths = max_paths
                append_output(f"最大推理路径: {args.max_paths}")
                return

            if line.startswith("/dataset "):
                dataset_key = line.split(None, 1)[1].strip()

                if dataset_key not in MODEL_CONFIG:
                    append_output("当前不支持这个数据集。使用 webqsp 或者 cwq.")
                    return

                if dataset_key == args.dataset:
                    append_output(f"正在使用数据集: {dataset_key}.")
                    return

                model = None
                tokenizer = None
                cleanup_cuda_memory()

                args.dataset = dataset_key
                args.data_dir = None
                cfg = get_model_config(args)

                busy = False
                input_area.read_only = False

                await load_current_model()
                return

            if line.startswith("/model "):
                model_key = line.split(None, 1)[1].strip()

                if model_key not in get_dataset_config(args)["models"]:
                    append_output("不支持当前模型。请使用 llama2 或者 llama3.")
                    return

                if model_key == args.model:
                    append_output(f"正在使用模型 {model_key}.")
                    return

                model = None
                tokenizer = None
                cleanup_cuda_memory()

                args.model = model_key
                cfg = get_model_config(args)

                busy = False
                input_area.read_only = False

                await load_current_model()
                return

            if tokenizer is None or model is None:
                append_output("模型还没有加载完成，请稍等。")
                return

            _, text = await asyncio.to_thread(
                run_with_captured_output,
                run_one_question,
                args,
                cfg,
                tokenizer,
                model,
                line,
            )

            append_output(text)

        except Exception as exc:
            append_output(f"错误: {exc}")
            append_output(traceback.format_exc())

        finally:
            busy = False
            input_area.read_only = False
            app.layout.focus(input_area)
            app.invalidate()

    def accept_input(buffer):
        line = buffer.text
        buffer.text = ""
        app.create_background_task(handle_line(line))
        return True

    input_area = TextArea(
        height=1,
        multiline=False,
        prompt="> ",
        accept_handler=accept_input,
    )

    root_container = HSplit([
        status_bar,
        Frame(output_area, title="输出"),
        Frame(input_area, title="输入"),
    ])

    kb = KeyBindings()

    @kb.add("c-q")
    def _(event):
        event.app.exit()

    @kb.add("c-c")
    def _(event):
        input_area.buffer.text = ""

    app = Application(
        layout=Layout(root_container, focused_element=input_area),
        key_bindings=kb,
        full_screen=True,
    )

    def pre_run():
        app.create_background_task(
            load_current_model()
        )

    try:
        app.run(pre_run=pre_run)
    finally:
        model = None
        tokenizer = None
        cleanup_cuda_memory()
def main():
    args = parse_args()
    interactive = args.interactive or (args.question is None)

    if interactive:
        run_interactive_tui(args)
        return

    tokenizer = None
    model = None
    cfg = get_model_config(args)

    try:
        tokenizer, model = load_model_and_tokenizer(args, cfg)

        question = args.question.strip()
        if not question:
            raise ValueError("Empty question is not allowed.")

        run_one_question(args, cfg, tokenizer, model, question)

    finally:
        model = None
        tokenizer = None
        cleanup_cuda_memory()


if __name__ == "__main__":
    main()
