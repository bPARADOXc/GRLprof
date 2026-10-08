# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import sys

# Support both `python train/train.py` and the FlashAttention wrapper entrypoint.
# Python otherwise places only `train/` at the front of sys.path, which makes
# repository packages such as `utils` unavailable in the direct entrypoint.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_REPO_ROOT, _TRAIN_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import copy
from dataclasses import dataclass, field
import json
import logging
import math
import pathlib
import shutil
import time
from typing import Dict, Optional, Sequence
import pandas as pd

import torch

import transformers

from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
from torch.utils.data import Dataset
from llaga_trainer import LLaGATrainer

from model import *

import random
from tqdm import trange
from utils import conversation as conversation_lib
from utils.embedding_config import embedding_hidden_size, embedding_path, format_embedding_choices
from utils.utils import tokenizer_graph_token
import scipy.sparse as sp
import numpy as np


local_rank = None


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.1f}s"


# 递归地把各种训练/评估过程中产生的对象转换成 JSON 可序列化的基础类型。
# 这样后面保存指标到 json 文件时，不会因为 Tensor、numpy 标量等类型而报错。
def _json_safe(value):
    # 如果传入的是字典，就逐个 value 继续递归处理，保留原来的 key。
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    # 如果传入的是 list 或 tuple，就对其中每个元素继续递归处理。
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # 如果是 PyTorch Tensor：
    # 1. 标量张量直接转成 Python 标量；
    # 2. 多维张量转成普通 list，方便写入 json。
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    # 如果是 numpy 的标量类型（如 np.float32），也转成 Python 原生标量。
    if isinstance(value, np.generic):
        return value.item()
    # 其他本来就可序列化的类型（如 int/float/str/bool/None）直接原样返回。
    return value


# 将某一次 split（validation / test）的评估结果写入磁盘。
# 这里既会保存“当前这一次”的单独 json，也会把同类结果追加到 history.jsonl 中，方便后续画曲线或对比。
def save_split_metrics(output_dir: str, split_name: str, metrics: Dict, global_step: int, filename: Optional[str] = None):
    # 所有评估结果统一放在 output_dir/eval_results 目录下。
    results_dir = os.path.join(output_dir, "eval_results")
    # 如果目录不存在就创建；存在则不报错。
    os.makedirs(results_dir, exist_ok=True)

    # 构造本次评估记录的主体内容：
    # split 表示是 validation 还是 test；
    # global_step 表示这次结果对应训练到了第几步；
    # metrics 是真正的评估指标字典。
    record = {
        "split": split_name,
        "global_step": int(global_step),
        "metrics": _json_safe(metrics),
    }
    # 对 validation 额外记录一个 progress 字段，表示训练进度比例，方便之后快速看“这是 25% 还是 50% 时的结果”。
    if split_name == "validation" and metrics.get("validation_runtime") is not None:
        # 从 metrics 里读取总训练步数（这是我们在 callback 里额外塞进去的）。
        max_steps = metrics.get("validation_max_steps")
        # 只有 max_steps 存在且非 0 时才计算比例，避免除零错误。
        if max_steps:
            record["progress"] = float(global_step) / float(max_steps)

    # 如果调用方没有手动指定文件名，就按 split + step 生成一个稳定文件名。
    if filename is None:
        filename = f"{split_name}_step_{int(global_step):08d}.json"

    # 保存“本次评估”的单独结果文件，便于精确回看某一个 step 的结果。
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # 再把同一类 split 的结果追加到 jsonl 历史文件里，便于顺序追踪全部结果。
    history_path = os.path.join(results_dir, f"{split_name}_history.jsonl")
    with open(history_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# 这个 callback 用来控制“每训练 1/4 跑一次 validation”的节奏。
# 它不会改动训练本身，只是在特定 step 把 control.should_evaluate 置为 True。
class QuarterValidationCallback(transformers.TrainerCallback):
    def __init__(self, output_dir: str):
        # 保存输出目录，后面会把调度信息和评估结果写到这里。
        self.output_dir = output_dir
        # target_steps 用来保存应该触发 validation 的 step 列表，例如 [250, 500, 750, 1000]。
        self.target_steps = []
        # triggered_steps 用来记录已经触发过的 step，避免同一步被重复评估。
        self.triggered_steps = set()
        # checkpoint_steps 用来保存应该额外触发保存的 step 列表。
        self.checkpoint_steps = []
        self.triggered_checkpoint_steps = set()
        # 某些 Transformers/Accelerate/DeepSpeed 组合在训练中途调用 trainer.evaluate()
        # 后会把 deepspeed_engine_wrapped 置空，下一次 backward() 报
        # "'NoneType' object has no attribute 'backward'"。
        # 默认在 DeepSpeed 训练时跳过这类中途 validation，必要时可通过环境变量强制打开。
        self.allow_mid_eval_with_deepspeed = os.environ.get("LLAGA_ALLOW_DEEPSPEED_MID_EVAL", "0") == "1"
        # 是否在 1/4、1/2、3/4、1.0 训练进度处额外保存 checkpoint。
        self.save_quarter_checkpoints = os.environ.get("LLAGA_SAVE_QUARTER_CHECKPOINTS", "0") == "1"

    def on_train_begin(self, args, state, control, **kwargs):
        # 读取 Trainer 计算出的总训练步数 max_steps。
        max_steps = int(state.max_steps)
        # 如果总步数小于等于 0，说明当前没有有效训练计划，直接不设置任何验证点。
        if max_steps <= 0:
            self.target_steps = []
            return control

        # 计算 25% / 50% / 75% / 100% 对应的 step。
        # 用 ceil 是为了保证即使总步数不能被 4 整除，也能覆盖到每一个阶段。
        quarter_steps = [
            max(1, math.ceil(max_steps * fraction))
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ]
        # 排序并去重，防止小数据集时多个比例落到同一个 step。
        self.target_steps = sorted(set(quarter_steps))
        self.checkpoint_steps = list(self.target_steps)

        # 只让主进程负责把调度文件写到磁盘，避免多卡训练时重复写文件。
        if args.process_index == 0:
            # 评估结果目录固定在 output_dir/eval_results。
            results_dir = os.path.join(self.output_dir, "eval_results")
            # 若目录不存在则创建。
            os.makedirs(results_dir, exist_ok=True)
            # 把这次训练对应的验证计划单独落盘，方便后续核对到底在哪些 step 做了 validation。
            schedule = {
                "max_steps": max_steps,
                "validation_steps": self.target_steps,
                "checkpoint_steps": self.checkpoint_steps,
                "initial_validation_step": 0,
            }
            # 保存验证计划到 validation_schedule.json。
            with open(os.path.join(results_dir, "validation_schedule.json"), "w") as f:
                json.dump(schedule, f, indent=2, ensure_ascii=False)
        # 返回 control，保持 Trainer callback 接口约定。
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # 如果当前 step 恰好在目标验证点里，并且之前还没触发过，
        # 就告诉 Trainer 在这个 step 结束后执行一次 evaluate()。
        if self.save_quarter_checkpoints:
            if state.global_step in self.checkpoint_steps and state.global_step not in self.triggered_checkpoint_steps:
                control.should_save = True
                self.triggered_checkpoint_steps.add(state.global_step)
        if not (args.deepspeed and not self.allow_mid_eval_with_deepspeed):
            if state.global_step in self.target_steps and state.global_step not in self.triggered_steps:
                control.should_evaluate = True
                # 记录一下这个 step 已经触发，避免重复。
                self.triggered_steps.add(state.global_step)
        # 把 control 返回给 Trainer。
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 非主进程不负责保存结果；
        # 如果 metrics 为空，也没有可保存内容，直接返回。
        if args.process_index != 0 or metrics is None:
            return control

        # 只处理 validation 结果。
        # 这里通过 key 是否以 validation_ 开头来区分，避免把 test 结果误写到 validation 文件里。
        if any(key.startswith("validation_") for key in metrics):
            # 复制一份 metrics，避免直接修改 Trainer 传进来的原对象。
            metrics_to_save = dict(metrics)
            # 把总训练步数也一起塞进去，方便保存 progress 比例。
            metrics_to_save["validation_max_steps"] = int(state.max_steps)
            # 将本次 validation 结果落盘。
            save_split_metrics(
                output_dir=self.output_dir,
                split_name="validation",
                metrics=metrics_to_save,
                global_step=state.global_step,
            )
        # 返回 control，保持 callback 链正常工作。
        return control


def rank0_print(*args):
    if local_rank == 0:
        print(f"[{_timestamp()}]", *args, flush=True)


def describe_embedding(pretrained_emb):
    if isinstance(pretrained_emb, torch.Tensor):
        return f"tensor(shape={tuple(pretrained_emb.shape)}, dtype={pretrained_emb.dtype})"
    if isinstance(pretrained_emb, list):
        desc = []
        for emb in pretrained_emb:
            if isinstance(emb, torch.Tensor):
                desc.append(tuple(emb.shape))
            else:
                desc.append(type(emb).__name__)
        return f"list(len={len(pretrained_emb)}, shapes={desc})"
    return type(pretrained_emb).__name__


def _preprocess_mutates_sources() -> bool:
    conv = conversation_lib.default_conversation
    return conv.sep_style not in (
        conversation_lib.SeparatorStyle.LLAMA_2,
        conversation_lib.SeparatorStyle.LLAMA_3,
        conversation_lib.SeparatorStyle.TWO,
        conversation_lib.SeparatorStyle.MPT,
    )


def _clone_cached_sample(sample: Dict) -> Dict:
    cloned = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = copy.deepcopy(value)
        elif isinstance(value, dict):
            cloned[key] = copy.deepcopy(value)
        else:
            cloned[key] = value
    return cloned


def _normalize_answer_text(answer: str) -> str:
    answer = str(answer).strip()#先把 answer 强制转成字符串，再去掉首尾空白字符（空格、换行、制表符等）。
    answer = " ".join(answer.split())#把中间连续空格、换行、Tab 全部压缩为一个空格。
    return answer.rstrip(".;:,")#返回时删除字符串右侧（末尾）连续出现的 . ; : ,


def _serialize_answers_payload(answers) -> str:
    return json.dumps({"answers": answers}, ensure_ascii=False)


def _truncate_answer_to_fit_json(answer: str, char_budget: int) -> str:
    if char_budget <= 0:
        return ""

    low, high = 0, len(answer)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = answer[:mid].rstrip()
        if len(_serialize_answers_payload([candidate])) <= char_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _format_joint_answers(answers, char_budget: int):
    normalized_answers = []
    seen = set()
    for answer in answers:
        normalized = _normalize_answer_text(answer)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        normalized_answers.append(normalized)

    if not normalized_answers:
        return _serialize_answers_payload([]), 0, 0, False
#避免有些答案过于长，做了截断的分支
    kept_answers = []
    truncated = False
    for answer in normalized_answers:
        candidate_answers = kept_answers + [answer]
        candidate_payload = _serialize_answers_payload(candidate_answers)
        if kept_answers and len(candidate_payload) > char_budget:
            truncated = True
            break
        kept_answers = candidate_answers

    if not kept_answers:
        first_answer = normalized_answers[0]
        kept_answers = [_truncate_answer_to_fit_json(first_answer, char_budget)]
        truncated = len(first_answer) > len(kept_answers[0]) or len(normalized_answers) > 1

    return (
        _serialize_answers_payload(kept_answers),
        len(normalized_answers),
        len(kept_answers),
        truncated,
    )


def _build_qa_user_prompt(question: str, use_graph_start_end: bool = True) -> str:
    graph_token = DEFAULT_GRAPH_TOKEN
    if use_graph_start_end:
        graph_token = f"{DEFAULT_GRAPH_START_TOKEN}{DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_END_TOKEN}"
    return f"Based on the reasoning graph {graph_token}, answer the question: {question}"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat")#会先检查本地的吗
    version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp')
    mm_use_graph_start_end: bool = field(default=True)
    mm_use_graph_patch_token: bool = field(default=True)#?
    word_embed_proj_dim: Optional[int] = field(default=4096)


@dataclass
class DataArguments:
    max_samples_per_split: int = field(
        default=-1,
        metadata={"help": "Optional sample cap per split for a short smoke run; -1 keeps all samples."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    pretrained_embedding_type: Optional[str] = field(default='E5-base')
    data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing train.json, validation.json, test.json and graph embedding .pt files."
        },
    )
    use_hop: Optional[int] = field(default=0)
    sample_neighbor_size: Optional[int] = field(default=-1)#这个逻辑后面哪里处理的
    use_task:Optional[str] = field(default="QA")
    use_dataset:Optional[str] = field(default="webqsp")
    template: Optional[str] = field(default="QA")



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)#如果从本地加载的话这个要不要设置
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    resume_mm_projector_only_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "Warm-start from an adapter-only checkpoint that contains mm_projector.bin. "
                    "This loads projector weights but does not restore optimizer/scheduler state."
        },
    )
    save_resume_state: bool = field(
        default=False,
        metadata={
            "help": "Whether checkpoint directories should save optimizer/scheduler/RNG/trainer "
                    "resume state. Disable this to keep checkpoints model-only."
        },
    )
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def normalize_mm_adapter_state_dict(state_dict):
    state_dict = dict(state_dict)
    graph_token_key = "model.graph_start_end_emb"
    if graph_token_key not in state_dict:
        for legacy_key in (
            "model.embed_tokens.weight",
            "model.model.embed_tokens.weight",
            "base_model.model.model.embed_tokens.weight",
        ):
            if legacy_key in state_dict:
                state_dict[graph_token_key] = state_dict[legacy_key][-2:].clone()
                break

    for key in list(state_dict.keys()):
        if "embed_tokens" in key or "embed_in" in key:
            del state_dict[key]
    return state_dict


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'graph_path_encoder']
        if getattr(trainer.args, "mm_use_graph_start_end", False):
            keys_to_match.extend(['graph_start_end_emb'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _checkpoint_step_key(checkpoint_path: pathlib.Path) -> int:
    try:
        return int(checkpoint_path.name.split("-")[-1])
    except (TypeError, ValueError):
        return -1


def _find_latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    checkpoint_dirs = [
        checkpoint_dir
        for checkpoint_dir in pathlib.Path(output_dir).glob("checkpoint-*")
        if checkpoint_dir.is_dir()
    ]
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=_checkpoint_step_key)
    return str(checkpoint_dirs[-1])


def _has_deepspeed_resume_state(checkpoint_dir: str) -> bool:
    return any(pathlib.Path(checkpoint_dir).glob("global_step*"))


def _has_hf_resume_weights(checkpoint_dir: str) -> bool:
    checkpoint_path = pathlib.Path(checkpoint_dir)
    known_weight_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    return any((checkpoint_path / file_name).is_file() for file_name in known_weight_files)


def _has_mm_projector_checkpoint(checkpoint_dir: str) -> bool:
    return (pathlib.Path(checkpoint_dir) / "mm_projector.bin").is_file()


def _cleanup_checkpoint_resume_state(checkpoint_dir: str):
    checkpoint_path = pathlib.Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        return

    for path in checkpoint_path.glob("global_step*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt"):
        path = checkpoint_path / filename
        if path.is_file():
            path.unlink()

    for path in checkpoint_path.glob("rng_state*.pth"):
        if path.is_file():
            path.unlink()


def _distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _load_mm_projector_checkpoint(model: torch.nn.Module, checkpoint_dir: str) -> int:
    checkpoint_path = pathlib.Path(checkpoint_dir)
    mm_projector_path = checkpoint_path / "mm_projector.bin"
    if not mm_projector_path.is_file():
        raise FileNotFoundError(f"mm_projector checkpoint not found: {mm_projector_path}")

    mm_projector_weights = torch.load(mm_projector_path, map_location="cpu")
    mm_projector_weights = normalize_mm_adapter_state_dict(mm_projector_weights)
    incompatible = model.load_state_dict(mm_projector_weights, strict=False)
    unexpected_keys = getattr(incompatible, "unexpected_keys", [])
    if unexpected_keys:
        rank0_print(
            "Warning: unexpected keys while loading mm_projector checkpoint:",
            unexpected_keys,
        )
    return len(mm_projector_weights)

#新加的 token 的嵌入向量用原有 token 嵌入的均值来初始化，避免随机初始化带来的不稳定
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        _get_non_padded_length(tokenized.input_ids[0], tokenizer.pad_token_id)
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _get_non_padded_length(token_ids: torch.Tensor, pad_token_id: Optional[int]) -> int:
    # Some eval-only tokenizers never set pad_token_id because they do not pad inputs.
    # Fall back to the full sequence length instead of comparing against None.
    if pad_token_id is None:
        return int(token_ids.shape[0])
    return int(token_ids.ne(pad_token_id).sum())


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation





def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    # 1. 复制 Llama-2 的默认对话模板，并建立角色映射 (human -> user, gpt -> assistant)
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    # 2. 遍历每个样本 (source)
    for i, source in enumerate(sources):
        # 如果第一句话不是 human 说的，跳过第一句（通常要求对话由 human 发起）
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        # 将多轮对话按照角色和内容添加到模板中
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}" # 确保 human 和 gpt 交替出现
            conv.append_message(role, sentence["value"])
        # 生成完整的 Prompt 字符串 (包含特殊 token 如 [INST], <s> 等)
        # lama-2 的标准格式通常是：<s>[INST] 用户指令 [/INST] AI回答 </s><s>[INST] 下一轮指令...
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # 3. 将文本转为 Token ID (Input IDs)

    if has_graph:
        # 如果包含图数据，使用特定的 tokenizer_graph_token 处理
        # 这通常是为了处理 <graph> 占位符，使其不被错误切分
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        # 普通文本直接使用 tokenizer批量处理 (padding 到最长)
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # 4. 准备 Labels (训练目标)
    # 初始化 labels 为 input_ids 的副本，后续会将"不计算Loss"的部分 Mask 掉
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    # 5. 对 Human 的提问部分进行 Mask (设置为 IGNORE_INDEX)
# 在进行 Mask 操作时，我们通过切分字符串并分别 tokenize 来计算每一部分的长度。但有时，分段 Tokenize 的长度之和，不一定等于整体 Tokenize 的长度。

# 为什么会不一致？
# Tokenization 通常基于 BPE (Byte-Pair Encoding) 或类似算法。上下文会影响分词结果。


    sep = "[/INST] " # Llama-2 的指令分隔符
    for conversation, target in zip(conversations, targets):
        # 计算真实有效长度 (去除 padding)
        total_len = _get_non_padded_length(target, tokenizer.pad_token_id)

        # 按每轮对话结束符 (</s>) 切分
        rounds = conversation.split(conv.sep2)
        cur_len = 1 # 初始跳过 bos token (通常是一个 token)
        target[:cur_len] = IGNORE_INDEX
        
        # 遍历每一轮对话
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            # 将每一轮切分为 "User指令" 和 "AI回答" 两部分
            # parts[0]: [INST] user instruction 
            # parts[1]: assistant answer
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep # 把分隔符加回 parts[0]

            # 计算长度用于确定 mask 范围

            # -2 是为了保留 Prompt 的最后一部分后缀（即 [/INST] ）不被 Mask，从而让模型在看到这个结束符时，不仅知道 User 说完了，还能产生 Loss 去学习“紧接着该我开口了”。这样有助于模型学会何时接话
            #看到/inst 知道该回答了
            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2 # 减去 offset，一个bos一个/inst(可能因为特殊 token 处理)
            else:
                round_len = len(tokenizer(rou, verbose=False).input_ids)
                instruction_len = len(tokenizer(parts[0], verbose=False).input_ids) - 2

            # Mask 掉 instruction_len 长度的部分 (即用户说的话不产生 loss)
            # 只有 AI 回答的部分保留原来的 token id
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len # 更新指针到下一轮起点
        
        #注意这里没加对，所有的数据都会变为ignore index
        # Mask 掉 padding 部分
        target[cur_len:] = IGNORE_INDEX # 超过实际对话长度的部分 mask 掉

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )


    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou, verbose=False).input_ids)
                instruction_len = len(tokenizer(parts[0], verbose=False).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

#label和inputid长度一致
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    def tokenize_piece(text: str) -> Sequence[int]:
        if has_graph:
            input_ids = []
            chunks = text.split(DEFAULT_GRAPH_TOKEN)
            for chunk_idx, chunk in enumerate(chunks):
                if chunk:
                    input_ids.extend(
                        tokenizer(chunk, add_special_tokens=False, verbose=False).input_ids
                    )
                if chunk_idx < len(chunks) - 1:
                    input_ids.append(GRAPH_TOKEN_INDEX)
            return input_ids
        return tokenizer(text, add_special_tokens=False, verbose=False).input_ids

    input_id_tensors = []
    label_tensors = []
    max_length = getattr(tokenizer, "model_max_length", None)
    should_truncate = isinstance(max_length, int) and max_length > 0

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        input_ids = []
        labels = []

        system_ids = tokenize_piece(conv.system + conv.sep)
        input_ids.extend(system_ids)
        labels.extend([IGNORE_INDEX] * len(system_ids))

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"

            header = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            is_assistant = role == conv.roles[1]
            header_ids = tokenize_piece(header)
            content_ids = tokenize_piece(sentence["value"] + conv.sep)

            input_ids.extend(header_ids)
            labels.extend([IGNORE_INDEX] * len(header_ids))
            input_ids.extend(content_ids)
            labels.extend(content_ids if is_assistant else [IGNORE_INDEX] * len(content_ids))

        if should_truncate and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        if not any(label != IGNORE_INDEX for label in labels):
            print(f"WARNING: llama_3 sample {i} produced no trainable assistant labels.")

        input_id_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_tensors,
        batch_first=True,
        padding_value=pad_token_id,
    )
    targets = torch.nn.utils.rnn.pad_sequence(
        label_tensors,
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = _get_non_padded_length(target, tokenizer.pad_token_id)

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_graph_token(rou, tokenizer)) + len(tokenizer_graph_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_graph_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_graph_token(prompt, tokenizer)) for prompt in prompts]

    if has_graph:
        input_ids = [tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_graph:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# 这个数据集类实现了以下功能：

# 多数据集支持: 可以混合处理 arxiv, products, pubmed, cora 等多个图数据集。
# 多任务支持: 支持节点分类 (nc), 链接预测 (lp), 节点描述 (nd) 等多种类型的任务。
# 预训练 Embeddings 加载: 加载并拼接各种预训练模型的节点 Embedding（如 SBERT, RoBERTa 等）。
# Prompt 构建: 根据特定任务（Task）和数据集（Dataset）构建“Human-GPT”的对话 Prompt。
# “懒加载”机制 (Implicit): 虽然名字叫 Lazy...，但在这个 __init__ 中我们看到它一次性加载了所有数据的元数据（list_data_dict），真正的“懒”体现在 __getitem__ 时才进行 Tokenization 和 Graph Feature 的提取，避免一开始就消耗大量内存存 Token ID。
class LazySupervisedGraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 datasets: str):#train/validation
        super(LazySupervisedGraphDataset, self).__init__()
        self.split_name = datasets
        self.use_dataset = data_args.use_dataset
        self.use_hop = data_args.use_hop
        self.template = data_args.template
        self.progress_log_interval = int(os.environ.get("LLAGA_PROGRESS_LOG_INTERVAL", "200"))
        self._getitem_calls = 0
        self._logged_missing_graph_embeddings = set()
        self.answer_char_budget = int(
            os.environ.get("LLAGA_QA_ANSWER_CHAR_BUDGET", str(max(2048, tokenizer.model_max_length * 8)))
        )
        self.enable_item_cache = os.environ.get(
            "LLAGA_CACHE_PROCESSED_SAMPLES",
            "0" if datasets == "train" else "1",
        ) == "1"
        self.item_cache = {}
        self.datas={}
        list_data_dict = []
        self.pretrained_embs={}
        dataset_init_start = time.perf_counter()
        rank0_print(
            f"[{self.split_name}] Building dataset with task={data_args.use_task}, "
            f"template={data_args.template}, embedding={data_args.pretrained_embedding_type}"
        )
        # self.index={}
        # 1. 循环处理每个要使用的数据集
        for d, dataset in enumerate([datasets]):
            split_load_start = time.perf_counter()
            repeat=1
            # 如果数据集名字带类似 ".3" 后缀，表示这个数据集的数据要重复 3 次 (Oversampling)
            if "." in dataset:
                # cora: use cora dataset
                # cora.3: repeat cora 3 times
                ds=dataset.split('.')
                repeat=int(ds[1])
                dataset=ds[0]

            data_root = pathlib.Path(data_args.data_dir or f"data/{data_args.use_dataset}_may_merged")
            data_path = data_root / f"{dataset}.json"
            if not data_path.exists():
                print(f"{dataset} not exists under {data_root}!!!!")
                raise FileNotFoundError(data_path)
            rank0_print(f"[{self.split_name}] Loading raw data from {data_path}")
            
            if str(data_path).endswith(".json"):
                raw_load_start = time.perf_counter()
                with open(data_path, 'r') as f:
                    data_dict=json.load(f)
                
                class JSONObject:
                    def __init__(self, data):
                        for key, value in data.items():
                            setattr(self, key, value)
                
                if isinstance(data_dict, list):
                    data = data_dict
                else:
                    data = JSONObject(data_dict)
                raw_count = len(data) if isinstance(data, list) else 1
                rank0_print(
                    f"[{self.split_name}] Raw data loaded: {raw_count} records in {_elapsed(raw_load_start)}"
                )

            self.datas[dataset]=data
            data_dir=str(data_path.parent)
            emb_load_start = time.perf_counter()
            if data_args.template == "ND":
                pretrained_emb = self.load_pretrain_embedding_graph(data_dir, data_args.pretrained_embedding_type,dataset)
                self.structure_emb = torch.load(
                    f"dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt")
            elif data_args.template == "HO":
                pretrained_emb = self.load_pretrain_embedding_hop(data_dir, data_args.pretrained_embedding_type, data_args.use_hop)
                # n = data.num_nodes
                # index = torch.full([n],fill_value=n+1, dtype=torch.long)
                # train_index = torch.arange(data.train_mask.sum())
                # index[data.train_mask] = train_index
                # self.index[dataset]=index
                self.structure_emb = None
            elif data_args.template == "QA":
                pretrained_emb = self.load_pretrain_embedding_graph(data_dir, data_args.pretrained_embedding_type,dataset)
                self.structure_emb = None
            else:
                raise ValueError

            self.pretrained_embs[dataset] = pretrained_emb
            rank0_print(
                f"[{self.split_name}] Embeddings ready: {describe_embedding(pretrained_emb)} "
                f"in {_elapsed(emb_load_start)}"
            )

            self.use_task = data_args.use_task.split('-')

            for task in self.use_task:
                task_build_start = time.perf_counter()
                truncated_answer_samples = 0
                task_list_data_dict = []
                if task == "nc":
                    if data_args.template == "HO":
                        data_path = os.path.join(data_dir,
                                                 f"sampled_2_10_train.jsonl")
                    else:
                        data_path = os.path.join(data_dir,
                                                 f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"]=dataset
                                if dataset == "products":
                                    l["conversations"][0][
                                        'value'] = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError
                elif task == "lp":
                    if data_args.template == "HO":
                        data_path = os.path.join(data_dir,
                                                 f"edge_sampled_2_10_only_train.jsonl")
                    else:
                        data_path = os.path.join(data_dir,
                                                 f"edge_sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_only_train.jsonl")
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l["conversations"][0][
                                    'value'] = f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError
                elif task == "nd":
                    if data_args.template == "HO":
                        data_path = os.path.join(data_dir,
                                                 f"sampled_2_10_train.jsonl")
                    else:
                        data_path = os.path.join(data_dir,
                                                 f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    user_prompt = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                id = l['id']
                                label = data.label_texts[data.y[id]]
                                if dataset in ["arxiv", "cora", "pubmed"]:
                                    title = data.title[id]
                                    if title == "":
                                        assistant_prompt = f"This is a paper in {label} domain"
                                    else:
                                        assistant_prompt = f"This is a paper in {label} domain, it's about {title}."
                                elif dataset == "products":
                                    desc = data.raw_texts[id]
                                    if desc == "":
                                        assistant_prompt = f"This is an amazon product which can be categorized as {label}."
                                    else:
                                        assistant_prompt = f"This is an amazon product which can be categorized as {label}. It can be described as {desc}"
                                else:
                                    raise ValueError
                                l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                                      {'from': 'gpt', 'value': assistant_prompt}]
                                task_list_data_dict.append(l)
                elif task == "nda":
                    if data_args.template == "HO":
                        data_path = os.path.join(data_dir,
                                                 f"sampled_2_10_train.jsonl")
                    else:
                        data_path = os.path.join(data_dir,
                                                 f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    user_prompt = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                id = l['id']
                                label = data.label_texts[data.y[id]]
                                if dataset in ["arxiv", "cora", "pubmed"]:
                                    title = data.title[id]
                                    ab = data.abs[id]
                                    if title == "" and ab == "":
                                        assistant_prompt = f"This is a paper in {label} domain"
                                    elif title == "":
                                        assistant_prompt = f"This is a paper in {label} domain, its title is {title}."
                                    elif ab == "":
                                        assistant_prompt = f"This is a paper in {label} domain, its abstract is {ab}."
                                    else:
                                        assistant_prompt = f"This is a paper in {label} domain, its title is {title}, its abstract is {ab}."
                                elif dataset == "products":
                                    desc = data.raw_texts[id]
                                    if desc == "":
                                        assistant_prompt = f"This is an amazon product which can be categorized as {label}."
                                    else:
                                        assistant_prompt = f"This is an amazon product which can be categorized as {label}. It can be described as {desc}"
                                else:
                                    raise ValueError

                                l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                                      {'from': 'gpt', 'value': assistant_prompt}]
                                task_list_data_dict.append(l)
                elif task == "QA":
                    data = self.datas[dataset]
                    if isinstance(data, list):
                        total_examples = len(data)
                        rank0_print(
                            f"[{self.split_name}] Building QA samples: 0/{total_examples}"
                        )
                        for idx, l in enumerate(data, start=1):
                            question = l['question']
                            # Handle different answer formats if necessary
                            if isinstance(l['answer'], list):
                                answers = l['answer']
                            elif isinstance(l['answer'], str):
                                answers = [l['answer']]
                            else:
                                answers = [str(l['answer'])]

                            # 旧逻辑：多答案样本会被拆成多条训练样本。
                            # for ans in answers:
                            #     l_new = copy.deepcopy(l)
                            #     l_new["dataset"] = dataset
                            #
                            #     user_prompt = f"Based on the reasoning graph {DEFAULT_GRAPH_TOKEN}, answer the question: {question}"
                            #     assistant_prompt = ans
                            #
                            #     l_new["conversations"] = [{'from': 'human', 'value': user_prompt},
                            #                           {'from': 'gpt', 'value': assistant_prompt}]
                            #
                            #     # Use ID as graph identifier for embedding lookup
                            #     l_new['graph'] = l_new['id']
                            #     task_list_data_dict.append(l_new)

                            # 新逻辑：多答案保留为同一条样本中的一个联合 answer。
                            assistant_prompt, total_unique_answers, kept_answers, was_truncated = _format_joint_answers(
                                answers,
                                char_budget=self.answer_char_budget,
                            )
                            if was_truncated:
                                truncated_answer_samples += 1

                            l_new = copy.deepcopy(l)
                            l_new["dataset"] = dataset
                            user_prompt = _build_qa_user_prompt(
                                question,
                                use_graph_start_end=getattr(data_args, "mm_use_graph_start_end", True),
                            )

                            l_new["conversations"] = [{'from': 'human', 'value': user_prompt},
                                                  {'from': 'gpt', 'value': assistant_prompt}]
                            l_new["answer_count"] = total_unique_answers
                            l_new["kept_answer_count"] = kept_answers
                            l_new["answer_was_truncated"] = was_truncated

                            # Use ID as graph identifier for embedding lookup
                            l_new['graph'] = l_new['id']
                            task_list_data_dict.append(l_new)
                            if (
                                idx == 1
                                or idx % self.progress_log_interval == 0
                                or idx == total_examples
                            ):
                                rank0_print(
                                    f"[{self.split_name}] Building QA samples: {idx}/{total_examples}, "
                                    f"expanded={len(task_list_data_dict)}, truncated={truncated_answer_samples}"
                                )
                    else:
                        raise ValueError("Data format for QA task should be a list.")
                else:
                    print(f"{task} not exist!!!")
                    raise ValueError

                if repeat > 1:
                    base_task_list_data_dict = copy.copy(task_list_data_dict)
                    for _ in range(repeat-1):
                        task_list_data_dict += base_task_list_data_dict
                rank0_print(
                    f"[{self.split_name}] Dataset {dataset} Task {task}, size {len(task_list_data_dict)}, "
                    f"built in {_elapsed(task_build_start)}"
                )
                if task == "QA":
                    rank0_print(
                        f"[{self.split_name}] QA answer compaction: "
                        f"{truncated_answer_samples}/{len(task_list_data_dict)} samples exceeded "
                        f"{self.answer_char_budget} chars"
                    )
                list_data_dict.extend(task_list_data_dict)


        random.shuffle(list_data_dict)
        if self.template == "QA":
            kept_samples = []
            skipped_sample_ids = []
            for sample in list_data_dict:
                if self._sample_has_graph_embedding(sample):
                    kept_samples.append(sample)
                else:
                    skipped_sample_ids.append(sample.get("id", "<unknown>"))

            if skipped_sample_ids:
                preview_ids = ", ".join(skipped_sample_ids[:10])
                rank0_print(
                    f"[{self.split_name}] Skipped {len(skipped_sample_ids)} QA samples without graph embeddings. "
                    f"Example ids: {preview_ids}"
                )
            list_data_dict = kept_samples
        rank0_print(
            f"[{self.split_name}] Formatting inputs...Skip in lazy mode, size {len(list_data_dict)}, "
            f"done in {_elapsed(dataset_init_start)}"
        )
        self.tokenizer = tokenizer
        if data_args.max_samples_per_split > 0:
            list_data_dict = list_data_dict[:data_args.max_samples_per_split]
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def load_pretrain_embedding_graph(self, data_dir, pretrained_embedding_type,dataset):
        if pretrained_embedding_type == "simteg":
            simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
            simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
            simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
            pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
        else:
            try:
                emb_path = embedding_path(
                    data_dir,
                    pretrained_embedding_type,
                    dataset,
                    dataset=self.use_dataset,
                )
            except ValueError:
                pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
            else:
                if not emb_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {self.use_dataset} graph embedding file: {emb_path}. "
                        f"Supported embedding types: {format_embedding_choices()}. "
                        "Generate it with scripts/process_webqsp_e5.py before training."
                    )
                pretrained_emb = torch.load(emb_path)
        return pretrained_emb

    def load_pretrain_embedding_hop(self, data_dir, pretrained_embedding_type, hop):
        if pretrained_embedding_type == "simteg":
            simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt")) for i in range(1, hop + 1)]
            simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt")) for i in range(1, hop + 1)]
            simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt")) for i in range(1, hop + 1)]
            pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
        else:
            pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt")) for i in range(1, hop+1)]

        return pretrained_embs

    def _get_graph_lookup_key(self, sample):
        graph = sample.get("graph")
        if isinstance(graph, list):
            if not graph:
                return None
            first_item = graph[0]
            if isinstance(first_item, list):
                return first_item[0] if first_item else None
            return first_item
        return graph

    def _sample_has_graph_embedding(self, sample) -> bool:
        if self.template != "QA" or "graph" not in sample:
            return True
        dataset_name = sample.get("dataset")
        dataset_embeddings = self.pretrained_embs.get(dataset_name)
        if not isinstance(dataset_embeddings, dict):
            return True
        graph_lookup_key = self._get_graph_lookup_key(sample)
        return graph_lookup_key in dataset_embeddings

    def _find_next_valid_sample_index(self, start_index: int) -> Optional[int]:
        total_samples = len(self.list_data_dict)
        for offset in range(1, total_samples):
            candidate_index = (start_index + offset) % total_samples
            if self._sample_has_graph_embedding(self.list_data_dict[candidate_index]):
                return candidate_index
        return None


    def __len__(self):
        return len(self.list_data_dict)



    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample['graphs']) if 'graphs' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'graph' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

#分dataset？

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        self._getitem_calls += 1
        if (
            self._getitem_calls == 1
            or self._getitem_calls % self.progress_log_interval == 0
            or self._getitem_calls == len(self.list_data_dict)
        ):
            rank0_print(
                f"[{self.split_name}] Lazy preprocess progress: "
                f"{self._getitem_calls}/{len(self.list_data_dict)} samples accessed"
            )
        if self.enable_item_cache and i in self.item_cache:
            if self._getitem_calls == 1 or self._getitem_calls % self.progress_log_interval == 0:
                rank0_print(f"[{self.split_name}] Cache hit for sample {i}")
            return _clone_cached_sample(self.item_cache[i])
        # 1. 获取原始文本数据
        sources = self.list_data_dict[i]

        # 2. 文本预处理与 Tokenization
    # 调用全局的 preprocess 函数，将对话文本转为 input_ids 和 labels (mask 掉 human 的部分)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if _preprocess_mutates_sources():
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = [e["conversations"] for e in sources]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_graph=('graph' in self.list_data_dict[i]))#判断是否有这个key
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        # 3. 图特征处理 (Graph Feature Processing)
        if 'graph' in self.list_data_dict[i]:
            if not isinstance(self.list_data_dict[i]['graph'][0], list):
                self.list_data_dict[i]['graph'] = [self.list_data_dict[i]['graph']]#包装成list
            if self.template == "ND":
                # 获取采样的子图节点 ID
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                # 从巨大的预训练 embedding 表中查表 (Lookup)，取出对应节点的特征
                masked_graph_emb = self.pretrained_embs[self.list_data_dict[i]["dataset"]][graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                # 拼接结构特征 (Laplacian Position Encoding)
                if self.structure_emb is not None:
                    graph_emb = torch.cat([graph_emb, self.structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)

            elif self.template == "HO":
                 # 处理基于 Hop 的特征提取逻辑
            # ...
                for g in range(len(self.list_data_dict[i]['graph'])):
                    center_id = self.list_data_dict[i]['graph'][g][0]
                    self.list_data_dict[i]['graph'][g] = [center_id]*(self.use_hop+1)
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                # center_id = self.index[self.list_data_dict[i]["dataset"]][graph[:, 0]]
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[center_id] for emb in self.pretrained_embs[self.list_data_dict[i]["dataset"]]], dim=1)
            elif self.template == "QA":
                graph =self.list_data_dict[i]['graph'][0] #字符串
                dataset_name = self.list_data_dict[i]["dataset"]
                dataset_embeddings = self.pretrained_embs[dataset_name]
                if isinstance(dataset_embeddings, dict) and graph not in dataset_embeddings:
                    sample_id = self.list_data_dict[i].get("id", graph)
                    if sample_id not in self._logged_missing_graph_embeddings:
                        rank0_print(
                            f"[{self.split_name}] Missing graph embedding for sample {sample_id} "
                            f"(lookup key: {graph}); skipping it."
                        )
                        self._logged_missing_graph_embeddings.add(sample_id)
                    fallback_index = self._find_next_valid_sample_index(i)
                    if fallback_index is None:
                        raise KeyError(
                            f"No valid QA sample with graph embedding found after missing key {graph} "
                            f"in split {self.split_name}."
                        )
                    return self.__getitem__(fallback_index)
                graph_emb = dataset_embeddings[graph]#list列表

            else:
                raise ValueError
            data_dict['graph'] = graph
            data_dict['graph_emb'] = graph_emb

        if self.enable_item_cache:
            self.item_cache[i] = _clone_cached_sample(data_dict)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 拆包: 把 list of dicts 拆成 list of input_ids 和 list of labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
            # 2. 填充 (Padding):
    # 因为每个样本的句子长度不一样，必须补齐到当前 Batch 中最长的那个长度。
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)# 空白处填 pad_token (通常是 0)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)# 标签空白处填 -100，计算 Loss 时会忽略
        
         # 3. 截断 (Truncation):
    # 防止由于某些异常样本导致长度超过模型的最大上下文窗口（如 4096），强制截断。
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
            
    # 4. 生成 Attention Mask:
    # 告诉模型哪些位置是真 Token（1），哪些是填充的空气（0）。
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),#这里attention_mask的作用
        )
    # 检查样本里是否有图数据
        if 'graph' in instances[0]:
                    # 1. 提取: 把所有样本的 graph 和 graph_emb 拿出来
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
               # 2. 拼接 (Concatenate):
        # 注意：这里用的是 cat 而不是 stack。
        # 假设 batch_size=2，每个样本有 k 个节点。
        # 结果通常是把所有节点拼成一个长条，或者把所有图特征拼在一起。
        # 具体取决于 Dataset 里返回的形状，但目的是将它们合并成一个大 Tensor 传给模型。
            # batch['graph'] = torch.cat(graph, dim=0)
            # batch['graph_emb'] = torch.cat(graph_emb, dim=0)
            batch['graph'] = graph
            batch['graph_emb'] = graph_emb
        return batch

#这个函数的作用是为有监督微调（supervised fine-tuning）准备训练数据集和数据整理器（collator），并以字典形式返回，方便后续 Trainer 使用。
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    build_start = time.perf_counter()
    rank0_print("Building supervised data module...")
    # 构建训练集，读取 data_args.data_dir/train.json，并沿用现有图特征与对话构造逻辑。
    split_start = time.perf_counter()
    train_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args,datasets="train")
    rank0_print(f"[train] Dataset ready with {len(train_dataset)} samples in {_elapsed(split_start)}")
    # 构建验证集，读取 data_args.data_dir/validation.json。
    split_start = time.perf_counter()
    eval_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args,datasets="validation")
    rank0_print(f"[validation] Dataset ready with {len(eval_dataset)} samples in {_elapsed(split_start)}")
    # 构建测试集，读取 data_args.data_dir/test.json。
    # 训练过程中不会把它交给 Trainer 常规评估，而是在训练结束后手动调用 evaluate。
    split_start = time.perf_counter()
    test_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args,datasets="test")
    rank0_print(f"[test] Dataset ready with {len(test_dataset)} samples in {_elapsed(split_start)}")
    # 构建 batch 拼接器，负责 padding / attention_mask / graph graph_emb 的整理。
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    rank0_print(f"Supervised data module built in {_elapsed(build_start)}")
    # 把三个 split 和 collator 统一打包返回，供 Trainer 初始化阶段使用。
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                test_dataset=test_dataset,
                data_collator=data_collator)


def _train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank#继承的父类有这个参数
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    rank0_print(
        f"Parsed arguments: model={model_args.model_name_or_path}, output={training_args.output_dir}, "
        f"max_len={training_args.model_max_length}, batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, deepspeed={training_args.deepspeed}, "
        f"use_dataset={data_args.use_dataset}, data_dir={data_args.data_dir}"
    )

    if "tmp" not in training_args.output_dir and os.path.exists(training_args.output_dir):
        if bool(os.listdir(training_args.output_dir)):
            print(f"{training_args.output_dir} already exists and not empty!!!!")
            return
        print(f"{training_args.output_dir} already exists!!!!")

#编码后的维度
    if data_args.pretrained_embedding_type in ['sbert', 'simteg_sbert']:
        model_args.mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["simteg_e5", "simteg_roberta", "roberta"]:
        model_args.mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["simteg"]:
        model_args.mm_hidden_size = 1024*2+384
    else:
        try:
            model_args.mm_hidden_size = embedding_hidden_size(data_args.pretrained_embedding_type)
        except ValueError as exc:
            raise ValueError(
                f"{exc}. Non-WebQSP embeddings still supported here: "
                "sbert, simteg_sbert, simteg_e5, simteg_roberta, roberta, simteg"
            ) from exc
    #：Python 的 dataclass 其实就是普通类，允许你在实例化后动态添加新属性（不像 C++/Java 那样严格）。
    #最终的图嵌入特征维度 = 文本嵌入维度 + 结构编码维度。
    if data_args.template == "ND":
        data_args.structure_embedding_dim = int((data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1) / (data_args.sample_neighbor_size - 1))
        model_args.mm_hidden_size += data_args.structure_embedding_dim
    rank0_print(f"mm_hidden_size: {model_args.mm_hidden_size}")

#这些参数会在后续 from_pretrained 加载模型时传入，实现低比特量化加载。
# 主要参数说明：
# device_map: 指定模型加载到哪个设备（如 "cuda:0"）。
# load_in_4bit/load_in_8bit: 是否以 4bit/8bit 量化方式加载。
# quantization_config: 具体的量化配置，包括量化类型、计算精度、是否双重量化等。
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))


    model_load_start = time.perf_counter()
    rank0_print("Loading base model...")
    if 'mpt' in model_args.model_name_or_path:
        if LlagaMPTForCausalLM is None:
            raise ImportError(
                "MPT support is unavailable in the current environment. "
                f"Original import error: {LLAGA_MPT_IMPORT_ERROR}"
            )
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = training_args.mpt_attn_impl
        model = LlagaMPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif 'opt' in model_args.model_name_or_path:
        model = LlagaOPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = LlagaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    rank0_print(f"Base model loaded in {_elapsed(model_load_start)}")

#??
    model.config.use_cache = False

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)
# 目的: 量化后的模型权重通常是冻结的，如果在上面训练，需要做一些特殊处理（例如层归一化层保持 fp32）。prepare_model_for_kbit_training 就是做这个预处理的。
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

#启动梯度检查点：目的: 以时间换空间。训练时不保存所有中间激活值，反向传播时重新计算。这能大幅减少显存占用，允许更大的 Batch Size。

#后面冻结不影响，这个hook在前向传播时会自动触发
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


#目的: 使用 LoRA 技术进行参数高效微调 (PEFT)。不是微调整个大模型的所有参数，而是只训练旁边挂载的小低秩矩阵。
#细节: find_all_linear_names(model) 是一个辅助函数，它会遍历模型，找出所有的 Linear 层（如 Q, K, V 投影层）并把它们设为 LoRA 的微调目标。这样效果通常比只微调 Attention 模块好
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer_load_start = time.perf_counter()
    use_fast_tokenizer = os.environ.get("LLAGA_USE_FAST_TOKENIZER", "1") == "1"
    rank0_print(f"Loading tokenizer... use_fast={use_fast_tokenizer}")
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=use_fast_tokenizer,
        )
    elif 'opt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # model_max_length = 4096
            model_max_length=training_args.model_max_length,
            use_fast=use_fast_tokenizer,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=use_fast_tokenizer,
        )
    rank0_print(f"Tokenizer loaded in {_elapsed(tokenizer_load_start)}")

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
    elif model_args.version == "llama_3":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # if model_args.vision_tower is not None:projector
    #配置projector映射，估计在这里面，也就是arch里改

    # FSDP (Fully Sharded Data Parallel) 是 PyTorch 提供的一种先进的分布式并行训练技术
    # 将 TrainingArguments 中配置的 FSDP（Fully Sharded Data Parallel）设置，传递给模型初始化的方法。
    #这里构建projector了
    #emb设置了grad true ，但encoder还没设置
    graph_module_start = time.perf_counter()
    rank0_print("Initializing graph modules...")
    model.get_model().initialize_graph_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    rank0_print(f"Graph modules initialized in {_elapsed(graph_module_start)}")

    data_args.is_multimodal = True
# 决定是否只微调多模态适配器
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False) # 先冻结整个模型的所有参数
        # 再按实际架构解冻图适配模块，兼容旧版 mm_projector 和当前 graph_path_encoder 命名
        for module_name in ("mm_projector", "graph_path_encoder"):
            module = getattr(model.get_model(), module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
# 决定是否冻结多模态适配器 (例如在第二阶段微调时可能需要)
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for module_name in ("mm_projector", "graph_path_encoder"):
            module = getattr(model.get_model(), module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

    if training_args.bits in [4, 8]:
        for module_name in ("mm_projector", "graph_path_encoder"):
            module = getattr(model.get_model(), module_name, None)
            if module is not None:
                module.to(dtype=compute_dtype, device=training_args.device)
# ... (初始化 tokenizer 的代码，添加特殊 token 如 <graph_start>) ...
    model.config.mm_use_graph_start_end = data_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    training_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    #设置图相关的特殊token，并设置为需要更新参数
    graph_tokenizer_start = time.perf_counter()
    rank0_print("Initializing graph tokenizer...")
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)
    rank0_print(f"Graph tokenizer initialized in {_elapsed(graph_tokenizer_start)}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)# LoRA 层必须高精度
            if 'norm' in name:
                module = module.to(torch.float32)  # LayerNorm 必须保持 FP32 以保持数值稳定
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    # 最后的输出头和 Embedding 层如果是浮点型，也要转为 bf16
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
# 混合精度训练的陷阱: 在 4-bit 量化训练时，权重的精度非常低。为了保证训练的稳定性：
# LoRA 层：必须是高精度（BF16/FP16/FP32），不能量化。
# LayerNorm：层归一化对数值非常敏感，通常强制转为 FP32。
# 输出头 (lm_head)：通常保持高精度以计算 Loss。

    #主要修改这里
    # 一次性构建 train / validation / test 三个数据集以及 collator。
    data_module_start = time.perf_counter()
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    rank0_print(f"Data module ready in {_elapsed(data_module_start)}")
    # test_dataset 不作为 Trainer 的常驻 eval_dataset 使用，
    # 所以先从 data_module 里取出来，后面训练结束后再单独评估。
    test_dataset = data_module.pop("test_dataset")
    # 初始化自定义 Trainer。
    # 这里传进去的 eval_dataset 仍然是 validation 集，因此训练中的 evaluate 会默认跑 validation。
    trainer_init_start = time.perf_counter()
    trainer = LLaGATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    rank0_print(f"Trainer initialized in {_elapsed(trainer_init_start)}")
    # 创建“按 1/4 训练进度触发验证”的 callback。
    quarter_validation_callback = QuarterValidationCallback(training_args.output_dir)
    # 把 callback 注册到 Trainer 中，使其能接管 on_train_begin / on_step_end / on_evaluate 这些时机。
    trainer.add_callback(quarter_validation_callback)
    if training_args.deepspeed and not quarter_validation_callback.allow_mid_eval_with_deepspeed:
        rank0_print("Skipping mid-training validation under DeepSpeed. Set LLAGA_ALLOW_DEEPSPEED_MID_EVAL=1 to force-enable it.")
    if quarter_validation_callback.save_quarter_checkpoints:
        rank0_print("Saving quarter checkpoints at 1/4, 1/2, 3/4, and 1.0 progress because LLAGA_SAVE_QUARTER_CHECKPOINTS=1.")

    # 在正式训练开始前先跑一次 validation。
    # 这对应你的需求里的“刚开始训练时也要跑一次验证集”。
    skip_initial_eval = os.environ.get("LLAGA_SKIP_INITIAL_EVAL", "1") == "1"
    if training_args.deepspeed and not quarter_validation_callback.allow_mid_eval_with_deepspeed:
        skip_initial_eval = True
        rank0_print(
            "Skipping initial validation under DeepSpeed because it can invalidate the backward engine. "
            "Set LLAGA_ALLOW_DEEPSPEED_MID_EVAL=1 to force-enable it."
        )
    elif skip_initial_eval:
        rank0_print("Skipping initial validation because LLAGA_SKIP_INITIAL_EVAL=1")
    else:
        initial_eval_start = time.perf_counter()
        rank0_print("Starting initial validation before training...")
        trainer.evaluate(metric_key_prefix="validation")
        rank0_print(f"Initial validation finished in {_elapsed(initial_eval_start)}")

    latest_checkpoint = _find_latest_checkpoint_dir(training_args.output_dir)
    adapter_only_resume_checkpoint = training_args.resume_mm_projector_only_from
    if adapter_only_resume_checkpoint is not None:
        adapter_only_resume_checkpoint = str(pathlib.Path(adapter_only_resume_checkpoint))
        if not pathlib.Path(adapter_only_resume_checkpoint).is_dir():
            raise FileNotFoundError(
                f"resume_mm_projector_only_from directory does not exist: {adapter_only_resume_checkpoint}"
            )
        if not _has_mm_projector_checkpoint(adapter_only_resume_checkpoint):
            raise FileNotFoundError(
                f"resume_mm_projector_only_from does not contain mm_projector.bin: {adapter_only_resume_checkpoint}"
            )

    train_start = time.perf_counter()
    if latest_checkpoint is not None and (
        (training_args.deepspeed and _has_deepspeed_resume_state(latest_checkpoint))
        or (not training_args.deepspeed and _has_hf_resume_weights(latest_checkpoint))
    ):
        rank0_print(f"Resuming training from full checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    elif adapter_only_resume_checkpoint is not None:
        loaded_keys = _load_mm_projector_checkpoint(model, adapter_only_resume_checkpoint)
        rank0_print(
            f"Warm-starting from adapter-only checkpoint: {adapter_only_resume_checkpoint} "
            f"(loaded {loaded_keys} tensors, optimizer/scheduler state will restart)"
        )
        trainer.train()
    elif latest_checkpoint is not None and _has_mm_projector_checkpoint(latest_checkpoint):
        loaded_keys = _load_mm_projector_checkpoint(model, latest_checkpoint)
        rank0_print(
            f"Found adapter-only checkpoint in output_dir: {latest_checkpoint} "
            f"(loaded {loaded_keys} tensors, optimizer/scheduler state will restart)"
        )
        trainer.train()
    else:
        rank0_print("Starting training from scratch...")
        trainer.train()
    rank0_print(f"Training finished in {_elapsed(train_start)}")
    if data_args.max_samples_per_split > 0 and trainer.is_world_process_zero():
        # Retain proof of optimizer steps/losses without enabling heavyweight
        # optimizer-state checkpoints for short pipeline checks.
        smoke_report = {
            "global_step": trainer.state.global_step,
            "samples_per_split_limit": data_args.max_samples_per_split,
            "log_history": _json_safe(trainer.state.log_history),
        }
        with open(os.path.join(training_args.output_dir, "smoke_train_metrics.json"), "w") as handle:
            json.dump(smoke_report, handle, indent=2)
    # 只有显式需要断点续训时，才额外保存 Trainer 的恢复状态；
    # 默认 checkpoint 仅保留模型相关文件，避免 optimizer/scheduler 等状态占用额外空间。
    if training_args.save_resume_state:
        trainer.save_state()
    else:
        rank0_print("Skipping trainer resume state save because save_resume_state=False.")

    # 理论上 100% 训练进度时 callback 会再触发一次 validation。
    # 这里额外补一个保险：如果由于边界情况没有触发到最后一次，就在训练结束后手动补跑一次。
    if trainer.state.global_step not in quarter_validation_callback.triggered_steps:
        final_eval_start = time.perf_counter()
        rank0_print("Running final validation...")
        trainer.evaluate(metric_key_prefix="validation")
        rank0_print(f"Final validation finished in {_elapsed(final_eval_start)}")

    # test 不会再更新参数，所以先把最终模型参数落盘；
    # 这样即使后续 test 因为脏数据等原因失败，也不会丢掉最后一版训练结果。
    final_use_cache = model.config.use_cache
    model.config.use_cache = True
    rank0_print("Saving final model before final test...")
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    rank0_print("Final model save finished.")
    model.config.use_cache = final_use_cache

    # 训练完成后，用 test_dataset 单独跑一次最终测试。
    # metric_key_prefix="test" 会让结果字典中的 key 变成 test_loss / test_runtime 等。
    test_eval_start = time.perf_counter()
    rank0_print("Running final test evaluation...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    rank0_print(f"Final test evaluation finished in {_elapsed(test_eval_start)}")
    # 只在主进程保存 test 结果，避免多卡下重复写同一个文件。
    if trainer.is_world_process_zero():
        # 把最终 test 结果保存到固定文件名 test_final.json，并同步追加到 test_history.jsonl。
        save_split_metrics(
            output_dir=training_args.output_dir,
            split_name="test",
            metrics=test_metrics,
            global_step=trainer.state.global_step,
            filename="test_final.json",
        )

    # 训练完整结束后，再做一次兜底清理；
    # 如果目录里残留了旧版逻辑留下的恢复状态文件，这里统一删掉。
    latest_checkpoint = _find_latest_checkpoint_dir(training_args.output_dir)
    _distributed_barrier()
    if latest_checkpoint is not None and trainer.is_world_process_zero():
        _cleanup_checkpoint_resume_state(latest_checkpoint)
        rank0_print(f"Removed resume state from final checkpoint: {latest_checkpoint}")
    _distributed_barrier()


if __name__ == "__main__":
    random.seed(0)
    _train()
