import os
import random
import shutil

import numpy as np
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    # 处理 DeepSpeed ZeRO-3 优化下的参数获取。
    # ZeRO-3 会将模型参数分片存储在不同的 GPU 上以节省显存。
    # 当需要保存模型或者获取完整参数时，需要使用此函数将分散的参数片聚合(Gather)到当前设备(通常是CPU)。
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        # 使用 GatheredParameters 上下文管理器从所有 GPU 收集参数
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    # 专门用于获取多模态适配器 (Projector) 相关的参数。
    # 筛选出名字中包含 keys_to_match 的参数，并处理 ZeRO-3 分片情况。
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # 将数据索引列表分成 num_chunks 个块，尽量使每个块中样本的总长度（Token数）大致相等。
    # 这在多 GPU 训练中用于负载均衡，防止某个 GPU 分到的数据总长度过长而等待。

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # 基于"模态"和"长度"对样本索引进行分组。
    # 这里的 lengths 列表中，正数表示一种模态（如含图数据），负数表示另一种模态（如纯文本），绝对值表示长度。
    # 这种区分是在 Dataset 的 modality_lengths 属性中定义的。
    # 目的：确保同一个 batch 内的样本要么全是多模态，要么全是纯文本，且长度相近，减少 Padding。
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    # 分别对两种模态的数据进行按长度分组
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    
    # 将分组后的数据按 mega-batch (world_size * batch_size) 组织
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    # 处理剩余的不足一个 mega-batch 的数据
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    # 随机打乱 mega-batches 的顺序，保证训练过程中的随机性
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # 核心的分组逻辑：
    # 1. 对所有索引进行随机打乱（基于 generator）。
    # 2. 将索引切分为 Mega-batches。
    # 3. 在每个 Mega-batch 内部，按长度从长到短排序。
    # 4. 将排序后的 Mega-batch 再均分给各个 GPU (world_size) 的 batch。
    # 结果：每个 GPU 获取到的 batch 内样本长度是相近的。
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """
    # 自定义采样器：在保持一定随机性的同时，将长度相近的样本组合在一起。
    # 这可以显著减少 Batch Padding 的开销，提高 Transformer 模型的训练速度。

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaGATrainer(Trainer):
    # 自定义 Trainer 类，继承自 HuggingFace 的 Trainer。
    # 主要修改了数据采样逻辑 (Sampler) 和模型保存逻辑 (Checkpoint Saving)。

    @staticmethod
    def _checkpoint_step(checkpoint_dir: str) -> int:
        try:
            return int(os.path.basename(checkpoint_dir).split("-")[-1])
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _dist_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _list_checkpoint_dirs(self, run_dir: str):
        checkpoint_dirs = []
        if not os.path.isdir(run_dir):
            return checkpoint_dirs

        for entry in os.listdir(run_dir):
            checkpoint_dir = os.path.join(run_dir, entry)
            if entry.startswith("checkpoint-") and os.path.isdir(checkpoint_dir):
                checkpoint_dirs.append(checkpoint_dir)
        checkpoint_dirs.sort(key=self._checkpoint_step)
        return checkpoint_dirs

    def _cleanup_checkpoint_resume_state(self, checkpoint_dir: str):
        if not os.path.isdir(checkpoint_dir):
            return

        for entry in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, entry)
            if entry.startswith("global_step") and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

        removable_files = (
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        )
        for filename in removable_files:
            path = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(path):
                os.remove(path)

        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("rng_state") and entry.endswith(".pth"):
                path = os.path.join(checkpoint_dir, entry)
                if os.path.isfile(path):
                    os.remove(path)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # 获取训练时的采样器。
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # 如果开启了按模态长度分组的选项 (通常在训练多模态模型时开启)
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            # 使用自定义的 LengthGroupedSampler
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # 否则使用父类默认的采样器
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        # 重写保存 Checkpoint 的逻辑。
        # 目的是在做 Parameter-Efficient Fine-Tuning (PEFT) 时，只保存微调部分的参数，节省空间。
        
        # 如果配置了只微调多模态适配器 (Projector)
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            save_resume_state = getattr(self.args, "save_resume_state", False)
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            older_checkpoint_dirs = [
                checkpoint_dir
                for checkpoint_dir in self._list_checkpoint_dirs(run_dir)
                if os.path.abspath(checkpoint_dir) != os.path.abspath(output_dir)
            ]

            # 如果这个 step 之前保存失败过，先清掉当前目录里残留的恢复状态，
            # 但保留 config/mm_projector，避免旧的半截 DeepSpeed 文件影响本次保存。
            self._cleanup_checkpoint_resume_state(output_dir)

            # Only save Adapter
            # 定义需要保存的参数关键字
            keys_to_match = ['mm_projector', 'graph_path_encoder']
            if getattr(self.args, "mm_use_graph_start_end", False):
                keys_to_match.extend(['graph_start_end_emb']) # 只保存独立的图起止 token 参数
            if getattr(self.args, "mm_use_graph_special_token", False):
                keys_to_match.extend(['special_token_emb'])

            # 获取并聚合参数 (处理 ZeRO-3)
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
                if save_resume_state:
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            if save_resume_state:
                if self.is_deepspeed_enabled:
                    # 显式开启断点续训时，才保存 DeepSpeed 的优化器/调度器状态。
                    self.model_wrapped.save_checkpoint(output_dir)

                rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "cpu": torch.random.get_rng_state(),
                }
                if torch.cuda.is_available():
                    if self.args.world_size <= 1:
                        rng_states["cuda"] = torch.cuda.random.get_rng_state()
                    else:
                        rng_states["cuda"] = torch.cuda.random.get_rng_state_all()

                os.makedirs(output_dir, exist_ok=True)
                if self.args.world_size <= 1:
                    torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
                else:
                    torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

            # 默认 checkpoint 只保留模型文件；这里顺手清理当前目录和旧目录中残留的恢复状态，
            # 兼容之前版本或中断保存留下的 global_step*/optimizer.pt/trainer_state.json 等文件。
            self._dist_barrier()
            if self.args.process_index == 0:
                self._cleanup_checkpoint_resume_state(output_dir)
                for checkpoint_dir in older_checkpoint_dirs:
                    self._cleanup_checkpoint_resume_state(checkpoint_dir)
            self._dist_barrier()
        else:
            # 正常全量微调或 LoRA 微调 (LoRA 有自己的保存机制，但如果是全量，则走默认逻辑)
            super(LLaGATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 如果只微调适配器，这里可能会跳过默认的保存逻辑（因为 _save_checkpoint 已经处理了自定义保存）
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaGATrainer, self)._save(output_dir, state_dict)
