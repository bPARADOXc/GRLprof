#    Copyright 2023 Haotian Liu
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

"""
  - 先读 model_path 的 config，判断是否需要 <GH> / </GH>
  - tokenizer 先补这两个 token
  - 但模型加载 base 权重时，临时用 model_base 自己的 vocab_size 去构建，避免 32000 -> 32002 的 shape mismatch
  - base 权重加载完后，立刻 resize_token_embeddings(len(tokenizer))
  - 然后再加载 mm_projector.bin 或 non_lora_trainables.bin

  关键位置在：

  - model/builder.py:28
  - model/builder.py:59
  - model/builder.py:93
  - model/builder.py:166
"""

import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from model import *
from utils.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download



# --- Added: prepare tokenizer/config for checkpoints with graph start/end tokens ---
def _requires_fast_tokenizer(model_path) -> bool:
    model_path = os.fspath(model_path).lower()
    return "llama-3" in model_path or "llama3" in model_path or "meta-llama-3" in model_path


def _load_auto_tokenizer(model_path, use_fast=False, **kwargs):
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast or _requires_fast_tokenizer(model_path),
        **kwargs,
    )


def _prepare_llaga_tokenizer_and_config(model_path, model_base, use_fast=False):
    tokenizer = _load_auto_tokenizer(model_base, use_fast=use_fast)
    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    base_vocab_size = AutoConfig.from_pretrained(model_base).vocab_size

    if getattr(cfg_pretrained, "mm_use_graph_start_end", False):
        tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)

    return tokenizer, cfg_pretrained, base_vocab_size


def _normalize_mm_projector_weights(state_dict):
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
# --- End added: prepare tokenizer/config for checkpoints with graph start/end tokens ---


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    cache_dir="../../checkpoint",
    max_memory=None,
    torch_dtype=None,
):
    kwargs = {"device_map": device_map}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    target_dtype = torch_dtype or torch.float16
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=target_dtype if target_dtype in (torch.float16, torch.bfloat16) else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = target_dtype

    model_name_lower = model_name.lower()
    is_graph_adapter_model = 'llaga' in model_name_lower or 'grlprof' in model_name_lower

    if is_graph_adapter_model:
        # Load the llaga-compatible graph adapter model.
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            # Old code kept for reference:
            # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            # tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # print('Loading LLaGA from base model...')
            # model = LlagaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            # token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            # if model.lm_head.weight.shape[0] != token_num:
            #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            # --- Added: load base weights with base vocab first, then expand to tokenizer size ---
            tokenizer, lora_cfg_pretrained, base_vocab_size = _prepare_llaga_tokenizer_and_config(
                model_path, model_base, use_fast=False
            )
            lora_cfg_pretrained.vocab_size = base_vocab_size
            print('Loading graph adapter from base model...')
            model = LlagaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
            # --- End added: load base weights with base vocab first, then expand to tokenizer size ---

            print('Loading additional graph adapter weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading graph adapter from base model...')
            # if 'mpt' in model_name.lower():
            #     if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
            #         shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
            #     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            #     cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            #     model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            #     cfg_pretrained = AutoConfig.from_pretrained(model_path)
            #     model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            if 'opt' in model_base:
                # Old code kept for reference:
                # tokenizer = AutoTokenizer.from_pretrained(model_base)
                # cfg_pretrained = AutoConfig.from_pretrained(model_path)
                # model = LlagaOPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                #                                               cache_dir=cache_dir,
                #                                               **kwargs)
                # --- Added: load base weights with base vocab first, then expand to tokenizer size ---
                tokenizer, cfg_pretrained, base_vocab_size = _prepare_llaga_tokenizer_and_config(
                    model_path, model_base
                )
                cfg_pretrained.vocab_size = base_vocab_size
                model = LlagaOPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                              cache_dir=cache_dir,
                                                              **kwargs)
                # --- End added: load base weights with base vocab first, then expand to tokenizer size ---
            else:
                # Old code kept for reference:
                # tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                # cfg_pretrained = AutoConfig.from_pretrained(model_path)
                # model = LlagaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir,
                #                                               **kwargs)
                # --- Added: load base weights with base vocab first, then expand to tokenizer size ---
                tokenizer, cfg_pretrained, base_vocab_size = _prepare_llaga_tokenizer_and_config(
                    model_path, model_base, use_fast=False
                )
                cfg_pretrained.vocab_size = base_vocab_size
                model = LlagaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir,
                                                              **kwargs)
                # --- End added: load base weights with base vocab first, then expand to tokenizer size ---
            if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
            # model.get_model().initialize_graph_modules(cfg_pretrained)
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                print("Load from local path")
            else:
                from huggingface_hub import hf_hub_download
                model_path_hf = hf_hub_download(repo_id=model_path,  filename='mm_projector.bin')
                mm_projector_weights = torch.load(model_path_hf, map_location='cpu')
                print("Load from huggingface")
            mm_projector_weights = _normalize_mm_projector_weights(mm_projector_weights)
            mm_projector_weights = {k: v.to(target_dtype) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # if 'mpt' in model_name.lower():
            #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            #     model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            #     model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            tokenizer = _load_auto_tokenizer(model_path, use_fast=False)
            model = LlagaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = _load_auto_tokenizer(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=target_dtype, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print(f'Convert to {target_dtype}...')
            model.to(target_dtype)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = _load_auto_tokenizer(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = _load_auto_tokenizer(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, cache_dir=cache_dir, **kwargs)


    if is_graph_adapter_model:
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        # Old code kept for reference:
        # if mm_use_graph_start_end:
        #     tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        # model.resize_token_embeddings(len(tokenizer))
        # --- Added: avoid double-adding tokens and resize only when vocab/tokenizer still differ ---
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            if mm_use_graph_start_end:
                tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
        # --- End added: avoid double-adding tokens and resize only when vocab/tokenizer still differ ---

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
