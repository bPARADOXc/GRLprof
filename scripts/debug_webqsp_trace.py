#!/usr/bin/env python3
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from model.language_model.llaga_llama import LlagaConfig, LlagaLlamaForCausalLM
from utils import conversation as conversation_lib
from utils.constants import GRAPH_TOKEN_INDEX, IGNORE_INDEX


# 直接从文件路径加载 train/train.py，避免脚本模式下与 train 包名解析冲突。
TRAIN_SPEC = importlib.util.spec_from_file_location("llaga_train_module", ROOT / "train" / "train.py")
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


def parse_args():
    parser = argparse.ArgumentParser(description="Trace the WebQSP training pipeline step by step.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--meta-llama-root", default=str(ROOT / "llama"))
    parser.add_argument("--meta-llama-model-dir", default=str(ROOT / "llama" / "llama-2-7b-chat"))
    parser.add_argument("--use-meta-llama", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
    )
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument(
        "--max-debug-layers",
        type=int,
        default=2,
        help="When using local Meta Llama params, cap the instantiated layer count to keep tracing affordable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-backward", action="store_true")
    return parser.parse_args()


def describe_graph_emb(graph_emb):
    # WebQSP QA 分支里的 graph_emb 是嵌套 list，不是规则二维张量；
    # 这里只做轻量摘要，便于快速确认“路径数/每条路径 triple 数/单个 triple 维度”。
    if not isinstance(graph_emb, list):
        return f"type={type(graph_emb).__name__}"
    path_count = len(graph_emb)
    triple_counts = [len(path) for path in graph_emb]
    triple_shape = None
    if path_count > 0 and triple_counts[0] > 0 and torch.is_tensor(graph_emb[0][0]):
        triple_shape = tuple(graph_emb[0][0].shape)
    return (
        f"paths={path_count}, "
        f"triples_per_path={triple_counts}, "
        f"first_triple_shape={triple_shape}"
    )


def recursive_to_device(value, device):
    # batch 中 graph_emb 是多层 list + tensor 的混合结构，不能直接 .to(device)。
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: recursive_to_device(item, device) for key, item in value.items()}
    return value


def infer_use_meta_llama(args):
    return (
        args.use_meta_llama
        or (
            Path(args.meta_llama_root).exists()
            and Path(args.meta_llama_model_dir).exists()
            and args.tokenizer_path is None
        )
    )


def load_meta_llama_params(model_dir):
    params = json.loads((Path(model_dir) / "params.json").read_text())
    dim = params["dim"]
    multiple_of = params["multiple_of"]
    hidden_dim = int(2 * (4 * dim) / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return {
        "hidden_size": dim,
        "intermediate_size": hidden_dim,
        "num_attention_heads": params["n_heads"],
        "num_hidden_layers": params["n_layers"],
        "rms_norm_eps": params.get("norm_eps", 1e-6),
        "vocab_size": params.get("vocab_size", -1),
    }


def load_tokenizer(args, use_meta_llama):
    if use_meta_llama:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(args.meta_llama_root)
    else:
        tokenizer_path = args.tokenizer_path
        if tokenizer_path is None:
            raise ValueError("Please provide --tokenizer-path, or enable local Meta Llama via --use-meta-llama.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.model_max_length = args.model_max_length
    return tokenizer


def install_trace_hooks(model):
    # 通过 monkey-patch 在不改主训练逻辑的前提下，观察图编码和图注入的关键中间结果。
    encoder = model.get_model().graph_path_encoder

    original_encode_path = encoder.encode_path
    original_forward = encoder.forward
    original_encode_graphs = model.encode_graphs
    original_prepare = model.prepare_inputs_labels_for_multimodal
    path_counter = {"value": 0}

    def traced_encode_path(self, triples):
        # 单条 reasoning path 的编码细节：输入若干 triple，输出一个 path token。
        path_idx = path_counter["value"]
        triple_shapes = [tuple(triple.shape) for triple in triples]
        print(
            f"[GraphPathEncoder.encode_path] path={path_idx} "
            f"triple_count={len(triples)} triple_shapes={triple_shapes}"
        )
        output = original_encode_path(triples)
        print(f"[GraphPathEncoder.encode_path] path={path_idx} output_shape={tuple(output.shape)}")
        path_counter["value"] += 1
        return output

    def traced_forward(self, sample_paths):
        # 一个样本包含多条 path；这里验证 path 数量和最终 stack 后的形状。
        path_counter["value"] = 0
        print(
            "[GraphPathEncoder.forward] "
            f"sample_path_count={len(sample_paths)} "
            f"triples_per_path={[len(path) for path in sample_paths]}"
        )
        output = original_forward(sample_paths)
        print(f"[GraphPathEncoder.forward] stacked_path_tokens_shape={tuple(output.shape)}")
        return output

    def traced_encode_graphs(self, graphs, graph_emb):
        # batch 级别观察：每个样本各自有多少条 reasoning path。
        print(f"[encode_graphs] batch_graphs={graphs}")
        for sample_idx, sample_paths in enumerate(graph_emb):
            print(
                f"[encode_graphs] sample={sample_idx} "
                f"path_count={len(sample_paths)} "
                f"triples_per_path={[len(path) for path in sample_paths]}"
            )
        outputs = original_encode_graphs(graphs, graph_emb)
        for sample_idx, sample_output in enumerate(outputs):
            print(f"[encode_graphs] sample={sample_idx} encoded_shape={tuple(sample_output.shape)}")
        return outputs

    def traced_prepare(self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb):
        # 这是多模态链路里最关键的一步：
        # 把文本中的 <graph> 占位符替换成若干 path token，并同步扩展 mask / labels。
        print(
            "[prepare_inputs_labels_for_multimodal] "
            f"input_ids={tuple(input_ids.shape)} "
            f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None} "
            f"labels={tuple(labels.shape) if labels is not None else None}"
        )
        for sample_idx, sample_ids in enumerate(input_ids):
            graph_positions = (sample_ids == GRAPH_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
            non_ignore = None
            if labels is not None:
                non_ignore = int((labels[sample_idx] != IGNORE_INDEX).sum().item())
            print(
                f"[prepare_inputs_labels_for_multimodal] sample={sample_idx} "
                f"text_seq_len={sample_ids.shape[0]} graph_token_positions={graph_positions} "
                f"label_tokens={non_ignore}"
            )
        result = original_prepare(input_ids, attention_mask, past_key_values, labels, graphs, graph_emb)
        _, new_attention_mask, _, new_input_embeds, new_labels = result
        print(
            "[prepare_inputs_labels_for_multimodal] "
            f"new_input_embeds={tuple(new_input_embeds.shape)} "
            f"new_attention_mask={tuple(new_attention_mask.shape) if new_attention_mask is not None else None} "
            f"new_labels={tuple(new_labels.shape) if new_labels is not None else None}"
        )
        if new_labels is not None:
            for sample_idx, sample_labels in enumerate(new_labels):
                valid_labels = int((sample_labels != IGNORE_INDEX).sum().item())
                print(
                    f"[prepare_inputs_labels_for_multimodal] sample={sample_idx} "
                    f"post_injection_valid_labels={valid_labels}"
                )
        return result

    encoder.encode_path = MethodType(traced_encode_path, encoder)
    encoder.forward = MethodType(traced_forward, encoder)
    model.encode_graphs = MethodType(traced_encode_graphs, model)
    model.prepare_inputs_labels_for_multimodal = MethodType(traced_prepare, model)


def print_raw_sample(dataset, index):
    # 先看原始样本，方便把 trace 和具体 question / answer / reasoning path 数对应起来。
    sample = dataset.list_data_dict[index]
    answers = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]
    preview_answers = answers[:5]
    print(
        f"[raw sample] idx={index} id={sample['id']} "
        f"question={sample['question']!r} "
        f"answer_count={len(answers)} answers_preview={preview_answers}"
    )
    print(
        f"[raw sample] reasoning_paths={sample.get('num_reasoning_paths')} "
        f"graph_lookup_key={sample.get('graph', sample['id'])}"
    )


def print_item_trace(index, item):
    # 再看 __getitem__ 之后的结果：这里已经经过 prompt 拼接、tokenize 和 graph_emb 查表。
    graph_positions = (item["input_ids"] == GRAPH_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
    valid_labels = int((item["labels"] != IGNORE_INDEX).sum().item())
    print(
        f"[dataset __getitem__] idx={index} "
        f"input_ids={tuple(item['input_ids'].shape)} labels={tuple(item['labels'].shape)} "
        f"graph_positions={graph_positions} valid_labels={valid_labels}"
    )
    print(
        f"[dataset __getitem__] idx={index} graph={item['graph']} "
        f"graph_emb={describe_graph_emb(item['graph_emb'])}"
    )


def build_debug_model(vocab_size, tokenizer, args):
    # 这里构造的是“小尺寸随机初始化”的调试模型：
    # 目标是离线验证数据流和 shape，而不是复现真实 LLaMA 权重表现。
    config = LlagaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        max_position_embeddings=args.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rms_norm_eps=getattr(args, "rms_norm_eps", 1e-6),
    )
    config.mm_hidden_size = 768
    config.use_hop = 0
    config.sample_neighbor_size = 10
    config.mm_use_graph_special_token = False
    config.mm_use_graph_start_end = False
    model = LlagaLlamaForCausalLM(config)
    # mm_hidden_size=768 对应当前 E5 图向量输入维度；
    # output_dim 会自动跟随 hidden_size，这样图 token 才能和文本 embedding 拼接。
    model.get_model().initialize_graph_modules(
        model_args=SimpleNamespace(mm_projector_type="2-layer-mlp", mm_hidden_size=768),
        fsdp=None,
    )
    model.config.use_cache = False
    install_trace_hooks(model)
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    conversation_lib.default_conversation = conversation_lib.conv_templates["llaga_llama_2"]

    use_meta_llama = infer_use_meta_llama(args)
    if use_meta_llama:
        meta_params = load_meta_llama_params(args.meta_llama_model_dir)
        args.hidden_size = meta_params["hidden_size"]
        args.intermediate_size = meta_params["intermediate_size"]#前馈层的中间维度
        args.num_attention_heads = meta_params["num_attention_heads"]
        args.rms_norm_eps = meta_params["rms_norm_eps"]#RMSNorm 里的 epsilon（极小常数），主要作用是 防止数值不稳定或除零
        real_num_hidden_layers = meta_params["num_hidden_layers"]#几层block
        if args.max_debug_layers is not None:
            args.num_hidden_layers = min(real_num_hidden_layers, args.max_debug_layers)
        else:
            args.num_hidden_layers = real_num_hidden_layers
    else:
        args.rms_norm_eps = 1e-6

    # 优先使用本地 Meta Llama tokenizer；这样 <s>/<\s> 的分词行为会更接近真实训练。
    tokenizer = load_tokenizer(args=args, use_meta_llama=use_meta_llama)

    data_args = train_mod.DataArguments(
        lazy_preprocess=True,
        is_multimodal=True,
        pretrained_embedding_type="E5-base",
        use_hop=0,
        sample_neighbor_size=10,
        use_task="QA",
        use_dataset="webqsp",
        template="QA",
    )

    # 直接复用正式训练的数据集和 collator，确保调试路径与训练入口保持一致。
    dataset = train_mod.LazySupervisedGraphDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        datasets=args.split,
    )
    collator = train_mod.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    indices = args.indices if args.indices else list(range(min(args.num_samples, len(dataset))))
    print(f"[setup] split={args.split} dataset_size={len(dataset)} selected_indices={indices}")
    print(f"[setup] tokenizer={type(tokenizer).__name__} pad={tokenizer.pad_token_id} bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id}")
    if use_meta_llama:
        print(
            f"[setup] local_meta_llama hidden_size={args.hidden_size} "
            f"intermediate_size={args.intermediate_size} heads={args.num_attention_heads} "
            f"layers_for_trace={args.num_hidden_layers} real_layers={real_num_hidden_layers}"
        )

    instances = []
    max_seen_token_id = 0
    for index in indices:
        print_raw_sample(dataset, index)
        item = dataset[index]
        print_item_trace(index, item)
        instances.append(item)
        non_graph_tokens = item["input_ids"][item["input_ids"] >= 0]
        if non_graph_tokens.numel() > 0:
            max_seen_token_id = max(max_seen_token_id, int(non_graph_tokens.max().item()))

    # 这里只取前 batch_size 个样本做一次前向，避免调试输出过长。
    batch = collator(instances[: args.batch_size])
    print(
        f"[collator] input_ids={tuple(batch['input_ids'].shape)} "
        f"labels={tuple(batch['labels'].shape)} attention_mask={tuple(batch['attention_mask'].shape)}"
    )
    for sample_idx, (graph, graph_emb) in enumerate(zip(batch["graph"], batch["graph_emb"])):
        print(
            f"[collator] sample={sample_idx} graph={graph} graph_emb={describe_graph_emb(graph_emb)}"
        )

    special_ids = [
        value for value in [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        ] if value is not None
    ]
    vocab_size = max([max_seen_token_id] + special_ids) + 8
    print(f"[setup] debug_model_vocab_size={vocab_size}")

    model = build_debug_model(vocab_size=vocab_size, tokenizer=tokenizer, args=args)
    model.train()

    batch = recursive_to_device(batch, torch.device("gpu"))
    # 真正触发完整链路：
    # tokenizer 后的文本 -> graph path encoder -> <graph> 注入 -> LLM forward -> loss。
    outputs = model(**batch)
    print(
        f"[forward] logits={tuple(outputs.logits.shape)} "
        f"loss={float(outputs.loss.detach().cpu().item()):.6f}"
    )

    if args.run_backward:
        # backward 用来验证梯度是否能回流到图编码器，而不只是看前向 shape。
        outputs.loss.backward()
        grad_names = [
            "model.graph_path_encoder.position_fc.weight",
            "model.graph_path_encoder.text_linear.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        named_params = dict(model.named_parameters())
        for name in grad_names:
            grad = named_params[name].grad
            grad_norm = None if grad is None else float(grad.norm().detach().cpu().item())
            grad_shape = None if grad is None else tuple(grad.shape)
            print(f"[backward] {name} grad_shape={grad_shape} grad_norm={grad_norm}")


if __name__ == "__main__":
    main()
